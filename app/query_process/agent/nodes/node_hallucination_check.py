import sys
import json
import re

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.conf.query_threshold_config import query_threshold_config

cfg = query_threshold_config
HALLUCINATION_MAX_RETRIES = cfg.hallucination_max_retries


def _bool_from_state(state: dict, key: str, default: bool = False) -> bool:
    value = default
    for container in (
        state,
        state.get("route_overrides") or {},
        state.get("evaluation_overrides") or {},
    ):
        if isinstance(container, dict) and key in container:
            value = bool(container.get(key))
    return value


def _needs_llm_hallucination_check(state: dict, answer: str, reranked_docs: list) -> bool:
    if not _bool_from_state(state, "anchor_context_enabled", False):
        return True
    if not bool(state.get("grounded_mode", False)):
        return True

    reference_blob = "\n".join(str(doc.get("text") or "") for doc in reranked_docs)
    if not reranked_docs:
        return "抱歉，资料中未提供" not in answer

    target_coverage = state.get("target_coverage") or {}
    missing_targets = target_coverage.get("missing_targets") or []
    if missing_targets and "抱歉，资料中未提供" not in answer:
        return True

    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?%?|[A-Za-z]{1,8}\d[A-Za-z0-9._/-]*", answer))
    if answer_numbers:
        reference_numbers = set(
            re.findall(r"\d+(?:\.\d+)?%?|[A-Za-z]{1,8}\d[A-Za-z0-9._/-]*", reference_blob)
        )
        unsupported = [value for value in answer_numbers if value not in reference_numbers]
        if unsupported:
            return True

    return False


def step_1_check_hallucination(question: str, answer: str, reranked_docs: list) -> dict:
    """
    阶段1：利用LLM检查答案是否存在幻觉
    :param question: 用户问题
    :param answer: LLM 生成的答案
    :param reranked_docs: 参考文档列表
    :return: {"passed": bool, "hallucinations": str}
    """
    logger.info("Step 1: 开始幻觉检查")

    # 构造文档摘要（限制长度，避免 Prompt 过长）
    doc_summaries = []
    total_len = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        summary = f"[{i}] {text[:cfg.grader_doc_max_chars]}"
        if total_len + len(summary) > 6000:
            break
        doc_summaries.append(summary)
        total_len += len(summary)

    documents_str = "\n\n".join(doc_summaries) if doc_summaries else "无参考文档"

    try:
        client = get_llm_client(json_mode=True)
        prompt = load_prompt(
            "hallucination_check",
            question=question,
            documents=documents_str,
            answer=answer,
        )

        response = client.invoke(prompt)
        content = coerce_llm_content(response.content)

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)

        passed_raw = result.get("passed", True)
        if isinstance(passed_raw, bool):
            passed = passed_raw
        elif isinstance(passed_raw, str):
            passed = passed_raw.strip().lower() in {"true", "1", "yes", "是"}
        else:
            passed = True

        hallucinations = result.get("hallucinations", "无")

        logger.info(f"Step 1: 幻觉检查结果 passed={passed}, detail={hallucinations}")
        return {"passed": passed, "hallucinations": hallucinations}

    except Exception as e:
        logger.exception("Step 1: 幻觉检查失败")
        # 降级：检查失败时默认通过
        return {"passed": True, "hallucinations": f"检查失败(降级通过): {e}"}


def node_hallucination_check(state):
    """
    答案幻觉自检节点

    功能：
    1. 对 LLM 生成的答案进行事实一致性验证，检查是否存在与参考文档不一致的幻觉
    2. 若检查通过 → 流程正常结束
    3. 若检查未通过且可重试 → 清空答案，设置反馈信息，触发重新生成
    4. 若检查未通过但无法重试（流式模式 / 重试耗尽）→ 记录日志，流程结束

    跳过条件：
    - 非 RAG 模式（通用对话/商品名澄清）
    - 无参考文档
    - 无答案

    :param state: 查询流程全局状态
    :return: 更新后的状态字段
    """
    logger.info("---node_hallucination_check (幻觉自检) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    answer = (state.get("answer") or "").strip()
    reranked_docs = state.get("reranked_docs") or []
    question = state.get("rewritten_query") or state.get("original_query", "")
    is_stream = state.get("is_stream", False)
    hallucination_retry_count = state.get("hallucination_retry_count", 0)

    # 跳过条件检查
    skip_reason = None
    if not state.get("need_rag"):
        skip_reason = "非RAG模式"
    elif not reranked_docs:
        skip_reason = "无参考文档"
    elif not answer:
        skip_reason = "无答案"

    if skip_reason:
        logger.info(f"幻觉检查跳过: {skip_reason}")
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {"hallucination_check_passed": True}

    if not _needs_llm_hallucination_check(state, answer, reranked_docs):
        logger.info("幻觉检查轻量规则通过，跳过 LLM 检查")
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {
            "hallucination_check_passed": True,
            "hallucination_feedback": "",
        }

    # 阶段1：执行幻觉检查
    check_result = step_1_check_hallucination(question, answer, reranked_docs)
    passed = check_result.get("passed", True)

    result = {}

    if passed:
        logger.info("幻觉检查通过，答案事实一致性合格")
        result["hallucination_check_passed"] = True

    elif is_stream:
        # 流式模式下答案已推送，无法重新生成，仅记录警告
        logger.warning(
            f"幻觉检查未通过(流式模式，无法重试): {check_result.get('hallucinations')}"
        )
        result["hallucination_check_passed"] = True  # 标记为通过以结束流程

    elif hallucination_retry_count >= HALLUCINATION_MAX_RETRIES:
        # 重试耗尽
        logger.warning(
            f"幻觉检查未通过(重试耗尽): {check_result.get('hallucinations')}"
        )
        result["hallucination_check_passed"] = True  # 标记为通过以结束流程

    else:
        # 可重试：清空答案，设置反馈，触发重新生成
        feedback = check_result.get("hallucinations", "存在事实不一致")
        logger.info(f"幻觉检查未通过，触发重新生成 (feedback: {feedback})")
        result["hallucination_check_passed"] = False
        result["answer"] = ""  # 清空答案，使 node_answer_output 重新生成
        result["hallucination_feedback"] = feedback
        result["hallucination_retry_count"] = hallucination_retry_count + 1

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info(
        f"---node_hallucination_check 处理结束, passed={result.get('hallucination_check_passed')}---"
    )
    return result


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_hallucination_check 本地测试")
    print("=" * 50)

    # 测试1：答案与文档一致（应通过）
    mock_state_pass = {
        "session_id": "test_hallucination_001",
        "rewritten_query": "HAK 180 烫金机的默认温度设置是多少？",
        "answer": "HAK 180 烫金机的默认温度建议设置在 110℃ 左右。开机后通过操作面板进行设置。",
        "reranked_docs": [
            {
                "text": "HAK 180 烫金机开启电源后，先设置温度，默认建议设置在 110℃ 左右。",
                "score": 0.95,
                "source": "local",
            },
        ],
        "need_rag": True,
        "is_stream": False,
        "hallucination_retry_count": 0,
    }

    # 测试2：答案存在幻觉（编造了不存在的参数）
    mock_state_fail = {
        "session_id": "test_hallucination_002",
        "rewritten_query": "HAK 180 烫金机的默认温度设置是多少？",
        "answer": "HAK 180 烫金机的默认温度为 150℃，最高可达 350℃，配备智能温控芯片自动调节。",
        "reranked_docs": [
            {
                "text": "HAK 180 烫金机开启电源后，先设置温度，默认建议设置在 110℃ 左右。",
                "score": 0.95,
                "source": "local",
            },
        ],
        "need_rag": True,
        "is_stream": False,
        "hallucination_retry_count": 0,
    }

    # 测试3：非RAG模式（应跳过）
    mock_state_skip = {
        "session_id": "test_hallucination_003",
        "rewritten_query": "你好",
        "answer": "你好！有什么可以帮助您的？",
        "reranked_docs": [],
        "need_rag": False,
        "is_stream": False,
        "hallucination_retry_count": 0,
    }

    for name, state in [
        ("答案一致(应通过)", mock_state_pass),
        ("存在幻觉(应未通过)", mock_state_fail),
        ("非RAG(应跳过)", mock_state_skip),
    ]:
        try:
            print(f"\n>>> 测试场景: {name}")
            result = node_hallucination_check(state)
            print(f"hallucination_check_passed: {result.get('hallucination_check_passed')}")
            if result.get("hallucination_feedback"):
                print(f"feedback: {result.get('hallucination_feedback')}")
            print("-" * 30)
        except Exception as e:
            logger.exception(f"测试 [{name}] 失败: {e}")

    print("=" * 50)
