import sys
import json

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.conf.query_threshold_config import query_threshold_config
from app.query_process.agent.agentic_utils import (
    build_retrieval_rescue_plan,
    get_agentic_features,
)

cfg = query_threshold_config

# CRAG 全局常量（已迁移到 query_threshold_config）
CRAG_MAX_RETRIES = cfg.crag_max_retries
CRAG_MIN_DOCS = cfg.crag_min_docs


def step_1_grade_retrieval(question: str, reranked_docs: list) -> dict:
    """
    阶段1：利用LLM评估检索结果质量
    :param question: 用户改写后的问题
    :param reranked_docs: 重排序后的文档列表
    :return: {"grade": str, "reason": str, "suggested_query": str}
    """
    logger.info(f"Step 1: 开始评估检索质量, 文档数: {len(reranked_docs)}")

    # 快速检查：无文档直接判定不足
    if not reranked_docs:
        logger.warning("Step 1: 无检索文档，直接判定 insufficient")
        return {
            "grade": "insufficient",
            "reason": "无检索结果",
            "suggested_query": question,
        }

    # 构造文档摘要（限制长度，避免 Prompt 过长）
    doc_summaries = []
    total_len = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (doc.get("text") or "").strip()
        score = doc.get("score")
        source = doc.get("source", "unknown")
        summary = f"[{i}] [source={source}]"
        if score is not None:
            summary += f" [score={float(score):.4f}]"
        summary += f"\n{text[:cfg.grader_doc_max_chars]}"  # 截断过长文本
        if total_len + len(summary) > 6000:
            break
        doc_summaries.append(summary)
        total_len += len(summary)

    documents_str = "\n\n".join(doc_summaries)

    try:
        client = get_llm_client(json_mode=True)
        prompt = load_prompt(
            "retrieval_grader", question=question, documents=documents_str
        )

        response = client.invoke(prompt)
        content = coerce_llm_content(response.content)

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        grade = result.get("grade", "sufficient")
        reason = result.get("reason", "")
        suggested_query = result.get("suggested_query", "")

        logger.info(f"Step 1: 评估结果 grade={grade}, reason={reason}")
        return {
            "grade": grade,
            "reason": reason,
            "suggested_query": suggested_query,
        }

    except Exception as e:
        logger.exception("Step 1: 检索质量评估失败")
        # 降级：评估失败时默认通过（避免阻塞主流程）
        return {
            "grade": "sufficient",
            "reason": f"评估失败(降级通过): {e}",
            "suggested_query": "",
        }


def node_retrieval_grader(state):
    """
    CRAG（Corrective RAG）检索质量判断节点

    功能：
    1. 对 rerank 后的文档进行质量评估，判断是否足以回答用户问题
    2. 若评估为 sufficient → 直接进入答案生成
    3. 若评估为 insufficient 且未超过重试上限 → 改写查询，回退重新检索
    4. 若评估为 insufficient 且已达上限 → 标记低置信度，继续生成（带caveat）

    :param state: 查询流程全局状态
    :return: 更新后的状态字段
    """
    logger.info("---node_retrieval_grader (CRAG 检索质量判断) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    question = state.get("rewritten_query") or state.get("original_query", "")
    reranked_docs = state.get("reranked_docs") or []
    retry_count = state.get("retry_count", 0)
    evidence_coverage = state.get("evidence_coverage_summary") or {}
    agentic_features = get_agentic_features(state)

    # 阶段1：评估检索质量
    grade_result = step_1_grade_retrieval(question, reranked_docs)
    grade = grade_result.get("grade", "sufficient")
    rescue_plan = build_retrieval_rescue_plan(state, grade_result)
    needs_rescue = bool(evidence_coverage.get("needs_rescue")) and bool(
        agentic_features.get("retrieval_rescue", True)
    )

    result = {
        "rescue_plan": rescue_plan,
    }

    if rescue_plan.get("action") == "clarify":
        logger.info(
            "CRAG: 触发澄清守卫, reason=%s", rescue_plan.get("reason", "clarify")
        )
        result["retrieval_grade"] = "insufficient"
        result["answer"] = rescue_plan.get("clarification_question") or (
            "当前问题还缺少关键条件，请补充后我再继续检索。"
        )
        result["clarification_reason"] = rescue_plan.get("reason") or ""

    elif grade == "sufficient" and not needs_rescue:
        # 检索质量合格，继续流程
        logger.info(f"CRAG: 检索质量合格 (reason: {grade_result.get('reason')})")
        result["retrieval_grade"] = "sufficient"

    elif rescue_plan.get("action") == "retry" and retry_count < CRAG_MAX_RETRIES:
        # 检索质量不足 + 还有重试机会 → 进入 Agentic 补救重试
        new_query = rescue_plan.get("question") or question

        logger.info(
            f"CRAG: 检索质量不足，触发重试 (retry {retry_count + 1}/{CRAG_MAX_RETRIES}), "
            f"新查询: {new_query}，补救步骤: {rescue_plan.get('steps')}"
        )
        result["retrieval_grade"] = "retry"
        result["rewritten_query"] = new_query
        result["retry_count"] = retry_count + 1
        result["route_overrides"] = rescue_plan.get("route_overrides") or {}
        result["embedding_chunks"] = []
        result["hyde_embedding_chunks"] = []
        result["bm25_chunks"] = []
        result["kg_chunks"] = []
        result["rrf_chunks"] = []
        result["reranked_docs"] = []
        result["web_search_docs"] = []
        result["sub_query_routes"] = []
        result["sub_query_results"] = []
        result["context_expansion_summary"] = {}
        result["evidence_coverage_summary"] = {}
        result["answer_plan"] = {}
        if result["route_overrides"].get("drop_item_names"):
            result["item_names"] = []

    elif grade != "sufficient" and retry_count < CRAG_MAX_RETRIES:
        suggested_query = grade_result.get("suggested_query", "").strip()
        new_query = suggested_query if suggested_query else question
        logger.info(
            f"CRAG: 使用旧版重试策略 (retry {retry_count + 1}/{CRAG_MAX_RETRIES}), "
            f"新查询: {new_query}"
        )
        result["retrieval_grade"] = "retry"
        result["rewritten_query"] = new_query
        result["retry_count"] = retry_count + 1
        result["embedding_chunks"] = []
        result["hyde_embedding_chunks"] = []
        result["bm25_chunks"] = []
        result["kg_chunks"] = []
        result["rrf_chunks"] = []
        result["reranked_docs"] = []
        result["web_search_docs"] = []
        result["sub_query_routes"] = []
        result["sub_query_results"] = []
        result["context_expansion_summary"] = {}
        result["evidence_coverage_summary"] = {}
        result["answer_plan"] = {}

    else:
        # 检索质量不足 + 重试耗尽 → 标记低置信度，继续生成
        logger.warning(
            f"CRAG: 检索质量不足且重试耗尽 (reason: {grade_result.get('reason')})"
        )
        result["retrieval_grade"] = "insufficient"

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info(
        f"---node_retrieval_grader 处理结束, grade={result.get('retrieval_grade')}---"
    )
    return result


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_retrieval_grader 本地测试")
    print("=" * 50)

    # 测试1：充分的检索结果
    mock_state_sufficient = {
        "session_id": "test_grader_001",
        "rewritten_query": "HAK 180 烫金机的操作面板温度如何设置？",
        "reranked_docs": [
            {
                "text": "HAK 180 操作面板位于机器正前方，开机后先设置温度，默认建议 110℃。按上下键调整温度值。",
                "score": 0.95,
                "source": "local",
                "chunk_id": "chunk_001",
            },
            {
                "text": "温度设置范围为 50℃-200℃，步进值为 5℃，设置完成后按确认键生效。",
                "score": 0.88,
                "source": "local",
                "chunk_id": "chunk_002",
            },
        ],
        "retry_count": 0,
        "is_stream": False,
    }

    # 测试2：不足的检索结果
    mock_state_insufficient = {
        "session_id": "test_grader_002",
        "rewritten_query": "HAK 180 烫金机的保修政策是什么？",
        "reranked_docs": [
            {
                "text": "HAK 180 烫金机操作面板温度设置方法...",
                "score": 0.45,
                "source": "local",
                "chunk_id": "chunk_003",
            },
        ],
        "retry_count": 0,
        "is_stream": False,
    }

    # 测试3：重试已耗尽
    mock_state_exhausted = {
        "session_id": "test_grader_003",
        "rewritten_query": "HAK 180 烫金机的保修政策是什么？",
        "reranked_docs": [
            {
                "text": "无关内容...",
                "score": 0.3,
                "source": "web",
            },
        ],
        "retry_count": 1,
        "is_stream": False,
    }

    for name, state in [
        ("充分结果", mock_state_sufficient),
        ("不足结果(可重试)", mock_state_insufficient),
        ("不足结果(已耗尽)", mock_state_exhausted),
    ]:
        try:
            print(f"\n>>> 测试场景: {name}")
            result = node_retrieval_grader(state)
            print(f"retrieval_grade: {result.get('retrieval_grade')}")
            if result.get("rewritten_query"):
                print(f"new_query: {result.get('rewritten_query')}")
            print(
                f"retry_count: {result.get('retry_count', state.get('retry_count', 0))}"
            )
            print("-" * 30)
        except Exception as e:
            logger.exception(f"测试 [{name}] 失败: {e}")

    print("=" * 50)
