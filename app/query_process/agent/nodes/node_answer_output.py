import sys
from typing import Any, Dict
from urllib.parse import urlsplit
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.markdown_image_utils import extract_markdown_image_urls
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.conf.lm_config import lm_config
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.clients.mongo_history_utils import save_chat_message
from app.conf.query_threshold_config import query_threshold_config
from app.query_process.agent.agentic_utils import build_agentic_response_metadata
from app.utils.query_cache_utils import (
    get_current_request_cache_summary,
    normalize_cache_sequence,
    query_cache_get,
    query_cache_set,
)

_IMAGE_BLOCK_MARKER = "【图片】"
cfg = query_threshold_config
MAX_CONTEXT_CHARS = cfg.max_context_chars
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")


def _is_probably_image_url(url: str) -> bool:
    candidate = (url or "").strip()
    if not candidate:
        return False
    if candidate.startswith("data:image/") or candidate.startswith("blob:"):
        return True
    if any(ch.isspace() for ch in candidate):
        return False

    path = urlsplit(candidate).path.lower()
    return path.endswith(IMAGE_SUFFIXES)


def step_1_check_answer(state) -> bool:
    """
    阶段一：检查 state 中是否已有 answer。
    - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
    - 若不存在：返回 False
    """
    answer = state.get("answer", None)
    is_stream = state.get("is_stream")
    if answer:
        if is_stream:
            logger.info("---Step 1: 发现已有答案，执行流式推送---")
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    else:
        return False


def step_2_construct_prompt(state: QueryGraphState) -> str:
    """
    第一阶段：构建 Prompt
    根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt
    """
    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "")
    question = rewritten_query if rewritten_query else original_query
    history = state.get("history", [])
    item_names = state.get("item_names", [])
    reranked_docs = state.get("reranked_docs") or []

    docs = []
    used = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        source = doc.get("source") or ""
        chunk_id = doc.get("chunk_id")
        url = (doc.get("url") or "").strip()
        title = doc.get("title") or ""
        score = doc.get("score")

        meta_parts = [f"[{i}]"]
        if source:
            meta_parts.append(f"[{source}]")
        if chunk_id:
            meta_parts.append(f"[chunk_id={chunk_id}]")
        if url:
            meta_parts.append(f"[url={url}]")
        if score is not None:
            meta_parts.append(f"[score={float(score):.4f}]")
        if title:
            meta_parts.append(f"[title={title}]")
        doc_str = " ".join(meta_parts) + "\n" + text
        if used + len(doc_str) > MAX_CONTEXT_CHARS:
            break
        docs.append(doc_str)
        used += len(doc_str) + 2
    context_str = "\n\n".join(docs) if docs else "无参考内容"

    history_str = ""
    if history:
        for msg in history:
            role = msg.get("role")
            text = msg.get("text")
            if role == "user" and text:
                history_str += f"用户: {text}\n"
            elif role == "assistant" and text:
                history_str += f"助手: {text}\n"
            used += len(history_str) + 2
            if used > MAX_CONTEXT_CHARS:
                break
    else:
        history_str = "无历史对话"

    item_names_str = ", ".join(item_names) if item_names else "无指定商品"

    prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )

    hallucination_feedback = state.get("hallucination_feedback")
    if hallucination_feedback:
        prompt += (
            f"\n\n【重要约束】上一次回答存在以下事实错误，请在本次回答中严格避免：\n"
            f"{hallucination_feedback}\n"
            f"请务必只基于【参考内容】中明确提到的事实进行回答，不要编造任何具体数据或步骤。"
        )
        logger.info(f"已追加幻觉反馈约束: {hallucination_feedback}")

    answer_plan = state.get("answer_plan") or {}
    evidence_coverage = state.get("evidence_coverage_summary") or {}
    if answer_plan:
        sections = "\n".join(
            f"- {section}" for section in (answer_plan.get("sections") or [])
        )
        style_instructions = "\n".join(
            f"- {instruction}"
            for instruction in (answer_plan.get("style_instructions") or [])
        )
        must_cover = ", ".join(answer_plan.get("must_cover") or []) or "无硬性覆盖项"
        response_format = answer_plan.get("response_format") or "paragraph"
        prompt += (
            "\n\n【回答规划】\n"
            f"回答格式：{response_format}\n"
            f"建议章节：\n{sections or '- 直接回答'}\n"
            f"写作要求：\n{style_instructions or '- 直接、准确、保守回答'}\n"
            f"需要优先覆盖的对象：{must_cover}\n"
        )

    missing_focus_terms = evidence_coverage.get("missing_focus_terms") or []
    if missing_focus_terms:
        prompt += (
            "\n【证据覆盖提醒】当前证据未完全覆盖以下焦点词："
            f"{', '.join(missing_focus_terms[:5])}。"
            "如果证据不足，请明确说明“当前资料未直接给出”，不要自行补全。"
        )

    logger.info(f"组装后的提示词为：{prompt}")

    return prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
    """
    第二阶段：生成回答
    调用llm生成答案，支持流式输出
    """
    logger.info("---Step 3: 开始生成回答 (LLM Generation)---")
    logger.debug(f"最终Prompt内容: {prompt}")

    llm = get_llm_client()
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")

    if is_stream:
        logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
        final_text = ""
        try:
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    final_text += delta
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
            logger.info(f"流式输出完成，总长度: {len(final_text)}")
        except Exception as e:
            logger.exception("流式生成出错")
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        state["answer"] = final_text
    else:
        logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
        try:
            response = llm.invoke(prompt)
            content = coerce_llm_content(response.content)
            state["answer"] = content
            set_task_result(session_id, "answer", content)
            logger.info(f"生成回答完成，长度: {len(content)}")
        except Exception:
            logger.exception("生成回答出错")
            state["answer"] = "抱歉，生成回答时出现错误。"

    return state


def _extract_images_from_docs(docs):
    """
    从文档列表中提取图片URL。

    优先策略：
    1. 优先从 chunk 的 image_urls 字段获取（导入阶段已提取并存储到 Milvus）
    2. 兜底策略：正则扫描 text 正文内容中的 Markdown 图片语法
    3. 检查 url 字段（常见于联网搜索结果）
    """
    images = []
    seen = set()
    if not docs:
        return []

    for i, doc in enumerate(docs):
        # 策略1：优先从 image_urls 字段获取（推荐，数据已结构化存储）
        field_urls = doc.get("image_urls") or []
        if isinstance(field_urls, list):
            for url in field_urls:
                url = (url or "").strip()
                if _is_probably_image_url(url) and url not in seen:
                    seen.add(url)
                    images.append(url)
                    logger.debug(f"文档[{i}] 从 image_urls 字段获取图片: {url}")

        # 策略2：检查 url 字段（联网搜索结果）
        url = (doc.get("url") or "").strip()
        if url:
            if url.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
            ):
                if url not in seen:
                    logger.debug(f"文档[{i}] 从 url 字段发现图片 URL: {url}")
                    seen.add(url)
                    images.append(url)

        # 策略3：正则扫描 text 正文（兜底，适用于旧数据或不规范存储）
        text = (doc.get("text") or "").strip()
        if text:
            matches = extract_markdown_image_urls(text)
            for img_url in matches:
                img_url = img_url.strip()
                if _is_probably_image_url(img_url) and img_url not in seen:
                    logger.debug(f"文档[{i}] 正则提取正文图片: {img_url}")
                    seen.add(img_url)
                    images.append(img_url)

    logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
    return images


def step_4_write_history(
    state: QueryGraphState,
    image_urls=None,
    metadata: Dict[str, Any] | None = None,
) -> QueryGraphState:
    """
    阶段四：把本轮答案写入 MongoDB history。
    """
    session_id = state.get("session_id", "default")
    answer = (state.get("answer") or "").strip()
    item_names = state.get("item_names") or []
    if state.get("evaluation_mode"):
        return state

    try:
        if answer:
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query="",
                item_names=item_names,
                image_urls=image_urls,
                metadata=metadata,
                message_id=None,
            )
    except Exception:
        logger.exception("写入Mongo历史记录失败")

    return state


def _answer_cache_descriptor(state: QueryGraphState, prompt: str) -> Dict[str, Any]:
    reranked_docs = state.get("reranked_docs") or []
    return {
        "model": lm_config.llm_model or "default",
        "question": state.get("rewritten_query") or state.get("original_query") or "",
        "prompt": prompt,
        "item_names": normalize_cache_sequence(state.get("item_names") or []),
        "doc_ids": [
            doc.get("chunk_id") or doc.get("doc_id") or doc.get("url") or ""
            for doc in reranked_docs[:10]
        ],
    }


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
    3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
    4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
    5 做最后一次push操作（主要是为了触发前端图片渲染)
       {
          "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
          "status": "completed",
          "image_urls": [
              "http://local-server/images/panel_view.jpg",
              "http://local-server/images/button_detail.jpg"
          ]
        }
    """
    logger.info("---node_answer_output (答案生成) 节点开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    answer_exists = step_1_check_answer(state)

    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt
        answer_cache_descriptor = _answer_cache_descriptor(state, prompt)
        cached_answer = query_cache_get("answer", answer_cache_descriptor)
        if isinstance(cached_answer, str) and cached_answer.strip():
            logger.info("答案缓存命中，跳过 LLM 生成")
            state["answer"] = cached_answer
        else:
            step_3_generate_response(state, prompt)
            if state.get("answer"):
                query_cache_set("answer", answer_cache_descriptor, state.get("answer"))

    # 提取图片URL（用于历史记录和前端展示）
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
    state["cache_summary"] = get_current_request_cache_summary()
    if state.get("answer") and not state.get("is_stream"):
        set_task_result(state["session_id"], "answer", state.get("answer"))
    response_metadata = build_agentic_response_metadata(state, image_urls=image_urls)
    set_task_result(state["session_id"], "metadata", response_metadata)

    # 把答案写入到mongodb的history中
    if state.get("answer"):
        logger.info("---写入MongoDB历史记录---")
        step_4_write_history(state, image_urls=image_urls, metadata=response_metadata)

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # 流式输出结束，发送 final 事件
    logger.info(f"---发送 final 事件---图片为：{image_urls}")
    if state.get("is_stream"):
        push_to_session(
            state["session_id"],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "status": "completed",
                "image_urls": image_urls,
                "metadata": response_metadata,
            },
        )

    logger.info("---node_answer_output 节点处理结束---")
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "image_urls": [
                "http://local-server/images/panel_view.jpg",
                "http://local-server/images/knob_detail.png",
            ],
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            """,
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",
            "text": "如果机器无法加热，请检查保险丝是否熔断...",
        },
    ]

    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"},
    ]

    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }

    try:
        result = node_answer_output(mock_state)
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
        else:
            print("[FAIL] Prompt 未构建")

        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
        else:
            print(f"[WARN] 答案生成可能异常")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
