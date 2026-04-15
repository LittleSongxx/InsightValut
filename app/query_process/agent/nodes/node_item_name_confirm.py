import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import (
    get_milvus_client,
    create_hybrid_search_requests,
    hybrid_search,
)
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger
from app.conf.query_threshold_config import query_threshold_config
from app.query_process.agent.graph_query_utils import (
    apply_route_overrides,
    build_query_route,
)
from app.query_process.agent.agentic_utils import (
    build_clarification_request,
    is_agentic_feature_enabled,
)

load_dotenv(find_dotenv())


def _fallback_need_rag(query: str, item_names: List[str]) -> bool:
    """
    当 LLM 未返回 use_rag 时的兜底判定。
    - 有商品名/产品型号优先走 RAG
    - 命中产品相关关键词走 RAG
    - 否则按通用对话处理
    """
    if item_names:
        return True

    q = (query or "").strip().lower()
    if not q:
        return False

    rag_keywords = [
        "手册",
        "说明书",
        "型号",
        "参数",
        "规格",
        "操作",
        "使用",
        "安装",
        "故障",
        "维修",
        "设备",
        "产品",
        "商品",
        "机器",
        "配件",
        "零件",
        "保修",
        "售后",
    ]
    return any(k in q for k in rag_keywords)


def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """
    利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
    若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
    :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
    :param history: 列表[字典] - 近期会话历史
    :return: 字典 - 提取结果，格式：{"item_names": [], "rewritten_query": "", "use_rag": true/false}
    """
    logger.info("Step 3: 开始提取信息 (LLM)")

    # 1. 初始化准备
    client = get_llm_client(json_mode=True)

    # 构造历史对话文本
    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"

    logger.info(f"Step 3: 历史上下文构建完成，长度: {len(history_text)} 字符")

    # 2. 加载提示词
    try:
        # 使用关键字参数传递，避免参数位置错误
        prompt = load_prompt(
            "rewritten_query_and_itemnames", history_text=history_text, query=query
        )
        logger.debug(f"Step 3: 提示词加载成功，Prompt长度: {len(prompt)}")
    except Exception:
        logger.exception("Step 3: 加载提示词失败")
        return {
            "item_names": [],
            "rewritten_query": query,
            "use_rag": _fallback_need_rag(query, []),
        }

    messages = [
        SystemMessage(
            content="你是一个智能产品知识库助手，擅长理解用户意图、提取用户询问的商品名称和产品型号。"
        ),
        HumanMessage(content=prompt),
    ]

    try:
        logger.info("Step 3: 正在调用 LLM 进行提取...")
        response = client.invoke(messages)
        content = coerce_llm_content(response.content)
        logger.debug(f"Step 3: LLM 原始响应: {content}")

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)

        # 健壮性检查
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query

        # use_rag 解析与兜底
        use_rag_raw = result.get("use_rag")
        if isinstance(use_rag_raw, bool):
            use_rag = use_rag_raw
        elif isinstance(use_rag_raw, str):
            use_rag = use_rag_raw.strip().lower() in {"1", "true", "yes", "y", "是"}
        else:
            use_rag = _fallback_need_rag(
                result.get("rewritten_query", query), result.get("item_names", [])
            )
        result["use_rag"] = use_rag

        logger.info(
            f"Step 3: 提取结果解析成功 - 商品名: {result['item_names']}, 重写问题: {result['rewritten_query']}, use_rag: {result['use_rag']}"
        )
        return result

    except Exception:
        logger.exception("Step 3: LLM 提取或解析失败")
        return {
            "item_names": [],
            "rewritten_query": query,
            "use_rag": _fallback_need_rag(query, []),
        }


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    """
    对提取的 item_names 进行向量化并在 Milvus 中进行混合搜索
    """
    logger.info(f"Step 4: 开始向量化检索，目标商品: {item_names}")
    results = []

    client = get_milvus_client()
    if not client:
        logger.error("Step 4: 无法连接到 Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: 环境变量中未找到 ITEM_NAME_COLLECTION")
        return results

    try:
        logger.info("Step 4: 正在生成 Embedding (Dense + Sparse)...")
        embeddings = generate_embeddings(item_names)
        logger.info(
            f"Step 4: 向量生成完成，开始 Milvus 搜索 (Collection: {collection_name})"
        )

        for i, name in enumerate(item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                # 构造混合搜索请求
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector, sparse_vector=sparse_vector, limit=5
                )

                # 执行混合搜索
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(
                        query_threshold_config.item_name_dense_weight,
                        query_threshold_config.item_name_sparse_weight,
                    ),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"],
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        item_name = entity.get("item_name")
                        score = hit.get("distance")

                        if item_name:
                            matches.append({"item_name": item_name, "score": score})
                            logger.debug(
                                f"Step 4: '{name}' 匹配项: {item_name} (Score: {score:.4f})"
                            )

                results.append({"extracted_name": name, "matches": matches})
                logger.info(
                    f"Step 4: 商品 '{name}' 检索完成，找到 {len(matches)} 个匹配项"
                )

            except Exception as inner_e:
                logger.error(f"Step 4: 处理商品 '{name}' 时出错: {inner_e}")
                results.append({"extracted_name": name, "matches": []})

    except Exception:
        logger.exception("Step 4: 向量化或搜索过程发生全局错误")

    return results


def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    """
    根据 Milvus 搜索评分，对齐商品名，生成「确认商品名」和「候选商品名」
    """
    logger.info("Step 5: 开始对齐商品名 (Score Analysis)")

    confirmed_item_names = []
    options = []

    for res in query_results:
        extracted_name = res.get("extracted_name", "").strip()
        matches = res.get("matches", []) or []

        if not matches:
            logger.info(f"Step 5: '{extracted_name}' 无匹配结果")
            continue

        # 按分数降序
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 打印详细评分日志辅助调试
        top_matches_log = ", ".join(
            [f"{m['item_name']}({m['score']:.3f})" for m in matches[:3]]
        )
        logger.info(f"Step 5: '{extracted_name}' Top匹配: {top_matches_log}")

        # 筛选
        cfg = query_threshold_config
        high = [m for m in matches if m.get("score", 0) > cfg.item_name_high_threshold]
        mid = [m for m in matches if m.get("score", 0) >= cfg.item_name_mid_threshold]

        # 规则 A: 单个高置信度
        if len(high) == 1:
            confirmed_name = high[0].get("item_name")
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Step 5: 规则A命中 (Single High) -> 确认: {confirmed_name}")
            continue

        # 规则 B: 多个高置信度
        if len(high) > 1:
            picked = None
            # 优先匹配同名
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        logger.info(
                            f"Step 5: 规则B命中 (Exact Match in High) -> 确认: {picked.get('item_name')}"
                        )
                        break

            # 否则取最高分
            if not picked:
                picked = high[0]
                logger.info(
                    f"Step 5: 规则B命中 (Highest Score) -> 确认: {picked.get('item_name')}"
                )

            confirmed_item_names.append(picked.get("item_name"))
            continue

        # 规则 C: 无高置信度，取中置信度候选
        if len(mid) > 0:
            current_options = [m.get("item_name") for m in mid[:5]]
            options.extend(current_options)
            logger.info(
                f"Step 5: 规则C命中 (Mid Confidence) -> 添加候选: {current_options}"
            )
            continue

        logger.info(f"Step 5: 规则D命中 (Low Confidence) -> 无匹配")

    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options)),
    }
    logger.info(f"Step 5: 对齐结果: {result}")
    return result


def step_6_check_confirmation(
    state: Dict,
    align_result: Dict,
    session_id: str,
    history: List[Dict],
    rewritten_query: str,
) -> Dict:
    """
    检查对齐结果，更新 State
    """
    logger.info("Step 6: 检查确认状态并更新 State")

    # 健壮性处理
    if align_result is None:
        align_result = {}

    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    # 分支 A: 有确认商品名
    if confirmed:
        logger.info(f"Step 6: [分支A] 存在确认商品名: {confirmed}")

        # 更新历史消息中的 item_names
        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))

        if ids_to_update and not state.get("evaluation_mode"):
            logger.info(f"Step 6: 更新 {len(ids_to_update)} 条历史消息的关联商品名")
            update_message_item_names(ids_to_update, confirmed)

        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    # 分支 B: 有候选名称
    if options:
        logger.info(f"Step 6: [分支B] 存在候选名称: {options}")
        options_str = "、".join(options[:3])
        answer = f"您是想问以下哪个主题：{options_str}？请明确一下以便精准检索。"
        state["answer"] = answer
        state["item_names"] = []
        return state

    # 分支 C: 无匹配结果 — 不阻断，清空 item_names 继续全库 RAG 检索
    logger.info("Step 6: [分支C] 无确认也无候选，将执行全库检索")
    state["item_names"] = []
    if "answer" in state:
        del state["answer"]
    return state


def step_7_write_history(
    state: Dict,
    session_id: str,
    history: List[Dict],
    rewritten_query: str,
    message_id: str,
) -> Dict:
    """
    写入最终历史记录
    """
    logger.info("Step 7: 写入会话历史")
    if state.get("evaluation_mode"):
        return state

    # 这里只更新用户消息；助手回答统一交由 node_answer_output 写入，避免重复历史记录
    logger.info(f"Step 7: 更新用户消息 (ID: {message_id})")
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id,
    )

    return state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    主节点函数：商品名称确认流程
    """
    logger.info(">>> node_item_name_confirm: 开始处理")

    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)
    # 检测是否为 CRAG retry 回退场景（retrieval_grade == "retry" 表示从 node_retrieval_grader 回退）
    is_retry_from_crag = state.get("retrieval_grade") == "retry"
    # 若从 CRAG retry 回退，不重新保存消息（避免重复），仅更新 rewritten_query
    crag_retry_message_id = state.get("pending_message_id") or state.get("message_id")
    current_query = (
        (state.get("rewritten_query") or "").strip()
        if is_retry_from_crag
        else original_query
    ) or original_query

    # 标记任务开始
    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # 1. 获取历史记录
    history = (
        []
        if state.get("evaluation_mode")
        else get_recent_messages(session_id, limit=10)
    )
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")

    # 2. 保存用户当前消息 (初始保存，后续 step 7 会更新)
    # 若从 CRAG retry 回退，跳过重新保存消息（避免重复），直接使用原消息ID
    if is_retry_from_crag and crag_retry_message_id:
        message_id = crag_retry_message_id
        logger.debug(f"Node: CRAG retry 场景，跳过消息重新保存，使用原ID: {message_id}")
    elif state.get("evaluation_mode"):
        message_id = state.get("message_id") or f"eval-{session_id}"
        state["message_id"] = message_id
    else:
        message_id = save_chat_message(
            session_id, "user", original_query, "", state.get("item_names", [])
        )
        logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")
        state["message_id"] = message_id

    # 3. 提取信息
    extract_res = step_3_extract_info(current_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", current_query)
    use_rag = bool(
        extract_res.get("use_rag", _fallback_need_rag(rewritten_query, item_names))
    )
    if "force_need_rag" in state.get("evaluation_overrides", {}):
        use_rag = bool(state["evaluation_overrides"].get("force_need_rag"))
    forced_item_names = state.get("evaluation_overrides", {}).get("force_item_names")
    if isinstance(forced_item_names, list):
        item_names = [
            str(name).strip() for name in forced_item_names if str(name).strip()
        ]
    if bool((state.get("route_overrides") or {}).get("drop_item_names")):
        logger.info("Node: 运行期补救计划要求丢弃 item_names 过滤，转为全库检索")
        item_names = []

    # 更新 State 中的 rewrite_query
    state["rewritten_query"] = rewritten_query
    state["need_rag"] = use_rag
    state["route_mode"] = "rag" if use_rag else "general_chat"

    # 通用对话：跳过 RAG 检索流程，直接进入答案生成节点
    if not use_rag:
        logger.info("Node: 判定为通用对话，跳过 RAG 检索流程")
        state["item_names"] = []
        route_info = apply_route_overrides(
            build_query_route(rewritten_query, state.get("item_names", [])), state
        )
        state["query_type"] = route_info.get("query_type", "general")
        state["graph_preferred"] = False
        state["query_focus_terms"] = route_info.get("focus_terms", [])
        state["query_route_reason"] = route_info.get("reason", "general_chat")
        state["retrieval_plan"] = route_info.get("retrieval_plan", {})
        state["kg_query_summary"] = {}
        if "answer" in state:
            del state["answer"]

        final_state = step_7_write_history(
            state, session_id, history, rewritten_query, message_id
        )
        final_state["history"] = history
        add_done_task(session_id, "node_item_name_confirm", is_stream)
        logger.info("Node: 通用对话路由完成，进入答案生成")
        return final_state

    align_result = {}

    # 4. & 5. 如果有提取到商品名，进行搜索和对齐
    if len(item_names) > 0 and isinstance(forced_item_names, list):
        state["item_names"] = item_names
        if "answer" in state:
            del state["answer"]
    elif len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
        # 6. 检查确认状态
        state = step_6_check_confirmation(
            state, align_result, session_id, history, rewritten_query
        )
    else:
        logger.info("Node: RAG 模式但未提取到商品名，后续将执行无商品过滤检索")
        state["item_names"] = []
        if "answer" in state:
            del state["answer"]

    route_info = apply_route_overrides(
        build_query_route(rewritten_query, state.get("item_names", [])), state
    )
    state["query_type"] = route_info.get("query_type", "general")
    state["graph_preferred"] = bool(route_info.get("graph_preferred", False))
    state["query_focus_terms"] = route_info.get("focus_terms", [])
    state["query_route_reason"] = route_info.get("reason", "")
    state["retrieval_plan"] = route_info.get("retrieval_plan", {})
    if not state.get("kg_query_summary"):
        state["kg_query_summary"] = {}
    state["clarification_reason"] = ""

    if (
        use_rag
        and not state.get("answer")
        and is_agentic_feature_enabled(state, "clarification_guard")
    ):
        clarification = build_clarification_request(state)
        if clarification.get("required"):
            state["answer"] = clarification.get("question") or ""
            state["clarification_reason"] = clarification.get("reason") or ""
            logger.info(
                "Node: 触发细粒度澄清策略, reason=%s",
                state["clarification_reason"],
            )

    # 7. 写入最终历史
    final_state = step_7_write_history(
        state, session_id, history, rewritten_query, message_id
    )

    # 将 history 存入 state，供后续节点（如 node_answer_output）使用
    final_state["history"] = history

    # 标记任务完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)

    logger.info(
        f"Node: 处理结束, Final State Item Names: {final_state.get('item_names')}"
    )
    return final_state


if __name__ == "__main__":
    # 测试代码块
    print("\n" + "=" * 50)
    print(">>> 启动 node_item_name_confirm 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_debug_session_001",
        "original_query": "HAK 180 烫金机多少钱？",  # 针对用户提到的具体 case
        "is_stream": False,
        "item_names": [],
    }

    try:
        # 运行节点
        result = node_item_name_confirm(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Item Names: {result.get('item_names')}")
        print(f"Answer: {result.get('answer')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
