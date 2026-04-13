import sys
import json
from typing import List, Dict

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.nodes.node_rrf import (
    reciprocal_rank_fusion,
    _as_entity_list,
)
from app.query_process.agent.retrieval_utils import (
    run_bm25_search,
    run_embedding_hybrid_search,
)
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.conf.query_threshold_config import query_threshold_config
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

cfg = query_threshold_config
MAX_SUB_QUERIES = cfg.max_sub_queries


def step_1_detect_compound(question: str, item_names: List[str]) -> Dict:
    """
    阶段1：利用LLM判断是否为复合问题，并分解子查询
    :param question: 改写后的用户问题
    :param item_names: 已确认的商品名列表
    :return: {"is_compound": bool, "sub_queries": list, "reason": str}
    """
    logger.info(f"Step 1: 开始检测复合问题, Query: {question}")

    try:
        client = get_llm_client(json_mode=True)
        item_names_str = ", ".join(item_names) if item_names else "无"
        prompt = load_prompt(
            "query_decompose", question=question, item_names=item_names_str
        )

        response = client.invoke(prompt)
        content = response.content

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        is_compound = bool(result.get("is_compound", False))
        sub_queries = result.get("sub_queries", [])

        # 限制子查询数量
        if len(sub_queries) > MAX_SUB_QUERIES:
            sub_queries = sub_queries[:MAX_SUB_QUERIES]
            logger.warning(f"Step 1: 子查询数量超限，截断为 {MAX_SUB_QUERIES} 个")

        logger.info(
            f"Step 1: 检测结果 is_compound={is_compound}, "
            f"sub_queries={sub_queries}, reason={result.get('reason', '')}"
        )
        return {
            "is_compound": is_compound,
            "sub_queries": sub_queries,
            "reason": result.get("reason", ""),
        }

    except Exception as e:
        logger.error(f"Step 1: 复合问题检测失败: {e}", exc_info=True)
        # 降级：视为简单问题
        return {"is_compound": False, "sub_queries": [], "reason": f"检测失败: {e}"}


def step_2_search_sub_queries(
    sub_queries: List[str], item_names: List[str]
) -> List[Dict]:
    """
    阶段2：对每个子查询独立执行 Milvus 混合检索，合并所有结果
    :param sub_queries: 子查询列表
    :param item_names: 商品名列表（用于过滤）
    :return: RRF 融合后的结果列表
    """
    logger.info(f"Step 2: 开始执行 {len(sub_queries)} 个子查询的独立检索")

    # 收集每个子查询的检索结果，用于后续 RRF 融合
    all_source_weights = []

    for i, sub_q in enumerate(sub_queries):
        logger.info(f"Step 2: 子查询 [{i + 1}/{len(sub_queries)}]: {sub_q}")
        try:
            embedding_results = run_embedding_hybrid_search(
                query_text=sub_q,
                item_names=item_names,
                req_limit=cfg.embedding_req_limit,
                top_k=cfg.embedding_top_k,
                output_fields=[
                    "chunk_id",
                    "content",
                    "title",
                    "parent_title",
                    "part",
                    "file_title",
                    "item_name",
                    "image_urls",
                ],
            )
            embedding_entities = _as_entity_list(embedding_results)
            if embedding_entities:
                all_source_weights.append(
                    (embedding_entities, cfg.rrf_weight_embedding)
                )
                logger.info(
                    f"Step 2: 子查询 [{i + 1}] Embedding 检索到 {len(embedding_entities)} 条结果"
                )

            bm25_results = run_bm25_search(
                query_text=sub_q,
                item_names=item_names,
                top_k=cfg.bm25_top_k,
                candidate_limit=cfg.bm25_candidate_limit,
                output_fields=[
                    "chunk_id",
                    "content",
                    "title",
                    "parent_title",
                    "part",
                    "file_title",
                    "item_name",
                    "image_urls",
                ],
            )
            bm25_entities = _as_entity_list(bm25_results)
            if bm25_entities:
                all_source_weights.append((bm25_entities, cfg.rrf_weight_bm25))
                logger.info(
                    f"Step 2: 子查询 [{i + 1}] BM25 检索到 {len(bm25_entities)} 条结果"
                )

            if not embedding_entities and not bm25_entities:
                logger.warning(f"Step 2: 子查询 [{i + 1}] 无检索结果")

        except Exception as e:
            logger.error(f"Step 2: 子查询 [{i + 1}] 检索失败: {e}", exc_info=True)

    # RRF 融合所有子查询的结果
    if not all_source_weights:
        logger.warning("Step 2: 所有子查询均无结果")
        return []

    rrf_results = reciprocal_rank_fusion(
        all_source_weights,
        k=cfg.rrf_k,
        max_results=cfg.rrf_max_results,
    )
    merged_chunks = [doc for doc, score in rrf_results]

    logger.info(f"Step 2: RRF 融合完成，共 {len(merged_chunks)} 条结果")
    return merged_chunks


def node_query_decompose(state):
    """
    复合问题分解节点

    功能：
    1. 检测用户问题是否为复合问题（多产品对比、多个独立子问题等）
    2. 若为复合问题：分解为子查询，独立检索每个子查询，RRF 融合结果写入 rrf_chunks
    3. 若为简单问题：直接透传，由后续正常多路检索处理

    :param state: 查询流程全局状态
    :return: 更新后的状态字段
    """
    logger.info("---node_query_decompose (复合问题分解) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    question = state.get("rewritten_query") or state.get("original_query", "")
    item_names = state.get("item_names", [])

    # 阶段1：检测复合问题
    detect_result = step_1_detect_compound(question, item_names)
    is_compound = detect_result.get("is_compound", False)
    sub_queries = detect_result.get("sub_queries", [])

    result = {
        "is_compound_query": is_compound,
        "sub_queries": sub_queries,
    }

    # 阶段2：复合问题 → 执行子查询检索
    if is_compound and sub_queries:
        logger.info(f"检测为复合问题，分解为 {len(sub_queries)} 个子查询")
        rrf_chunks = step_2_search_sub_queries(sub_queries, item_names)
        result["rrf_chunks"] = rrf_chunks
    else:
        logger.info("检测为简单问题，跳过分解，交由正常检索流程处理")

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info("---node_query_decompose 处理结束---")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_query_decompose 本地测试")
    print("=" * 50)

    # 测试1：复合问题
    mock_state_compound = {
        "session_id": "test_decompose_001",
        "rewritten_query": "HAK 180 烫金机和 HAK 200 烫金机有什么区别？各自的参数是什么？",
        "item_names": ["HAK 180 烫金机", "HAK 200 烫金机"],
        "is_stream": False,
    }

    # 测试2：简单问题
    mock_state_simple = {
        "session_id": "test_decompose_002",
        "rewritten_query": "HAK 180 烫金机的操作面板怎么设置温度？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }

    for name, state in [
        ("复合问题", mock_state_compound),
        ("简单问题", mock_state_simple),
    ]:
        try:
            print(f"\n>>> 测试场景: {name}")
            result = node_query_decompose(state)
            print(f"is_compound_query: {result.get('is_compound_query')}")
            print(f"sub_queries: {result.get('sub_queries')}")
            if result.get("rrf_chunks"):
                print(f"rrf_chunks 数量: {len(result['rrf_chunks'])}")
            print("-" * 30)
        except Exception as e:
            logger.exception(f"测试 [{name}] 失败: {e}")

    print("=" * 50)
