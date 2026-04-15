import asyncio
import json
import sys
from typing import Any, Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv

from app.clients.neo4j_graph_utils import query_graph_context
from app.clients.neo4j_utils import verify_connection
from app.conf.query_threshold_config import query_threshold_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.query_process.agent.graph_query_utils import (
    apply_route_overrides,
    build_query_route,
    get_bm25_enabled,
    should_run_retriever,
)
from app.query_process.agent.agentic_utils import is_agentic_feature_enabled
from app.query_process.agent.nodes.node_rrf import (
    _as_entity_list,
    reciprocal_rank_fusion,
)
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    step_1_create_hyde_doc,
    step_2_search_by_query_and_hyde,
)
from app.query_process.agent.nodes.node_web_search_mcp import mcp_call
from app.query_process.agent.retrieval_utils import (
    run_bm25_search,
    run_embedding_hybrid_search,
)
from app.utils.task_utils import add_done_task, add_running_task

load_dotenv(find_dotenv())

cfg = query_threshold_config
MAX_SUB_QUERIES = cfg.max_sub_queries
OUTPUT_FIELDS = [
    "chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    "image_urls",
]


def step_1_detect_compound(question: str, item_names: List[str]) -> Dict[str, Any]:
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
        content = coerce_llm_content(response.content)

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        is_compound = bool(result.get("is_compound", False))
        sub_queries = result.get("sub_queries", [])

        if len(sub_queries) > MAX_SUB_QUERIES:
            sub_queries = sub_queries[:MAX_SUB_QUERIES]
            logger.warning(f"Step 1: 子查询数量超限，截断为 {MAX_SUB_QUERIES} 个")

        logger.info(
            f"Step 1: 检测结果 is_compound={is_compound}, sub_queries={sub_queries}, "
            f"reason={result.get('reason', '')}"
        )
        return {
            "is_compound": is_compound,
            "sub_queries": sub_queries,
            "reason": result.get("reason", ""),
        }

    except Exception as e:
        logger.exception("Step 1: 复合问题检测失败")
        return {"is_compound": False, "sub_queries": [], "reason": f"检测失败: {e}"}


def _parse_web_docs(raw_result: Any) -> List[Dict[str, str]]:
    if not raw_result or getattr(raw_result, "isError", False) or not raw_result.content:
        return []
    try:
        raw_text = raw_result.content[0].text
        data = json.loads(raw_text)
    except Exception:
        return []

    docs: List[Dict[str, str]] = []
    for page in data.get("pages") or []:
        snippet = str(page.get("snippet") or "").strip()
        if not snippet:
            continue
        docs.append(
            {
                "title": str(page.get("title") or "").strip(),
                "url": str(page.get("url") or "").strip(),
                "snippet": snippet,
            }
        )
    return docs


def _dedupe_web_docs(docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    unique_docs: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for doc in docs:
        key = (str(doc.get("url") or "").strip(), str(doc.get("title") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def _subquery_temp_state(
    state: Dict[str, Any], retrieval_plan: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "retrieval_plan": retrieval_plan,
        "evaluation_overrides": state.get("evaluation_overrides") or {},
        "route_overrides": state.get("route_overrides") or {},
    }


def step_2_search_sub_queries(
    state: Dict[str, Any], sub_queries: List[str], item_names: List[str]
) -> Dict[str, Any]:
    """
    阶段2：对每个子查询独立执行带路由的多路检索，再统一融合
    """
    logger.info(f"Step 2: 开始执行 {len(sub_queries)} 个子查询的独立检索")

    all_source_weights: List[Tuple[List[Dict[str, Any]], float]] = []
    all_web_docs: List[Dict[str, str]] = []
    sub_query_routes: List[Dict[str, Any]] = []
    sub_query_results: List[Dict[str, Any]] = []
    compound_kg_templates: List[str] = []
    kg_total_hits = 0
    neo4j_available: bool | None = None
    subquery_routing_enabled = is_agentic_feature_enabled(state, "subquery_routing")

    for index, sub_query in enumerate(sub_queries, start=1):
        logger.info(f"Step 2: 子查询 [{index}/{len(sub_queries)}]: {sub_query}")
        route_info = apply_route_overrides(build_query_route(sub_query, item_names), state)
        if not subquery_routing_enabled:
            route_info["query_type"] = "general"
            route_info["graph_preferred"] = False
            route_info["reason"] = "subquery_routing_disabled"
            route_info["retrieval_plan"] = {
                "graph_first": False,
                "run_kg": False,
                "run_embedding": True,
                "run_bm25": get_bm25_enabled(state),
                "run_hyde": False,
                "run_web": False,
                "graph_limit": 0,
                "kg_weight_multiplier": 0.0,
                "embedding_weight_multiplier": 1.0,
                "bm25_weight_multiplier": 1.0,
                "hyde_weight_multiplier": 0.0,
            }
        retrieval_plan = route_info.get("retrieval_plan") or {}
        sub_query_routes.append(
            {
                "query": sub_query,
                "query_type": route_info.get("query_type", "general"),
                "graph_preferred": bool(route_info.get("graph_preferred", False)),
                "focus_terms": route_info.get("focus_terms", []),
                "retrieval_plan": retrieval_plan,
                "reason": route_info.get("reason", ""),
            }
        )

        temp_state = _subquery_temp_state(state, retrieval_plan)
        summary: Dict[str, Any] = {
            "query": sub_query,
            "query_type": route_info.get("query_type", "general"),
            "graph_preferred": bool(route_info.get("graph_preferred", False)),
            "embedding_hits": 0,
            "bm25_hits": 0,
            "hyde_hits": 0,
            "kg_hits": 0,
            "web_hits": 0,
            "errors": [],
        }

        try:
            if should_run_retriever(temp_state, "embedding"):
                embedding_results = run_embedding_hybrid_search(
                    query_text=sub_query,
                    item_names=item_names,
                    req_limit=cfg.embedding_req_limit,
                    top_k=cfg.embedding_top_k,
                    output_fields=list(OUTPUT_FIELDS),
                )
                embedding_entities = _as_entity_list(embedding_results)
                summary["embedding_hits"] = len(embedding_entities)
                if embedding_entities:
                    all_source_weights.append(
                        (
                            embedding_entities,
                            cfg.rrf_weight_embedding
                            * float(
                                retrieval_plan.get("embedding_weight_multiplier", 1.0)
                                or 1.0
                            ),
                        )
                    )
        except Exception as exc:
            summary["errors"].append(f"embedding:{exc}")
            logger.exception(f"Step 2: 子查询 [{index}] Embedding 检索失败")

        try:
            if should_run_retriever(temp_state, "bm25"):
                bm25_results = run_bm25_search(
                    query_text=sub_query,
                    item_names=item_names,
                    top_k=cfg.bm25_top_k,
                    candidate_limit=cfg.bm25_candidate_limit,
                    output_fields=list(OUTPUT_FIELDS),
                )
                bm25_entities = _as_entity_list(bm25_results)
                summary["bm25_hits"] = len(bm25_entities)
                if bm25_entities:
                    all_source_weights.append(
                        (
                            bm25_entities,
                            cfg.rrf_weight_bm25
                            * float(
                                retrieval_plan.get("bm25_weight_multiplier", 1.0)
                                or 1.0
                            ),
                        )
                    )
        except Exception as exc:
            summary["errors"].append(f"bm25:{exc}")
            logger.exception(f"Step 2: 子查询 [{index}] BM25 检索失败")

        try:
            if should_run_retriever(temp_state, "hyde"):
                hyde_doc = step_1_create_hyde_doc(sub_query)
                hyde_results = step_2_search_by_query_and_hyde(
                    rewritten_query=sub_query,
                    hyde_doc=hyde_doc,
                    item_names=item_names,
                    req_limit=cfg.hyde_req_limit,
                    top_k=cfg.hyde_top_k,
                    ranker_weights=(cfg.hybrid_dense_weight, cfg.hybrid_sparse_weight),
                    output_fields=list(OUTPUT_FIELDS),
                )
                hyde_entities = _as_entity_list(hyde_results)
                summary["hyde_hits"] = len(hyde_entities)
                if hyde_entities:
                    all_source_weights.append(
                        (
                            hyde_entities,
                            cfg.rrf_weight_hyde
                            * float(
                                retrieval_plan.get("hyde_weight_multiplier", 1.0)
                                or 1.0
                            ),
                        )
                    )
        except Exception as exc:
            summary["errors"].append(f"hyde:{exc}")
            logger.exception(f"Step 2: 子查询 [{index}] HyDE 检索失败")

        try:
            if should_run_retriever(temp_state, "kg"):
                if neo4j_available is None:
                    neo4j_available = verify_connection()
                if neo4j_available:
                    graph_result = query_graph_context(
                        sub_query,
                        item_names,
                        query_type=route_info.get("query_type", "general"),
                        focus_terms=route_info.get("focus_terms") or [],
                        limit=int(retrieval_plan.get("graph_limit", 8) or 8),
                    )
                    kg_results = graph_result.get("kg_chunks") or []
                    kg_entities = _as_entity_list(kg_results)
                    kg_summary = graph_result.get("summary") or {}
                    kg_total_hits += len(kg_entities)
                    template = str(kg_summary.get("template") or "").strip()
                    if template and template not in compound_kg_templates:
                        compound_kg_templates.append(template)
                    summary["kg_hits"] = len(kg_entities)
                    summary["kg_template"] = template
                    if kg_entities:
                        all_source_weights.append(
                            (
                                kg_entities,
                                cfg.rrf_weight_kg
                                * float(
                                    retrieval_plan.get("kg_weight_multiplier", 1.0)
                                    or 1.0
                                ),
                            )
                        )
        except Exception as exc:
            summary["errors"].append(f"kg:{exc}")
            logger.exception(f"Step 2: 子查询 [{index}] KG 检索失败")

        try:
            if should_run_retriever(temp_state, "web"):
                web_result = asyncio.run(mcp_call(sub_query))
                web_docs = _parse_web_docs(web_result)
                summary["web_hits"] = len(web_docs)
                if web_docs:
                    all_web_docs.extend(web_docs)
        except Exception as exc:
            summary["errors"].append(f"web:{exc}")
            logger.exception(f"Step 2: 子查询 [{index}] Web 检索失败")

        sub_query_results.append(summary)

    if not all_source_weights:
        logger.warning("Step 2: 所有子查询均无本地检索结果")
        return {
            "rrf_chunks": [],
            "web_search_docs": _dedupe_web_docs(all_web_docs),
            "sub_query_routes": sub_query_routes,
            "sub_query_results": sub_query_results,
            "kg_query_summary": {
                "compound": True,
                "sub_query_count": len(sub_queries),
                "result_count": kg_total_hits,
                "templates": compound_kg_templates,
            },
        }

    rrf_results = reciprocal_rank_fusion(
        all_source_weights,
        k=cfg.rrf_k,
        max_results=cfg.rrf_max_results,
    )
    merged_chunks = [doc for doc, _score in rrf_results]

    logger.info(f"Step 2: 多子查询融合完成，共 {len(merged_chunks)} 条结果")
    return {
        "rrf_chunks": merged_chunks,
        "web_search_docs": _dedupe_web_docs(all_web_docs),
        "sub_query_routes": sub_query_routes,
        "sub_query_results": sub_query_results,
        "kg_query_summary": {
            "compound": True,
            "sub_query_count": len(sub_queries),
            "result_count": kg_total_hits,
            "templates": compound_kg_templates,
        },
    }


def node_query_decompose(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    复合问题分解节点

    功能：
    1. 检测用户问题是否为复合问题（多产品对比、多个独立子问题等）
    2. 若为复合问题：分解为子查询，分别做子问题级路由与检索
    3. 若为简单问题：直接透传，由后续正常多路检索处理
    """
    logger.info("---node_query_decompose (复合问题分解) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    question = state.get("rewritten_query") or state.get("original_query", "")
    item_names = state.get("item_names", [])

    detect_result = step_1_detect_compound(question, item_names)
    is_compound = detect_result.get("is_compound", False)
    sub_queries = detect_result.get("sub_queries", [])

    result: Dict[str, Any] = {
        "is_compound_query": is_compound,
        "sub_queries": sub_queries,
        "sub_query_routes": [],
        "sub_query_results": [],
    }

    if is_compound and sub_queries:
        logger.info(f"检测为复合问题，分解为 {len(sub_queries)} 个子查询")
        result.update(step_2_search_sub_queries(state, sub_queries, item_names))
    else:
        logger.info(
            f"检测为简单问题，跳过分解，交由正常检索流程处理 "
            f"(bm25_enabled={get_bm25_enabled(state)})"
        )

    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info("---node_query_decompose 处理结束---")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_query_decompose 本地测试")
    print("=" * 50)

    mock_state_compound = {
        "session_id": "test_decompose_001",
        "rewritten_query": "HAK 180 烫金机和 HAK 200 烫金机有什么区别？各自的参数是什么？",
        "item_names": ["HAK 180 烫金机", "HAK 200 烫金机"],
        "is_stream": False,
    }

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
            print(f"sub_query_routes: {len(result.get('sub_query_routes') or [])}")
            if result.get("rrf_chunks"):
                print(f"rrf_chunks 数量: {len(result['rrf_chunks'])}")
            print("-" * 30)
        except Exception as e:
            logger.exception(f"测试 [{name}] 失败: {e}")

    print("=" * 50)
