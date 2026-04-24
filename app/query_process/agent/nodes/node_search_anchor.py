import sys

from app.core.logger import logger
from app.query_process.agent.graph_query_utils import should_run_retriever
from app.query_process.agent.retrieval_utils import run_anchor_search
from app.utils.anchor_context_utils import extract_query_anchor_targets
from app.utils.task_utils import add_done_task, add_running_task


OUTPUT_FIELDS = [
    "chunk_id",
    "stable_chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    "image_urls",
    "section_path",
    "chunk_context",
    "bm25_text",
    "anchor_terms",
]


def node_search_anchor(state):
    logger.info("---node_search_anchor 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    if not should_run_retriever(state, "anchor"):
        logger.info("Anchor 检索按计划跳过")
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {"anchor_chunks": [], "anchor_hits": []}

    rewritten_query = state.get("rewritten_query") or ""
    original_query = state.get("original_query") or ""
    item_names = state.get("item_names") or []
    original_targets = extract_query_anchor_targets(original_query, item_names=item_names)
    rewritten_targets = extract_query_anchor_targets(rewritten_query, item_names=item_names)
    query = original_query if original_targets else (rewritten_query or original_query)
    targets = original_targets or rewritten_targets
    results = run_anchor_search(
        query_text=query,
        item_names=item_names,
        output_fields=list(OUTPUT_FIELDS),
    )
    anchor_hits = [
        {
            "chunk_id": str(
                (item.get("entity") or {}).get("chunk_id")
                or item.get("id")
                or ""
            ),
            "title": str((item.get("entity") or {}).get("title") or ""),
            "score": float(item.get("distance") or 0.0),
        }
        for item in results[:8]
        if isinstance(item, dict)
    ]

    logger.info(
        f"node_search_anchor 处理成功，targets={targets}，检索到 {len(results)} 条片段"
    )
    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    return {
        "anchor_chunks": results,
        "anchor_hits": anchor_hits,
        "query_anchor_targets": targets,
    }
