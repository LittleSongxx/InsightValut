import sys

from app.core.logger import logger
from app.query_process.agent.retrieval_utils import run_bm25_search
from app.query_process.agent.graph_query_utils import should_run_retriever
from app.utils.task_utils import add_done_task, add_running_task


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


def node_search_bm25(state):
    logger.info("---node_search_bm25 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    if not should_run_retriever(state, "bm25"):
        logger.info("BM25 检索按题型计划跳过")
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {"bm25_chunks": []}

    query = state.get("rewritten_query") or state.get("original_query") or ""
    item_names = state.get("item_names") or []
    results = run_bm25_search(
        query_text=query,
        item_names=item_names,
        output_fields=list(OUTPUT_FIELDS),
    )

    logger.info(f"node_search_bm25 处理成功，检索到 {len(results)} 条相关片段")
    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    return {"bm25_chunks": results}
