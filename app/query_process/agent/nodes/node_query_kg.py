import sys
from typing import Dict, Any, List

from app.utils.task_utils import add_running_task, add_done_task
from app.clients.neo4j_utils import verify_connection
from app.clients.neo4j_graph_utils import query_graph_context
from app.query_process.agent.graph_query_utils import should_run_retriever
from app.core.logger import logger


def node_query_kg(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Neo4j 知识图谱查询节点

    功能：根据已确认的商品名称(item_names)，从 Neo4j 图谱中检索关联切片。
    与向量检索互补：向量检索按语义相似度召回，图谱检索按产品关系召回，
    两者在 RRF 节点中融合，提升召回率。

    :param state: 查询流程全局状态
    :return: {"kg_chunks": [...]} 图谱检索到的切片列表
    """
    logger.info("---node_query_kg (知识图谱查询) 开始处理---")
    add_running_task(
        state["session_id"],
        sys._getframe().f_code.co_name,
        state.get("is_stream"),
    )

    kg_chunks: List[Dict[str, Any]] = []
    kg_query_summary: Dict[str, Any] = {}

    try:
        if not should_run_retriever(state, "kg"):
            logger.info("node_query_kg: 按题型计划跳过图谱查询")
            return {"kg_chunks": kg_chunks, "kg_query_summary": kg_query_summary}

        item_names = state.get("item_names") or []
        question = state.get("rewritten_query") or state.get("original_query") or ""
        query_type = state.get("query_type") or "general"
        focus_terms = state.get("query_focus_terms") or []
        retrieval_plan = state.get("retrieval_plan") or {}
        if not item_names and query_type == "general":
            logger.info("node_query_kg: item_names 为空且非图优先题型，跳过图谱查询")
            return {"kg_chunks": kg_chunks, "kg_query_summary": kg_query_summary}

        # 验证 Neo4j 连接
        if not verify_connection():
            logger.warning("node_query_kg: Neo4j 连接不可用，跳过图谱查询")
            return {"kg_chunks": kg_chunks, "kg_query_summary": kg_query_summary}

        graph_result = query_graph_context(
            question,
            item_names,
            query_type=query_type,
            focus_terms=focus_terms,
            limit=int(retrieval_plan.get("graph_limit", 8) or 8),
        )
        kg_chunks = graph_result.get("kg_chunks") or []
        kg_query_summary = graph_result.get("summary") or {}
        logger.info(
            f"node_query_kg: 查询完成, item_names={item_names}, query_type={query_type}, "
            f"返回 {len(kg_chunks)} 条切片"
        )

    except Exception as e:
        # 图谱查询失败不应阻断主流程
        logger.exception("node_query_kg 执行失败")
    finally:
        add_done_task(
            state["session_id"],
            sys._getframe().f_code.co_name,
            state.get("is_stream"),
        )

    return {"kg_chunks": kg_chunks, "kg_query_summary": kg_query_summary}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("\n>>> 启动 node_query_kg 本地测试")

    test_state = {
        "session_id": "test_kg_query_001",
        "item_names": ["测试产品_KG"],
        "is_stream": False,
    }

    try:
        result = node_query_kg(test_state)
        chunks = result.get("kg_chunks", [])
        print(f"✅ 查询到 {len(chunks)} 条切片")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] chunk_id={c.get('chunk_id')}, title={c.get('title')}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
