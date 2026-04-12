"""
LangGraph 节点：将文档切片导入 Neo4j 知识图谱
位于导入流水线末端，在 Milvus 入库之后执行

图谱结构：
  (:Product {name}) -[:HAS_CHUNK]-> (:Chunk {chunk_id, content, ...})
  (:Chunk)-[:NEXT]->(:Chunk)
"""

import sys
from typing import Dict, Any

from app.import_process.agent.state import ImportGraphState
from app.clients.neo4j_utils import import_chunks_to_kg
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def node_import_kg(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Neo4j 知识图谱导入节点

    前置条件：
      - state["chunks"] 已包含 chunk_id（由 node_import_milvus 回填）
      - state["item_name"] 已由 node_item_name_recognition 识别

    执行逻辑：
      1. 从 state 中提取 item_name 和 chunks
      2. 调用 neo4j_utils.import_chunks_to_kg 批量写入
      3. 幂等：自动清理同 item_name 的旧数据后再写入
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行 LangGraph 节点：{current_node}（Neo4j 知识图谱导入）")
    add_running_task(state["task_id"], current_node)

    try:
        item_name = (state.get("item_name") or "").strip()
        chunks = state.get("chunks") or []

        if not item_name:
            logger.warning("node_import_kg: item_name 为空，跳过知识图谱导入")
            return state

        if not chunks:
            logger.warning("node_import_kg: chunks 为空，跳过知识图谱导入")
            return state

        # 校验 chunk_id 是否存在（由 node_import_milvus 回填）
        sample = chunks[0]
        if not sample.get("chunk_id"):
            logger.warning(
                "node_import_kg: 首个 chunk 缺少 chunk_id，"
                "可能 node_import_milvus 未执行或回填失败，跳过"
            )
            return state

        count = import_chunks_to_kg(item_name, chunks)
        logger.info(f"node_import_kg 完成: {item_name}, 写入 {count} 个节点")

    except Exception as e:
        # 知识图谱导入失败不应阻断主流程（Milvus 已入库成功）
        logger.error(f"node_import_kg 执行失败: {e}", exc_info=True)
    finally:
        add_done_task(state["task_id"], current_node)

    return state


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print(">>> 启动 node_import_kg 本地测试")

    test_state = {
        "task_id": "test_kg_import",
        "item_name": "测试产品_KG",
        "chunks": [
            {
                "chunk_id": "test_chunk_1",
                "content": "这是第一个测试切片，介绍产品基本参数。",
                "title": "产品概述",
                "parent_title": "测试手册",
                "part": 1,
                "file_title": "测试产品手册.pdf",
                "item_name": "测试产品_KG",
            },
            {
                "chunk_id": "test_chunk_2",
                "content": "这是第二个测试切片，介绍安装步骤。",
                "title": "安装指南",
                "parent_title": "测试手册",
                "part": 2,
                "file_title": "测试产品手册.pdf",
                "item_name": "测试产品_KG",
            },
        ],
    }

    try:
        result = node_import_kg(test_state)
        print("✅ node_import_kg 测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
