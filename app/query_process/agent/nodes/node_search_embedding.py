import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.query_process.agent.retrieval_utils import run_embedding_hybrid_search


def node_search_embedding(state):
    """
    核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
    流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
    :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                  {
                      "session_id": str,        # 会话唯一标识
                      "rewritten_query": str,   # step3改写后的完整用户问题（含商品名）
                      "item_names": list[str],  # step6已确认的标准化商品名列表
                      "is_stream": bool/None    # 是否为流式响应，可选
                  }
    :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
             {
                 "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                 # 每个元素为一条匹配的向量数据，含业务字段
             }
    """
    logger.info("---search_milvus 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state["is_stream"]
    )

    # 1. 从会话状态中提取核心入参，为后续检索做准备
    query = state.get("rewritten_query")  # 提取改写后的用户问题（含商品名，独立完整）
    item_names = state.get("item_names")  # 提取已确认的标准化商品名列表（精准过滤用）

    logger.info(f"核心入参提取: query='{query}', item_names={item_names}")

    results = run_embedding_hybrid_search(
        query_text=query,
        item_names=item_names,
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

    # 打印节点处理成功日志，输出原始检索结果，便于调试
    hit_count = len(results)
    logger.info(f"节点 search_embedding 处理成功，检索到 {hit_count} 条相关片段")
    if hit_count > 0:
        logger.debug(f"Top1 检索结果示例: {results[0]}")

    # 标记当前任务完成，更新任务状态
    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    # 6. 构造并返回结果：若检索结果非空，取res[0]（适配Milvus批量搜索格式），否则返回空列表
    # res[0]为当前单条查询的检索结果，包含TOP5匹配的向量数据及业务字段
    return {"embedding_chunks": results}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False,
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get("entity", {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print(
                "\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配"
            )

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
