# HyDE节点
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.conf.query_threshold_config import query_threshold_config
from app.query_process.agent.graph_query_utils import should_run_retriever
from app.query_process.agent.retrieval_utils import run_embedding_hybrid_search


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
    HyDE的核心在于：利用LLM生成一个"虚构但相关"的文档，用该文档的向量去检索真实的文档，
    从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。
    """
    if not rewritten_query:
        logger.error("Step 1 Error: rewritten_query 为空")
        raise ValueError("rewritten_query 不能为空")

    logger.info(f"Step 1: 开始生成假设性文档 (HyDE), Query: {rewritten_query}")

    try:
        llm = get_llm_client()
        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        logger.debug(f"Step 1: Prompt加载成功, 长度: {len(hyde_prompt)}")

        response = llm.invoke(hyde_prompt)
        hyde_doc = coerce_llm_content(response.content)

        logger.info(f"Step 1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
        logger.debug(f"Step 1: 文档预览: {hyde_doc[:50]}...")

        return hyde_doc

    except Exception:
        logger.exception("Step 1: 生成假设文档失败")
        raise


def step_2_search_by_query_and_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),
    norm_score: bool = True,
    output_fields=None,
):
    """
    阶段2：分别对 Query 和 (Query + 假设文档) 执行检索，RRF 融合两路结果。

    解决 HyDE 原方案中 Query 被假设文档稀释的问题：
    - 原始方案：Query + 假设文档拼接后一次性向量化检索，短 Query 被长假设文档主导
    - 修复方案：Query 单独检索 + (Query+假设文档) 单独检索，分别得到相关文档，再 RRF 融合

    :param rewritten_query: 改写后的查询
    :param hyde_doc: Step 1 生成的假设性文档
    :param item_names: 商品名称列表，用于元数据过滤
    :param req_limit: Milvus 搜索时的候选召回数量
    :param top_k: 最终返回的 Top K 结果数量
    :param ranker_weights: 混合检索权重 (Dense, Sparse)
    :param norm_score: 是否对分数进行归一化
    :param output_fields: 返回结果中包含的字段
    :return: RRF 融合后的检索结果列表
    """
    if output_fields is None:
        output_fields = ["chunk_id", "content", "item_name", "image_urls"]

    if not rewritten_query:
        raise ValueError("rewritten_query 不能为空")
    if not hyde_doc:
        raise ValueError("hypothetical_doc 不能为空")

    cfg = query_threshold_config

    # ---- 2a. Query 单独检索 ----
    logger.info("Step 2a: Query 单独向量化并检索...")
    try:
        query_chunks = run_embedding_hybrid_search(
            query_text=rewritten_query,
            item_names=item_names,
            req_limit=req_limit,
            top_k=top_k,
            output_fields=list(output_fields),
            ranker_weights=ranker_weights,
            norm_score=norm_score,
        )
        logger.info(f"Step 2a: Query 单独检索完成，召回 {len(query_chunks)} 条")
    except Exception:
        logger.exception("Step 2a: Query 单独检索失败")
        query_chunks = []

    # ---- 2b. (Query + 假设文档) 拼接检索 ----
    combined_text = rewritten_query + " " + hyde_doc
    logger.info(f"Step 2b: (Query+假设文档) 向量化并检索, 总长度: {len(combined_text)}")
    try:
        combined_chunks = run_embedding_hybrid_search(
            query_text=combined_text,
            item_names=item_names,
            req_limit=req_limit,
            top_k=top_k,
            output_fields=list(output_fields),
            ranker_weights=ranker_weights,
            norm_score=norm_score,
        )
        logger.info(
            f"Step 2b: (Query+假设文档) 检索完成，召回 {len(combined_chunks)} 条"
        )
    except Exception:
        logger.exception("Step 2b: (Query+假设文档) 检索失败")
        combined_chunks = []

    if not query_chunks and not combined_chunks:
        logger.warning("Step 2: 两路检索均无结果，返回空列表")
        return []

    from app.query_process.agent.nodes.node_rrf import (
        _as_entity_list,
        reciprocal_rank_fusion,
    )

    query_entities = _as_entity_list(query_chunks)
    combined_entities = _as_entity_list(combined_chunks)

    rrf_results = reciprocal_rank_fusion(
        source_weights=[(query_entities, 1.0), (combined_entities, 1.0)],
        k=cfg.rrf_k,
        max_results=top_k,
    )

    merged = []
    for doc, score in rrf_results:
        merged.append(
            {
                "entity": doc,
                "distance": score,
                "id": doc.get("chunk_id") or doc.get("id"),
            }
        )

    logger.info(f"Step 2: RRF 融合完成，输出 {len(merged)} 条结果")
    return merged


def node_search_embedding_hyde(state):
    """
    HyDE (Hypothetical Document Embedding) 检索节点

    核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

    执行步骤：
    1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
    2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
    3. 双路检索 + RRF 融合 (Step 2)：
       - Query 单独向量化并检索
       - (Query + 假设文档) 拼接后向量化并检索
       - RRF 融合两路结果，解决 Query 被稀释的问题
    4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。
    """
    logger.info("---HyDE (假设文档检索) 节点开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    if not should_run_retriever(state, "hyde"):
        logger.info("HyDE 检索按题型计划跳过")
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        rewritten_query = state.get("original_query")

    if not rewritten_query:
        logger.error("HyDE节点错误: 未找到有效的用户查询")
        return {}

    item_names = state.get("item_names")
    logger.info(f"HyDE检索入参: query='{rewritten_query}', item_names={item_names}")

    # 阶段1：生成假设性文档
    hyde_doc = ""
    try:
        logger.info("Step 1: 开始生成假设性文档 (HyDE Doc)...")
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
        logger.info(f"Step 1: 假设文档生成成功 (长度: {len(hyde_doc)})")
        logger.debug(f"假设文档预览: {hyde_doc[:100]}...")
    except Exception:
        logger.exception("Step 1 (生成假设文档) 发生异常")
        return {}

    # 阶段2：双路检索 + RRF 融合
    try:
        logger.info("Step 2: 开始双路检索 + RRF 融合...")
        cfg = query_threshold_config
        res = step_2_search_by_query_and_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            req_limit=cfg.hyde_req_limit,
            top_k=cfg.hyde_top_k,
            ranker_weights=(cfg.hybrid_dense_weight, cfg.hybrid_sparse_weight),
        )

        hit_count = len(res) if res else 0
        logger.info(f"Step 2: 双路检索+RRF融合完成，召回 {hit_count} 条相关切片")

        if hit_count > 0:
            first_hit = res[0]
            score = first_hit.get("distance")
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]
            logger.debug(f"Top1 结果: Score={score}, Content='{content_preview}...'")

        return {
            "hyde_embedding_chunks": res if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception:
        logger.exception("Step 2 (双路检索+RRF融合) 发生异常")
        return {}
    finally:
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        logger.info("---HyDE 节点处理结束---")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }

    try:
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)}")
        if chunks:
            print(f"Top Chunk Distance: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
