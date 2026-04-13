from typing_extensions import TypedDict
from typing import List


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """

    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    bm25_chunks: list  # BM25 检索回来的切片
    kg_chunks: list  # 图谱检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记
    need_rag: bool  # 是否需要走RAG检索流程
    route_mode: str  # 路由模式：rag / general_chat

    # CRAG（Corrective RAG）检索质量判断
    retry_count: int  # 当前检索重试次数
    retrieval_grade: str  # 检索质量评级：sufficient / insufficient / retry

    # 复合问题分解
    sub_queries: list  # 分解后的子查询列表
    is_compound_query: bool  # 是否为复合查询

    # 答案幻觉自检
    hallucination_check_passed: bool  # 幻觉检查是否通过
    hallucination_retry_count: int  # 幻觉重试计数
    hallucination_feedback: str  # 幻觉反馈信息，用于重新生成时的约束
