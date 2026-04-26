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
    anchor_chunks: list  # Anchor 标题/章节精确召回切片
    kg_chunks: list  # 图谱检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档
    rerank_diagnostics: dict  # 重排序运行诊断：模式、候选数、fallback、cache 等

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
    query_type: str  # 问题类型：general / navigation / comparison / relation / constraint / explain
    query_complexity: str  # 复杂度：simple / complex
    query_complexity_reason: str  # 复杂度判定原因
    graph_preferred: bool  # 当前问题是否优先走图谱检索
    router_decision: str  # 路由决策：default_path / simple_fast_path / complex_deep_path
    router_deep_search_enabled: bool  # 是否启用 Router 控制深检索
    crag_router_enabled: bool  # 当前问题是否允许执行 CRAG Retry
    grounded_mode: bool  # 是否启用严格 grounded 回答约束
    query_focus_terms: list  # 图谱检索时使用的焦点词
    query_anchor_targets: list  # Anchor 检索识别出的标题/章节目标
    router_query_family: str  # 路由问题族：section_lookup / comparison / multi_hop_relation 等
    query_route_reason: str  # 问题类型识别原因
    retrieval_plan: dict  # 按题型生成的检索执行计划
    route_overrides: dict  # 运行期动态路由调整（补救重试/评测开关）
    kg_query_summary: dict  # 图谱查询摘要（命中模板、结果数等）
    sub_query_routes: list  # 复合问题中每个子查询对应的路由结果
    sub_query_results: list  # 复合问题中每个子查询的检索摘要
    context_expansion_summary: dict  # 上下文扩展摘要
    evidence_coverage_summary: dict  # 证据覆盖检查摘要
    target_coverage: dict  # Anchor target 覆盖摘要
    evidence_pack_summary: dict  # 最终回答证据包摘要
    context_budget_chars: int  # 回答上下文预算
    anchor_hits: list  # Anchor 命中摘要
    rescue_plan: dict  # 检索补救计划
    judge_skipped_reason: str  # 检索/幻觉 LLM Judge 被规则跳过的原因
    retrieval_judge_skipped_reason: str  # 检索质量 LLM Judge 跳过原因
    hallucination_judge_skipped_reason: str  # 幻觉自检 LLM Judge 跳过原因
    answer_plan: dict  # 结构化回答规划
    clarification_reason: str  # 触发澄清的原因
    evaluation_mode: bool
    evaluation_overrides: dict
    evaluation_variant_name: str
    message_id: str

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
