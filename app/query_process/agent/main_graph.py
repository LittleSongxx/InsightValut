from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
from app.utils.perf_tracker import (
    perf_begin_stage,
    perf_end_stage,
    perf_mark_first_answer,
)

# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_decompose import node_query_decompose
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_answer_plan import node_answer_plan
from app.query_process.agent.nodes.node_context_expand import node_context_expand
from app.query_process.agent.nodes.node_evidence_coverage import (
    node_evidence_coverage,
)
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_bm25 import node_search_bm25
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    node_search_embedding_hyde,
)
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_retrieval_grader import node_retrieval_grader
from app.query_process.agent.nodes.node_hallucination_check import (
    node_hallucination_check,
)


def _perf_wrap(node_name: str, fn):
    """性能埋点装饰器：自动记录每个节点的耗时"""

    def wrapper(state):
        session_id = state.get("session_id", "")
        perf_begin_stage(session_id, node_name)
        try:
            result = fn(state)
            perf_end_stage(session_id, node_name, status="success")
            # 答案生成节点完成时，标记首次回答时间
            if node_name == "node_answer_output" and result and result.get("answer"):
                perf_mark_first_answer(session_id)
            return result
        except Exception as e:
            perf_end_stage(session_id, node_name, status="error", error=str(e))
            raise

    return wrapper


# 初始化状态图
builder = StateGraph(QueryGraphState)

# ===================== 注册所有节点 =====================
builder.add_node(
    "node_item_name_confirm",
    _perf_wrap("node_item_name_confirm", node_item_name_confirm),
)  # 确认商品
builder.add_node(
    "node_query_decompose", _perf_wrap("node_query_decompose", node_query_decompose)
)  # 复合问题分解
builder.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点
builder.add_node("node_multi_search_graph_first", lambda x: x)
builder.add_node(
    "node_search_embedding", _perf_wrap("node_search_embedding", node_search_embedding)
)  # 向量搜索
builder.add_node("node_search_bm25", _perf_wrap("node_search_bm25", node_search_bm25))
builder.add_node(
    "node_search_embedding_hyde",
    _perf_wrap("node_search_embedding_hyde", node_search_embedding_hyde),
)
builder.add_node("node_query_kg", _perf_wrap("node_query_kg", node_query_kg))
builder.add_node(
    "node_query_kg_primary", _perf_wrap("node_query_kg_primary", node_query_kg)
)
builder.add_node(
    "node_web_search_mcp", _perf_wrap("node_web_search_mcp", node_web_search_mcp)
)
builder.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
builder.add_node("node_rrf", _perf_wrap("node_rrf", node_rrf))  # 排序
builder.add_node(
    "node_context_expand", _perf_wrap("node_context_expand", node_context_expand)
)
builder.add_node("node_rerank", _perf_wrap("node_rerank", node_rerank))  # 重排
builder.add_node(
    "node_evidence_coverage",
    _perf_wrap("node_evidence_coverage", node_evidence_coverage),
)
builder.add_node(
    "node_retrieval_grader", _perf_wrap("node_retrieval_grader", node_retrieval_grader)
)  # CRAG 检索质量判断
builder.add_node(
    "node_answer_plan", _perf_wrap("node_answer_plan", node_answer_plan)
)
builder.add_node(
    "node_answer_output", _perf_wrap("node_answer_output", node_answer_output)
)  # 生成
builder.add_node(
    "node_hallucination_check",
    _perf_wrap("node_hallucination_check", node_hallucination_check),
)  # 幻觉自检

# 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑；
# lambda x:x 含义：接收 state 并原样返回，是最轻便的 "无逻辑传递" 方式；

# ===================== 设置起点 =====================
builder.set_entry_point("node_item_name_confirm")


# ===================== 条件路由函数 =====================
def route_after_item_confirm(state: QueryGraphState):
    """意图确认后的路由：直接输出 / 进入复合问题分解"""
    # 如果已有答案（如候选澄清/未命中），直接输出
    if state.get("answer"):
        return "node_answer_plan"
    # 若判定为通用对话，不走RAG检索，直接由生成节点回答
    if state.get("need_rag") is False:
        return "node_answer_plan"
    if state.get("graph_preferred"):
        return "node_query_kg_primary"
    # 否则进入复合问题分解（新增）
    return "node_query_decompose"


def route_after_decompose(state: QueryGraphState):
    """复合问题分解后的路由：复合查询已内联检索 / 简单查询走正常多路搜索"""
    if state.get("is_compound_query") and (
        state.get("rrf_chunks") or state.get("web_search_docs")
    ):
        # 复合查询已在 node_query_decompose 中完成内联检索并生成 rrf_chunks
        # 先做命中上下文扩展，再进入 rerank
        return "node_context_expand"
    if state.get("graph_preferred"):
        return "node_query_kg_primary"
    # 简单查询走正常多路搜索
    return "node_multi_search"


def route_after_grading(state: QueryGraphState):
    """CRAG 质量判断后的路由：通过 / 重试（重新确认商品名）/ 不足"""
    grade = state.get("retrieval_grade", "sufficient")
    if grade == "retry":
        # 检索质量不足，回退到商品名确认节点重新提取商品名和改写查询
        # 这样可以确保 suggested_query 涉及不同商品时，item_names 能被正确更新
        return "node_item_name_confirm"
    # sufficient 或 insufficient（重试耗尽）均进入答案生成
    return "node_answer_plan"


def route_after_hallucination_check(state: QueryGraphState):
    """幻觉检查后的路由：通过 → 结束 / 未通过 → 重新生成"""
    if state.get("hallucination_check_passed", True):
        return END
    # 幻觉未通过，回退到答案生成重新生成（answer 已被清空）
    return "node_answer_plan"


# ===================== 注册边和条件边 =====================

# 1. 意图确认 → (条件分叉) → 复合问题分解 / 答案输出
builder.add_conditional_edges("node_item_name_confirm", route_after_item_confirm)

# 2. 复合问题分解 → (条件分叉) → 多路搜索 / 直接重排
builder.add_conditional_edges("node_query_decompose", route_after_decompose)

# 3. 并发执行五路搜索
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_bm25")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")
builder.add_edge("node_multi_search", "node_query_kg")

builder.add_edge("node_query_kg_primary", "node_multi_search_graph_first")
builder.add_edge("node_multi_search_graph_first", "node_search_embedding")
builder.add_edge("node_multi_search_graph_first", "node_search_bm25")
builder.add_edge("node_multi_search_graph_first", "node_search_embedding_hyde")
builder.add_edge("node_multi_search_graph_first", "node_web_search_mcp")

# 4. 五路搜索 → 结果合并
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_bm25", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 5. 合并 → 排序 → 重排 → CRAG质量判断
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_context_expand")
builder.add_edge("node_context_expand", "node_rerank")
builder.add_edge("node_rerank", "node_evidence_coverage")
builder.add_edge("node_evidence_coverage", "node_retrieval_grader")

# 6. CRAG质量判断 → (条件分叉) → 回答规划 / 回退重试
builder.add_conditional_edges("node_retrieval_grader", route_after_grading)

# 7. 回答规划 → 答案生成 → 幻觉自检
builder.add_edge("node_answer_plan", "node_answer_output")
builder.add_edge("node_answer_output", "node_hallucination_check")

# 8. 幻觉自检 → (条件分叉) → 结束 / 回退重新生成
builder.add_conditional_edges(
    "node_hallucination_check", route_after_hallucination_check
)

# ===================== 编译 =====================
# recursion_limit 防止 CRAG/幻觉检查循环无限执行（默认25，此处显式设置）
query_app = builder.compile()
