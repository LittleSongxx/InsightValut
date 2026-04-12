"""
Query 流程阈值配置

集中管理 RAG 流程中所有 hard-coded 的阈值常量，便于根据不同模型/场景调整。
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryThresholdConfig:
    # --- 商品名确认阈值 ---
    # 高置信度阈值：超过此值的商品名校正结果直接确认
    item_name_high_threshold: float
    # 中置信度阈值：超过此值进入候选澄清流程
    item_name_mid_threshold: float

    # --- 检索召回数量 ---
    # Embedding 检索的底层召回数量（limit 参数）
    embedding_req_limit: int
    # HyDE 检索的底层召回数量
    hyde_req_limit: int
    # 每路检索最终返回的 TopK
    embedding_top_k: int
    hyde_top_k: int

    # --- RRF 融合参数 ---
    # RRF 融合的 k 常数（经典值为 60）
    rrf_k: int
    # RRF 最大输出数量
    rrf_max_results: int
    # 各路检索的 RRF 权重
    rrf_weight_embedding: float
    rrf_weight_hyde: float
    rrf_weight_kg: float

    # --- Rerank 参数 ---
    # Rerank 动态 TopK 硬上限
    rerank_max_topk: int
    # Rerank 动态 TopK 硬下限
    rerank_min_topk: int
    # Rerank 断崖截断 - 绝对阈值（分数差）
    rerank_gap_abs: float
    # Rerank 断崖截断 - 相对阈值（分数比例）
    rerank_gap_ratio: float

    # --- CRAG 参数 ---
    # CRAG 最大重试次数
    crag_max_retries: int
    # CRAG 最少所需文档数
    crag_min_docs: int

    # --- 幻觉检查参数 ---
    # 幻觉检查最大重试次数
    hallucination_max_retries: int

    # --- HyDE 参数 ---
    # HyDE 假设文档最大字符数
    hyde_max_doc_chars: int
    # HyDE 假设文档与 Query 拼接后的最大字符数
    hyde_combined_max_chars: int

    # --- 复合问题分解 ---
    # 复合问题最大子查询数量
    max_sub_queries: int

    # --- Prompt 上下文限制 ---
    # 答案生成时上下文最大字符数
    max_context_chars: int
    # 幻觉检查/CRAG 评估时文档摘要最大字符数
    grader_doc_max_chars: int


# 工厂函数：从环境变量加载配置，无则使用默认值
def _float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


query_threshold_config = QueryThresholdConfig(
    # 商品名确认阈值
    item_name_high_threshold=_float(os.getenv("ITEM_NAME_HIGH_THRESHOLD"), 0.85),
    item_name_mid_threshold=_float(os.getenv("ITEM_NAME_MID_THRESHOLD"), 0.60),

    # 检索召回数量
    embedding_req_limit=_int(os.getenv("EMBEDDING_REQ_LIMIT"), 10),
    hyde_req_limit=_int(os.getenv("HYDE_REQ_LIMIT"), 10),
    embedding_top_k=_int(os.getenv("EMBEDDING_TOP_K"), 5),
    hyde_top_k=_int(os.getenv("HYDE_TOP_K"), 5),

    # RRF 融合参数
    rrf_k=_int(os.getenv("RRF_K"), 60),
    rrf_max_results=_int(os.getenv("RRF_MAX_RESULTS"), 10),
    rrf_weight_embedding=_float(os.getenv("RRF_WEIGHT_EMBEDDING"), 1.0),
    rrf_weight_hyde=_float(os.getenv("RRF_WEIGHT_HYDE"), 1.0),
    rrf_weight_kg=_float(os.getenv("RRF_WEIGHT_KG"), 0.8),  # 启用 KG 路，与 Embedding/HyDE 互补

    # Rerank 参数
    rerank_max_topk=_int(os.getenv("RERANK_MAX_TOPK"), 10),
    rerank_min_topk=_int(os.getenv("RERANK_MIN_TOPK"), 1),
    rerank_gap_abs=_float(os.getenv("RERANK_GAP_ABS"), 0.5),
    rerank_gap_ratio=_float(os.getenv("RERANK_GAP_RATIO"), 0.25),

    # CRAG 参数
    crag_max_retries=_int(os.getenv("CRAG_MAX_RETRIES"), 1),
    crag_min_docs=_int(os.getenv("CRAG_MIN_DOCS"), 1),

    # 幻觉检查参数
    hallucination_max_retries=_int(os.getenv("HALLUCINATION_MAX_RETRIES"), 1),

    # HyDE 参数
    hyde_max_doc_chars=_int(os.getenv("HYDE_MAX_DOC_CHARS"), 300),
    hyde_combined_max_chars=_int(os.getenv("HYDE_COMBINED_MAX_CHARS"), 1000),

    # 复合问题分解
    max_sub_queries=_int(os.getenv("MAX_SUB_QUERIES"), 4),

    # Prompt 上下文限制
    max_context_chars=_int(os.getenv("MAX_CONTEXT_CHARS"), 12000),
    grader_doc_max_chars=_int(os.getenv("GRADER_DOC_MAX_CHARS"), 6000),
)
