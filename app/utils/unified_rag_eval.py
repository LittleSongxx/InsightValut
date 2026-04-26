import argparse
import hashlib
import copy
import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from app.clients.milvus_schema import extract_chunk_id, normalize_entity_for_business_id

try:
    from langchain_core.embeddings import Embeddings
except ImportError:

    class Embeddings:
        pass


from app.clients.milvus_utils import get_milvus_client, query_chunks_by_filter
from app.query_process.agent.agentic_utils import DEFAULT_AGENTIC_FEATURES
from app.lm.lm_utils import coerce_llm_content, get_llm_client
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.retrieval_utils import build_item_name_filter_expr
from app.utils.bm25_utils import rank_documents_bm25, tokenize_text
from app.utils.path_util import PROJECT_ROOT
from app.utils.perf_tracker import perf_finish, perf_start
from app.utils.query_cache_utils import (
    get_current_request_cache_summary,
    query_cache_request_context,
    reset_query_cache,
)
from app.utils.retrieval_eval import load_cases, mrr_at_k, normalize_ids, recall_at_k
from app.utils.task_utils import clear_task

try:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        FactualCorrectness,
        IDBasedContextPrecision,
        IDBasedContextRecall,
    )

    try:
        from ragas.metrics import LLMContextRecall
    except ImportError:
        from ragas.metrics import ContextRecall as LLMContextRecall
    try:
        from ragas.metrics import ResponseRelevancy
    except ImportError:
        from ragas.metrics import AnswerRelevancy as ResponseRelevancy
except ImportError as exc:
    EvaluationDataset = None
    evaluate = None
    Faithfulness = None
    FactualCorrectness = None
    IDBasedContextPrecision = None
    IDBasedContextRecall = None
    LLMContextRecall = None
    ResponseRelevancy = None
    _RAGAS_IMPORT_ERROR = exc
else:
    _RAGAS_IMPORT_ERROR = None

DEFAULT_K_VALUES = [1, 3, 5]
GROUND_TRUTH_OUTPUT_FIELDS = [
    "chunk_id",
    "stable_chunk_id",
    "item_name",
    "title",
    "parent_title",
    "part",
    "file_title",
    "content",
]
GROUND_TRUTH_SEARCH_LIMIT = int(
    os.environ.get("EVAL_GROUND_TRUTH_SEARCH_LIMIT") or 1000
)
DEFAULT_VARIANTS = [
    "baseline_rag",
    "bm25_hybrid",
    "hyde_hybrid",
    "kg_hybrid",
    "bm25_kg_hyde_hybrid",
    "final_system",
    "agentic_context_expansion",
    "agentic_rescue_system",
    "agentic_enhanced_system",
    "agentic_enhanced_system_cached",
    "router_anchor_contextual_grounded_cached",
    "router_anchor_rescue_structured_cached",
]

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
CancelCallback = Optional[Callable[[], bool]]


class EvaluationCancelledError(RuntimeError):
    """Raised when an evaluation job is cancelled by the user."""


def _raise_if_cancelled(cancel_callback: CancelCallback) -> None:
    if cancel_callback is not None and cancel_callback():
        raise EvaluationCancelledError("用户已取消评测任务")


def _agentic_features(**overrides: bool) -> Dict[str, bool]:
    features = {key: False for key in DEFAULT_AGENTIC_FEATURES.keys()}
    features.update(overrides)
    return features


FEATURE_CATEGORY_LABELS: Dict[str, str] = {
    "retrieval": "检索增强",
    "agentic": "Agentic",
    "quality": "质量控制",
    "performance": "性能缓存",
    "external": "外部功能",
}

EVALUATION_FEATURE_CATALOG: List[Dict[str, Any]] = [
    {
        "key": "bm25",
        "label": "BM25",
        "category": "retrieval",
        "description": "启用 BM25 稀疏检索并参与融合排序。",
    },
    {
        "key": "hyde",
        "label": "HyDE",
        "category": "retrieval",
        "description": "生成假设文档辅助向量检索。",
    },
    {
        "key": "kg",
        "label": "Neo4j KG",
        "category": "retrieval",
        "description": "启用 Neo4j 图谱检索作为额外证据源。",
    },
    {
        "key": "anchor",
        "label": "Anchor",
        "category": "retrieval",
        "description": "启用标题锚点与目标覆盖检索。",
    },
    {
        "key": "router",
        "label": "Router",
        "category": "quality",
        "description": "启用 Router 控制复杂问题深检索路径。",
    },
    {
        "key": "crag_retry",
        "label": "CRAG Retry",
        "category": "quality",
        "description": "启用检索质量评估和传统 CRAG 重试。",
    },
    {
        "key": "grounded_answer",
        "label": "Grounded Answer",
        "category": "quality",
        "description": "启用基于证据包的 grounded 回答约束。",
    },
    {
        "key": "subquery_routing",
        "label": "Subquery Routing",
        "category": "agentic",
        "description": "启用复杂问题子问题级路由。",
    },
    {
        "key": "context_expansion",
        "label": "Context Expansion",
        "category": "agentic",
        "description": "命中片段后补充章节或相邻上下文。",
    },
    {
        "key": "evidence_coverage",
        "label": "Evidence Coverage",
        "category": "agentic",
        "description": "计算证据覆盖并为补救提供依据。",
    },
    {
        "key": "retrieval_rescue",
        "label": "Retrieval Rescue",
        "category": "agentic",
        "description": "证据不足时进行受控检索补救。",
        "dependencies": ["evidence_coverage", "subquery_routing"],
    },
    {
        "key": "structured_answer",
        "label": "Structured Answer",
        "category": "agentic",
        "description": "对高风险问题启用结构化回答规划。",
        "dependencies": ["grounded_answer"],
    },
    {
        "key": "clarification_guard",
        "label": "Clarification Guard",
        "category": "agentic",
        "description": "证据或问题条件不足时优先澄清。",
    },
    {
        "key": "cache",
        "label": "Cache",
        "category": "performance",
        "description": "启用查询多级缓存，评测前重置并预热一轮。",
    },
    {
        "key": "web_search",
        "label": "Web Search",
        "category": "external",
        "description": "启用外部联网检索，网络波动会计入指标。",
    },
]

_FEATURE_MAP = {item["key"]: item for item in EVALUATION_FEATURE_CATALOG}
_FEATURE_ORDER = {item["key"]: index for index, item in enumerate(EVALUATION_FEATURE_CATALOG)}
CONTROLLED_BASELINE_VARIANT = "combo_baseline"


def get_feature_catalog() -> List[Dict[str, Any]]:
    return copy.deepcopy(EVALUATION_FEATURE_CATALOG)


def _normalize_feature_keys(features: Sequence[Any] | None) -> List[str]:
    ordered: List[str] = []
    for raw_key in features or []:
        key = str(raw_key or "").strip()
        if not key:
            continue
        if key not in _FEATURE_MAP:
            raise ValueError(f"未知评测功能: {key}")
        if key not in ordered:
            ordered.append(key)
    return sorted(ordered, key=lambda item: _FEATURE_ORDER[item])


def _resolve_feature_dependencies(features: Sequence[str]) -> Dict[str, List[str]]:
    requested = set(features)
    resolved = set(features)
    changed = True
    while changed:
        changed = False
        for feature_key in list(resolved):
            dependencies = _FEATURE_MAP.get(feature_key, {}).get("dependencies") or []
            for dependency in dependencies:
                if dependency not in _FEATURE_MAP:
                    continue
                if dependency not in resolved:
                    resolved.add(dependency)
                    changed = True
    ordered_resolved = sorted(resolved, key=lambda item: _FEATURE_ORDER[item])
    ordered_auto = [
        key for key in ordered_resolved if key not in requested
    ]
    return {"resolved": ordered_resolved, "auto_enabled": ordered_auto}


def _feature_labels(feature_keys: Sequence[str]) -> List[str]:
    return [str(_FEATURE_MAP[key].get("label") or key) for key in feature_keys]


def _feature_variant_name(feature_keys: Sequence[str]) -> str:
    if not feature_keys:
        return CONTROLLED_BASELINE_VARIANT
    digest = hashlib.sha1(
        json.dumps(list(feature_keys), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    return f"combo_{digest}"


def _build_feature_variant_overrides(feature_keys: Sequence[str]) -> Dict[str, Any]:
    features = set(feature_keys)
    agentic_features = _agentic_features(
        subquery_routing="subquery_routing" in features,
        context_expansion="context_expansion" in features,
        evidence_coverage="evidence_coverage" in features,
        retrieval_rescue="retrieval_rescue" in features,
        structured_answer="structured_answer" in features,
        clarification_guard="clarification_guard" in features,
    )
    overrides: Dict[str, Any] = {
        "force_need_rag": True,
        "force_graph_preferred": False,
        "bm25_enabled": "bm25" in features,
        "retrieval_grader_enabled": bool(
            {"router", "crag_retry", "retrieval_rescue"} & features
        ),
        "legacy_retry_enabled": "crag_retry" in features,
        "router_deep_search_enabled": "router" in features,
        "grounded_answer_enabled": "grounded_answer" in features,
        "anchor_context_enabled": "anchor" in features,
        "route_reason": (
            "eval_controlled_baseline"
            if not features
            else f"eval_dynamic_{_feature_variant_name(feature_keys)}"
        ),
        "agentic_features": agentic_features,
        "cache_enabled": "cache" in features,
        "retrieval_plan_overrides": {
            "graph_first": False,
            "run_kg": "kg" in features,
            "run_embedding": True,
            "run_anchor": "anchor" in features,
            "run_bm25": "bm25" in features,
            "run_hyde": "hyde" in features,
            "run_web": "web_search" in features,
            "kg_weight_multiplier": 1.2 if "kg" in features else 0.0,
            "anchor_weight_multiplier": 1.6 if "anchor" in features else 0.0,
            "bm25_weight_multiplier": 1.0 if "bm25" in features else 0.0,
            "hyde_weight_multiplier": 1.0 if "hyde" in features else 0.0,
        },
    }
    if "structured_answer" in features:
        overrides["structured_answer_high_risk_only"] = True
    if "retrieval_rescue" in features:
        overrides["rescue_min_coverage_score"] = 0.72
        overrides["rescue_force_target_coverage"] = True
    return overrides


def build_feature_variant_definition(spec: Dict[str, Any] | None) -> Dict[str, Any]:
    raw_spec = dict(spec or {})
    requested_features = _normalize_feature_keys(raw_spec.get("features") or [])
    dependency_result = _resolve_feature_dependencies(requested_features)
    feature_keys = dependency_result["resolved"]
    auto_enabled = dependency_result["auto_enabled"]
    variant_name = _feature_variant_name(feature_keys)
    labels = _feature_labels(feature_keys)
    default_label = "Baseline" if not labels else "Baseline + " + " + ".join(labels)
    label = str(raw_spec.get("label") or raw_spec.get("name") or default_label).strip()
    if not label:
        label = default_label
    config: Dict[str, Any] = {
        "description": (
            "受控消融基线：embedding 检索 + rerank + answer"
            if not labels
            else f"受控消融组合：{default_label}"
        ),
        "technique": label,
        "compare_to": None if not labels else CONTROLLED_BASELINE_VARIANT,
        "use_case_query_type": True,
        "is_feature_variant": True,
        "feature_variant": {
            "name": variant_name,
            "label": label,
            "requested_features": requested_features,
            "resolved_features": feature_keys,
            "auto_enabled_features": auto_enabled,
            "feature_labels": labels,
            "auto_enabled_feature_labels": _feature_labels(auto_enabled),
        },
        "evaluation_overrides": _build_feature_variant_overrides(feature_keys),
    }
    if "cache" in feature_keys:
        config["warmup_rounds"] = 1
        config["reset_cache_before_run"] = True
    return {"name": variant_name, "config": config}


def register_feature_variant(spec: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = build_feature_variant_definition(spec)
    name = payload["name"]
    VARIANTS[name] = copy.deepcopy(payload["config"])
    return {
        "name": name,
        **copy.deepcopy(payload["config"].get("feature_variant") or {}),
    }


def register_feature_variants(specs: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    registered: List[Dict[str, Any]] = []
    for spec in specs or []:
        registered.append(register_feature_variant(spec))
    return registered


def _evaluation_query_service_base_url() -> str:
    return str(os.environ.get("EVAL_QUERY_SERVICE_BASE_URL") or "").strip().rstrip("/")


def _evaluation_query_service_timeout() -> int:
    raw = os.environ.get("EVAL_QUERY_SERVICE_TIMEOUT_SEC") or "600"
    try:
        return max(30, int(raw))
    except (TypeError, ValueError):
        return 600


def _forced_warmup_rounds() -> Optional[int]:
    raw = os.environ.get("EVAL_FORCE_WARMUP_ROUNDS")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


def _reset_query_cache_for_evaluation(service_base_url: str, reason: str) -> None:
    if service_base_url:
        try:
            requests.post(
                f"{service_base_url}/cache/reset",
                json={"reason": reason},
                timeout=30,
            ).raise_for_status()
            return
        except Exception:
            reset_query_cache(reason=reason)
            return
    reset_query_cache(reason=reason)


def _ensure_ragas_ready() -> None:
    if _RAGAS_IMPORT_ERROR is not None:
        raise ImportError(
            "RAGAS 未安装，请先安装项目依赖后再运行统一评测。"
        ) from _RAGAS_IMPORT_ERROR


QUALITY_METRIC_KEYS = [
    "factual_correctness",
    "faithfulness",
    "response_relevancy",
    "llm_context_recall",
]
ID_CONTEXT_METRIC_KEYS = [
    "id_based_context_precision",
    "id_based_context_recall",
]
QUALITY_JUDGE_PROMPT_VERSION = "llm_judge_v2_strict_rubric"
QUALITY_JUDGE_SCORE_BUCKETS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
QUALITY_JUDGE_CONTEXT_CHARS = int(
    os.environ.get("EVAL_JUDGE_CONTEXT_CHARS") or 6000
)
QUALITY_JUDGE_TEXT_CHARS = int(
    os.environ.get("EVAL_JUDGE_TEXT_CHARS") or 1200
)


VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline_rag": {
        "description": "最原始基线：仅 embedding 检索 + rerank + answer",
        "technique": "Baseline RAG",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "retrieval_grader_enabled": False,
            "legacy_retry_enabled": False,
            "route_reason": "eval_baseline_rag",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": False,
                "run_embedding": True,
                "run_bm25": False,
                "run_hyde": False,
                "run_web": False,
                "kg_weight_multiplier": 0.0,
                "bm25_weight_multiplier": 0.0,
                "hyde_weight_multiplier": 0.0,
            },
        },
    },
    "bm25_hybrid": {
        "description": "在最原始基线上引入 BM25",
        "technique": "BM25",
        "compare_to": "baseline_rag",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": False,
            "legacy_retry_enabled": False,
            "route_reason": "eval_bm25_hybrid",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": False,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": False,
                "run_web": False,
                "kg_weight_multiplier": 0.0,
                "hyde_weight_multiplier": 0.0,
            },
        },
    },
    "hyde_hybrid": {
        "description": "在最原始基线上引入 HyDE",
        "technique": "HyDE",
        "compare_to": "baseline_rag",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "retrieval_grader_enabled": False,
            "legacy_retry_enabled": False,
            "route_reason": "eval_hyde_hybrid",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": False,
                "run_embedding": True,
                "run_bm25": False,
                "run_hyde": True,
                "run_web": False,
                "kg_weight_multiplier": 0.0,
                "bm25_weight_multiplier": 0.0,
            },
        },
    },
    "kg_hybrid": {
        "description": "在最原始基线上引入 Neo4j 作为额外检索源，但不启用 graph-first",
        "technique": "Neo4j KG",
        "compare_to": "baseline_rag",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "retrieval_grader_enabled": False,
            "legacy_retry_enabled": False,
            "route_reason": "eval_kg_hybrid",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": False,
                "run_hyde": False,
                "run_web": False,
                "bm25_weight_multiplier": 0.0,
                "hyde_weight_multiplier": 0.0,
            },
        },
    },
    "bm25_kg_hyde_hybrid": {
        "description": "纯 Base 组合：在最原始基线上同时引入 BM25、Neo4j KG 和 HyDE，不启用 Agentic / Router / Context Expansion / CRAG / Cache",
        "technique": "Baseline + BM25 + KG + HyDE",
        "compare_to": "baseline_rag",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": False,
            "legacy_retry_enabled": False,
            "router_deep_search_enabled": False,
            "grounded_answer_enabled": False,
            "anchor_context_enabled": False,
            "route_reason": "eval_bm25_kg_hyde_hybrid",
            "agentic_features": _agentic_features(),
            "cache_enabled": False,
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
    "hyde_kg_context_expansion_cached": {
        "description": "在 Baseline 上开启 HyDE、Neo4j KG、命中上下文扩展、多级缓存和 CRAG 重试，不引入 BM25 / 检索补救 / 结构化回答",
        "technique": "HyDE + KG + Context Expansion + Cache + CRAG Retry",
        "compare_to": "baseline_rag",
        "use_case_query_type": False,
        "warmup_rounds": 1,
        "reset_cache_before_run": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": True,
            "route_reason": "eval_hyde_kg_context_expansion_cached",
            "agentic_features": _agentic_features(context_expansion=True),
            "cache_enabled": True,
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": False,
                "run_hyde": True,
                "run_web": False,
                "bm25_weight_multiplier": 0.0,
            },
        },
    },
    "router_hybrid_grounded_cached": {
        "description": "在 HyDE + KG + Context Expansion + Cache + CRAG Retry 的基础上，引入 BM25 混合检索、Router 控制 HyDE/CRAG 深检索，并启用严格 grounded 回答约束",
        "technique": "Router Hybrid Grounded Cached",
        "compare_to": "hyde_kg_context_expansion_cached",
        "use_case_query_type": False,
        "warmup_rounds": 1,
        "reset_cache_before_run": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": True,
            "router_deep_search_enabled": True,
            "grounded_answer_enabled": True,
            "route_reason": "eval_router_hybrid_grounded_cached",
            "agentic_features": _agentic_features(context_expansion=True),
            "cache_enabled": True,
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
    "router_anchor_contextual_grounded_cached": {
        "description": "Anchor + Contextual Retrieval + Router + Grounded Prompt 的评测版方案，针对 comparison/relation 的标题目标覆盖和证据包约束优化",
        "technique": "Router Anchor Contextual Grounded Cached",
        "compare_to": "router_hybrid_grounded_cached",
        "use_case_query_type": False,
        "warmup_rounds": 1,
        "reset_cache_before_run": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": False,
            "router_deep_search_enabled": True,
            "grounded_answer_enabled": True,
            "anchor_context_enabled": True,
            "route_reason": "eval_router_anchor_contextual_grounded_cached",
            "agentic_features": _agentic_features(context_expansion=True),
            "cache_enabled": True,
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_anchor": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
                "anchor_weight_multiplier": 1.6,
                "bm25_weight_multiplier": 1.0,
            },
        },
    },
    "router_anchor_rescue_structured_cached": {
        "description": "质量优先评测版：常驻 BM25 + Embedding + Anchor + KG + Context Expansion + Grounded，comparison/relation/constraint/explain 进入受控 rescue，并优化结构化回答与缓存键稳定性",
        "technique": "Router Anchor Rescue Structured Cached",
        "compare_to": "router_anchor_contextual_grounded_cached",
        "use_case_query_type": False,
        "warmup_rounds": 1,
        "reset_cache_before_run": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": False,
            "router_deep_search_enabled": True,
            "grounded_answer_enabled": True,
            "anchor_context_enabled": True,
            "route_reason": "eval_router_anchor_rescue_structured_cached",
            "rescue_min_coverage_score": 0.72,
            "rescue_force_target_coverage": True,
            "answer_cache_descriptor_version": "answer_v2",
            "structured_answer_high_risk_only": True,
            "agentic_features": _agentic_features(
                subquery_routing=True,
                context_expansion=True,
                evidence_coverage=True,
                retrieval_rescue=True,
                structured_answer=True,
                clarification_guard=False,
            ),
            "cache_enabled": True,
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_anchor": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
                "anchor_weight_multiplier": 1.8,
                "bm25_weight_multiplier": 1.1,
                "kg_weight_multiplier": 1.2,
            },
        },
    },
    "final_system": {
        "description": "在 Baseline 上统一开启 BM25 / HyDE / Neo4j 与 CRAG 重试，作为 Agentic 消融底盘",
        "technique": "Final Unified RAG",
        "compare_to": "baseline_rag",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": True,
            "route_reason": "eval_final_system",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
    "agentic_context_expansion": {
        "description": "在统一多路检索底盘上只开启命中上下文扩展，用于量化章节/相邻步骤补文效果",
        "technique": "Agentic Context Expansion",
        "compare_to": "final_system",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": True,
            "route_reason": "eval_agentic_context_expansion",
            "agentic_features": _agentic_features(context_expansion=True),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
    "agentic_rescue_system": {
        "description": "在统一多路检索底盘上开启子问题级路由、证据覆盖和检索补救，用于量化复杂问题的自适应检索收益",
        "technique": "Agentic Retrieval Rescue",
        "compare_to": "final_system",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": False,
            "route_reason": "eval_agentic_rescue_system",
            "agentic_features": _agentic_features(
                subquery_routing=True,
                evidence_coverage=True,
                retrieval_rescue=True,
                clarification_guard=False,
            ),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
    "agentic_enhanced_system": {
        "description": "完整 Agentic 增强系统：在统一多路检索底盘上开启检索补救、上下文扩展和结构化回答",
        "technique": "Agentic Enhanced RAG",
        "compare_to": "final_system",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": False,
            "route_reason": "eval_agentic_enhanced_system",
            "agentic_features": _agentic_features(
                subquery_routing=True,
                context_expansion=True,
                evidence_coverage=True,
                retrieval_rescue=True,
                structured_answer=True,
                clarification_guard=False,
            ),
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
    "agentic_enhanced_system_cached": {
        "description": "完整 Agentic 增强系统 + 多级缓存（预热一轮后统计热缓存收益）",
        "technique": "Agentic Enhanced RAG + Cache",
        "compare_to": "agentic_enhanced_system",
        "use_case_query_type": True,
        "warmup_rounds": 1,
        "reset_cache_before_run": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_graph_preferred": False,
            "bm25_enabled": True,
            "retrieval_grader_enabled": True,
            "legacy_retry_enabled": False,
            "route_reason": "eval_agentic_enhanced_system_cached",
            "agentic_features": _agentic_features(
                subquery_routing=True,
                context_expansion=True,
                evidence_coverage=True,
                retrieval_rescue=True,
                structured_answer=True,
                clarification_guard=False,
            ),
            "cache_enabled": True,
            "retrieval_plan_overrides": {
                "graph_first": False,
                "run_kg": True,
                "run_embedding": True,
                "run_bm25": True,
                "run_hyde": True,
                "run_web": False,
            },
        },
    },
}


def get_variant_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for name, config in VARIANTS.items():
        if config.get("is_feature_variant"):
            continue
        catalog.append(
            {
                "name": name,
                "description": config.get("description") or name,
                "technique": config.get("technique") or name,
                "compare_to": config.get("compare_to"),
                "is_default": name in DEFAULT_VARIANTS,
            }
        )
    return catalog


def get_variant_definition(variant_name: str) -> Dict[str, Any]:
    config = VARIANTS.get(str(variant_name or "").strip())
    if not config:
        raise ValueError(f"未知评测变体: {variant_name}")
    return copy.deepcopy(config)


def build_variant_runtime_overrides(variant_name: str) -> Dict[str, Any]:
    config = get_variant_definition(variant_name)
    overrides = copy.deepcopy(config.get("evaluation_overrides") or {})
    overrides.setdefault("cache_enabled", False)
    if config.get("use_case_query_type"):
        overrides.pop("force_query_type", None)
        overrides.pop("force_graph_preferred", None)
    return overrides


def build_variant_runtime_state(
    query: str,
    variant_name: str,
    session_id: str,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "original_query": str(query or "").strip(),
        "is_stream": False,
        "evaluation_streaming_llm": True,
        "suppress_sse": True,
        "evaluation_mode": True,
        "evaluation_variant_name": str(variant_name or "").strip(),
        "evaluation_overrides": build_variant_runtime_overrides(variant_name),
    }


RAGAS_METRICS = [
    {
        "name": "llm_context_recall",
        "factory": LLMContextRecall,
        "aliases": ["llm_context_recall", "context_recall"],
        "requires": lambda row: bool(
            row.get("reference") and row.get("retrieved_contexts")
        ),
    },
    {
        "name": "faithfulness",
        "factory": Faithfulness,
        "aliases": ["faithfulness"],
        "requires": lambda row: bool(
            row.get("response") and row.get("retrieved_contexts")
        ),
    },
    {
        "name": "factual_correctness",
        "factory": FactualCorrectness,
        "aliases": ["factual_correctness"],
        "requires": lambda row: bool(row.get("response") and row.get("reference")),
    },
    {
        "name": "response_relevancy",
        "factory": ResponseRelevancy,
        "aliases": ["response_relevancy", "answer_relevancy"],
        "requires": lambda row: bool(row.get("user_input") and row.get("response")),
    },
    {
        "name": "id_based_context_precision",
        "factory": IDBasedContextPrecision,
        "aliases": ["id_based_context_precision"],
        "requires": lambda row: bool(
            row.get("retrieved_context_ids") and row.get("reference_context_ids")
        ),
    },
    {
        "name": "id_based_context_recall",
        "factory": IDBasedContextRecall,
        "aliases": ["id_based_context_recall"],
        "requires": lambda row: bool(
            row.get("retrieved_context_ids") and row.get("reference_context_ids")
        ),
    },
]


class BgeM3LangChainEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return generate_embeddings([str(text or "") for text in texts])["dense"]

    def embed_query(self, text: str) -> List[float]:
        return generate_embeddings([str(text or "")])["dense"][0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def _query(case: Dict[str, Any]) -> str:
    return str(case.get("query") or case.get("rewritten_query") or "").strip()


def _reference(case: Dict[str, Any]) -> str:
    return str(
        case.get("reference_answer")
        or case.get("reference")
        or case.get("ground_truth")
        or ""
    ).strip()


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _query_type(case: Dict[str, Any]) -> str:
    return str(case.get("query_type") or "general").strip() or "general"


def _item_names(case: Dict[str, Any]) -> List[str]:
    return [
        str(name).strip()
        for name in (case.get("item_names") or [])
        if str(name).strip()
    ]


def _relevant_ids(case: Dict[str, Any]) -> List[str]:
    return normalize_ids(
        case.get("relevant_chunk_ids")
        or case.get("reference_context_ids")
        or case.get("relevant_ids")
        or case.get("chunk_ids")
        or []
    )


def _resolved_relevant_ids(case: Dict[str, Any]) -> List[str]:
    ground_truth = case.get("_evaluation_ground_truth") or {}
    return normalize_ids(
        ground_truth.get("resolved_ids")
        or case.get("resolved_relevant_chunk_ids")
        or _relevant_ids(case)
    )


def _load_dataset_payload(dataset_path: str) -> Any:
    path = Path(dataset_path)
    if path.suffix.lower() == ".jsonl":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ground_truth_doc_text(doc: Dict[str, Any]) -> str:
    parts = [
        str(doc.get("item_name") or "").strip(),
        str(doc.get("title") or "").strip(),
        str(doc.get("parent_title") or "").strip(),
        str(doc.get("part") or "").strip(),
        str(doc.get("file_title") or "").strip(),
        str(doc.get("content") or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def _unique_text_values(values: Sequence[Any]) -> List[str]:
    unique: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _item_names_from_docs(docs: Sequence[Dict[str, Any]]) -> List[str]:
    return _unique_text_values(doc.get("item_name") for doc in docs or [])


def _token_overlap_ratio(reference: str, candidate: str) -> float:
    reference_tokens = set(tokenize_text(reference))
    if not reference_tokens:
        return 0.0
    candidate_tokens = set(tokenize_text(candidate))
    if not candidate_tokens:
        return 0.0
    return round(len(reference_tokens & candidate_tokens) / len(reference_tokens), 4)


def _chunks_collection_name() -> str:
    return os.environ.get("CHUNKS_COLLECTION") or ""


def _fetch_all_ground_truth_candidates(
    client: Any,
    collection_name: str,
) -> List[Dict[str, Any]]:
    return [
        normalize_entity_for_business_id(doc)
        for doc in query_chunks_by_filter(
            client=client,
            collection_name=collection_name,
            filter_expr="",
            output_fields=list(GROUND_TRUTH_OUTPUT_FIELDS),
            limit=GROUND_TRUTH_SEARCH_LIMIT,
        )
    ]


def _fetch_ground_truth_candidates(
    case: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], bool]:
    client = get_milvus_client()
    collection_name = _chunks_collection_name()
    if not client or not collection_name:
        return [], False

    item_filter = build_item_name_filter_expr(_item_names(case))
    if item_filter:
        docs = query_chunks_by_filter(
            client=client,
            collection_name=collection_name,
            filter_expr=item_filter,
            output_fields=list(GROUND_TRUTH_OUTPUT_FIELDS),
            limit=GROUND_TRUTH_SEARCH_LIMIT,
        )
        docs = [normalize_entity_for_business_id(doc) for doc in docs]
        if docs:
            return docs, False

        # The evaluation set may carry historical short aliases while the current
        # corpus stores the full imported item name. Fall back to the full corpus
        # so sync_evaluation_dataset can repair both chunk ids and item_names.
        return _fetch_all_ground_truth_candidates(client, collection_name), True

    return _fetch_all_ground_truth_candidates(client, collection_name), False


def _resolve_retrieval_ground_truth(case: Dict[str, Any]) -> Dict[str, Any]:
    declared_ids = _relevant_ids(case)
    reference_answer = _reference(case)
    item_names = _item_names(case)
    eligible = bool(declared_ids or reference_answer)
    result: Dict[str, Any] = {
        "eligible": eligible,
        "declared_ids": declared_ids,
        "resolved_ids": [],
        "source": "",
        "reason": "",
        "candidate_count": 0,
        "item_names": item_names,
        "declared_ids_stale": False,
        "item_name_filter_miss": False,
        "resolved_item_names": [],
    }
    if not eligible:
        result["reason"] = "no_retrieval_ground_truth"
        return result

    candidates, item_name_filter_miss = _fetch_ground_truth_candidates(case)
    result["candidate_count"] = len(candidates)
    result["item_name_filter_miss"] = item_name_filter_miss
    candidate_ids = {
        str(extract_chunk_id(doc))
        for doc in candidates
        if extract_chunk_id(doc) is not None
    }
    candidates_by_id = {
        str(extract_chunk_id(doc)): doc
        for doc in candidates
        if extract_chunk_id(doc) is not None
    }

    if declared_ids:
        matched_ids = [
            chunk_id for chunk_id in declared_ids if chunk_id in candidate_ids
        ]
        if matched_ids:
            result["resolved_ids"] = matched_ids
            result["source"] = "declared_ids"
            result["resolved_item_names"] = _item_names_from_docs(
                [candidates_by_id[chunk_id] for chunk_id in matched_ids if chunk_id in candidates_by_id]
            )
            if item_name_filter_miss:
                result["reason"] = "item_name_not_in_corpus"
            return result
        result["declared_ids_stale"] = True

    if item_names and not candidates:
        result["reason"] = "item_name_not_in_corpus"
        result["source"] = "unresolved"
        return result

    if not candidates:
        result["reason"] = "no_candidate_chunks"
        result["source"] = "unresolved"
        return result

    rank_query = "\n".join(part for part in [reference_answer, _query(case)] if part)
    ranked_docs = rank_documents_bm25(
        query_text=rank_query,
        documents=candidates,
        text_getter=_ground_truth_doc_text,
        top_k=5,
    )
    if not ranked_docs:
        result["reason"] = "no_reference_match"
        result["source"] = "unresolved"
        return result

    top_score = float(ranked_docs[0][1] or 0.0)
    selected_ids: List[str] = []
    selected_docs: List[Dict[str, Any]] = []
    for doc, score in ranked_docs:
        if score <= 0:
            continue
        doc_text = _ground_truth_doc_text(doc)
        overlap = _token_overlap_ratio(reference_answer or _query(case), doc_text)
        chunk_id = extract_chunk_id(doc)
        if chunk_id is None:
            continue
        if overlap >= 0.2 or (score >= top_score * 0.92 and overlap >= 0.08):
            selected_ids.append(str(chunk_id))
            selected_docs.append(doc)
        if len(selected_ids) >= 3:
            break

    if not selected_ids:
        result["reason"] = "reference_match_too_weak"
        result["source"] = "unresolved"
        return result

    result["resolved_ids"] = selected_ids
    result["source"] = "reference_answer_bm25"
    result["resolved_item_names"] = _item_names_from_docs(selected_docs)
    if result["declared_ids_stale"]:
        result["reason"] = "declared_ids_stale"
    if item_name_filter_miss:
        result["reason"] = "item_name_not_in_corpus"
    return result


def _prepare_cases(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared_cases: List[Dict[str, Any]] = []
    for case in cases:
        prepared_case = copy.deepcopy(case)
        prepared_case["_evaluation_ground_truth"] = _resolve_retrieval_ground_truth(
            prepared_case
        )
        prepared_cases.append(prepared_case)
    return prepared_cases


def _ground_truth_records_from_cases(
    cases: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        (
            item.get("_evaluation_ground_truth")
            or item.get("retrieval_ground_truth")
            or {}
        )
        for item in cases
        if (
            (
                item.get("_evaluation_ground_truth")
                or item.get("retrieval_ground_truth")
                or {}
            ).get("eligible")
        )
    ]


def _build_ground_truth_alignment(
    ground_truth_records: Sequence[Dict[str, Any]],
    resolved_cases: int,
) -> tuple[Dict[str, Any], List[str]]:
    source_breakdown = Counter(
        str(record.get("source") or "unresolved") for record in ground_truth_records
    )
    unresolved_reasons = Counter(
        str(record.get("reason") or "unknown")
        for record in ground_truth_records
        if not (record.get("resolved_ids") or [])
    )
    stale_declared_cases = sum(
        1 for record in ground_truth_records if record.get("declared_ids_stale")
    )
    item_name_filter_miss_cases = sum(
        1 for record in ground_truth_records if record.get("item_name_filter_miss")
    )
    item_name_filter_miss_resolved_cases = sum(
        1
        for record in ground_truth_records
        if record.get("item_name_filter_miss") and (record.get("resolved_ids") or [])
    )
    warnings: List[str] = []
    if ground_truth_records and resolved_cases < len(ground_truth_records):
        warnings.append(
            f"检索金标仅对齐了 {resolved_cases}/{len(ground_truth_records)} 个带金标样本；未对齐样本已从 Recall/MRR 统计中剔除。"
        )
    if unresolved_reasons.get("item_name_not_in_corpus"):
        warnings.append(
            f"{unresolved_reasons['item_name_not_in_corpus']} 个样本的 item_names 在当前知识库中不存在，说明评测集与当前入库内容未完全对齐。"
        )
    elif item_name_filter_miss_cases:
        warnings.append(
            f"{item_name_filter_miss_cases} 个样本的 item_names 与当前知识库名称不一致，系统已按 reference_answer 兜底重映射；正式评测前建议同步评测数据。"
        )
    if stale_declared_cases:
        warnings.append(
            f"{stale_declared_cases} 个样本的 declared chunk_id 已失效，系统已按 reference_answer 自动重映射当前 chunk_id。"
        )

    summary = {
        "eligible_cases": len(ground_truth_records),
        "resolved_cases": resolved_cases,
        "unresolved_cases": max(len(ground_truth_records) - resolved_cases, 0),
        "source_breakdown": dict(sorted(source_breakdown.items())),
        "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        "stale_declared_cases": stale_declared_cases,
        "item_name_filter_miss_cases": item_name_filter_miss_cases,
        "item_name_filter_miss_resolved_cases": item_name_filter_miss_resolved_cases,
    }
    return summary, warnings


def _load_dataset_payload(
    dataset_path: str,
) -> tuple[Path, str, Any, List[Dict[str, Any]]]:
    resolved_path = Path(dataset_path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"评测数据集不存在: {resolved_path}")

    if resolved_path.suffix.lower() == ".jsonl":
        cases: List[Dict[str, Any]] = []
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return resolved_path, "jsonl", cases, cases

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return resolved_path, "json", payload, payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return resolved_path, "json", payload, payload["cases"]
    raise ValueError("dataset 必须是 JSON 数组、带 cases 字段的 JSON 对象或 JSONL")


def _write_dataset_payload(
    output_path: Path,
    dataset_format: str,
    payload: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_format == "jsonl":
        text = "\n".join(
            json.dumps(item, ensure_ascii=False) for item in (payload or [])
        )
        if text:
            text += "\n"
        output_path.write_text(text, encoding="utf-8")
        return
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sync_evaluation_dataset(
    dataset_path: str,
    output_path: str | None = None,
    create_backup: bool = True,
) -> Dict[str, Any]:
    resolved_dataset_path, dataset_format, payload, cases = _load_dataset_payload(
        dataset_path
    )
    original_text = resolved_dataset_path.read_text(encoding="utf-8")
    prepared_before = _prepare_cases(cases)
    before_records = _ground_truth_records_from_cases(prepared_before)
    before_resolved_cases = sum(
        1 for record in before_records if record.get("resolved_ids")
    )
    before_summary, before_warnings = _build_ground_truth_alignment(
        before_records, before_resolved_cases
    )

    updated_cases = 0
    already_aligned_cases = 0
    updated_item_name_cases = 0
    unresolved_case_ids: List[str] = []
    for raw_case, prepared_case in zip(cases, prepared_before):
        record = prepared_case.get("_evaluation_ground_truth") or {}
        if not record.get("eligible"):
            continue
        resolved_ids = normalize_ids(record.get("resolved_ids") or [])
        case_id = str(
            raw_case.get("case_id")
            or raw_case.get("id")
            or prepared_case.get("case_id")
            or prepared_case.get("id")
            or f"case_{len(unresolved_case_ids) + 1}"
        )
        if not resolved_ids:
            unresolved_case_ids.append(case_id)
            continue

        declared_ids = _relevant_ids(raw_case)
        resolved_item_names = _unique_text_values(record.get("resolved_item_names") or [])
        current_item_names = _item_names(raw_case)
        item_names_changed = bool(
            resolved_item_names and resolved_item_names != current_item_names
        )
        if (
            declared_ids == resolved_ids
            and not record.get("declared_ids_stale")
            and not item_names_changed
        ):
            already_aligned_cases += 1
        else:
            updated_cases += 1

        raw_case["relevant_chunk_ids"] = list(resolved_ids)
        raw_case["resolved_relevant_chunk_ids"] = list(resolved_ids)
        if item_names_changed:
            raw_case["item_names"] = list(resolved_item_names)
            updated_item_name_cases += 1
        if "reference_context_ids" in raw_case:
            raw_case["reference_context_ids"] = list(resolved_ids)
        if "relevant_ids" in raw_case:
            raw_case["relevant_ids"] = list(resolved_ids)
        if "chunk_ids" in raw_case:
            raw_case["chunk_ids"] = list(resolved_ids)

    resolved_output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_dataset_path
    )
    backup_path = None
    should_write = resolved_output_path != resolved_dataset_path or updated_cases > 0
    if should_write and create_backup and resolved_output_path == resolved_dataset_path:
        backup_path = resolved_dataset_path.with_name(
            f"{resolved_dataset_path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{resolved_dataset_path.suffix}"
        )
        backup_path.write_text(original_text, encoding="utf-8")
    if should_write:
        _write_dataset_payload(resolved_output_path, dataset_format, payload)

    _, _, _, reloaded_cases = _load_dataset_payload(str(resolved_output_path))
    prepared_after = _prepare_cases(reloaded_cases)
    after_records = _ground_truth_records_from_cases(prepared_after)
    after_resolved_cases = sum(
        1 for record in after_records if record.get("resolved_ids")
    )
    after_summary, after_warnings = _build_ground_truth_alignment(
        after_records, after_resolved_cases
    )

    message_parts = []
    if updated_cases:
        message_parts.append(f"已同步 {updated_cases} 个样本的检索金标到当前知识库。")
    if updated_item_name_cases:
        message_parts.append(f"其中 {updated_item_name_cases} 个样本更新了 item_names。")
    message = (
        " ".join(message_parts)
        if message_parts
        else "当前评测集的检索金标已与现有知识库对齐，无需更新。"
    )
    if unresolved_case_ids:
        message += f" 仍有 {len(unresolved_case_ids)} 个样本未能自动对齐。"

    return {
        "dataset_path": str(resolved_dataset_path),
        "output_path": str(resolved_output_path),
        "backup_path": str(backup_path) if backup_path else "",
        "case_count": len(cases),
        "updated_cases": updated_cases,
        "updated_item_name_cases": updated_item_name_cases,
        "already_aligned_cases": already_aligned_cases,
        "unresolved_cases": len(unresolved_case_ids),
        "unresolved_case_ids": unresolved_case_ids,
        "stale_declared_cases_before": before_summary.get("stale_declared_cases", 0),
        "ground_truth_summary": after_summary,
        "warnings": after_warnings,
        "before_ground_truth_summary": before_summary,
        "before_warnings": before_warnings,
        "message": message,
    }


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        casted = float(value)
        if not math.isfinite(casted):
            return None
        return casted
    except (TypeError, ValueError):
        return None


def _avg(values: Sequence[Any]) -> Optional[float]:
    nums = [_num(value) for value in values]
    valid = [value for value in nums if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _pct(values: Sequence[Any], ratio: float) -> Optional[float]:
    nums = sorted(
        value for value in (_num(item) for item in values) if value is not None
    )
    if not nums:
        return None
    clamped = min(max(float(ratio), 0.0), 1.0)
    index = max(0, min(math.ceil(len(nums) * clamped) - 1, len(nums) - 1))
    return round(nums[index], 4)


def _default_output_path() -> Path:
    output_dir = Path(PROJECT_ROOT) / "reports" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / f"unified_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


def _resolve_variants(names: Sequence[str]) -> List[str]:
    requested = [name.strip() for name in names if name.strip()]
    if not requested or requested == ["all"]:
        return list(DEFAULT_VARIANTS)
    unknown = [name for name in requested if name not in VARIANTS]
    if unknown:
        raise ValueError(f"未知评测变体: {', '.join(unknown)}")
    return requested


def _build_overrides(variant_name: str, case: Dict[str, Any]) -> Dict[str, Any]:
    config = VARIANTS[variant_name]
    overrides = copy.deepcopy(config.get("evaluation_overrides") or {})
    overrides.setdefault("cache_enabled", False)
    if config.get("use_case_query_type"):
        query_type = _query_type(case)
        overrides["force_query_type"] = query_type
        overrides.setdefault("force_graph_preferred", False)
    item_names = _item_names(case)
    if item_names:
        overrides.setdefault("force_item_names", item_names)
    return overrides


def _variant_request_payload(variant_name: str) -> Dict[str, Any]:
    config = VARIANTS.get(variant_name) or {}
    feature_variant = config.get("feature_variant") or {}
    if config.get("is_feature_variant"):
        return {
            "variant_spec": {
                "label": feature_variant.get("label")
                or config.get("technique")
                or variant_name,
                "features": feature_variant.get("requested_features") or [],
            }
        }
    return {"variant_name": variant_name}


def _extract_contexts(docs: Sequence[Dict[str, Any]]) -> List[str]:
    contexts: List[str] = []
    for doc in docs or []:
        text = str(doc.get("text") or doc.get("content") or "").strip()
        if text:
            contexts.append(text)
    return contexts


def _extract_context_ids(docs: Sequence[Dict[str, Any]]) -> List[str]:
    chunk_ids: List[str] = []
    for doc in docs or []:
        chunk_id = extract_chunk_id(doc) or doc.get("doc_id")
        if chunk_id is None:
            continue
        value = str(chunk_id).strip()
        if value:
            chunk_ids.append(value)
    return chunk_ids


def _context_ids_from_summary(summary: Dict[str, Any] | None) -> List[str]:
    if not isinstance(summary, dict):
        return []
    return normalize_ids(summary.get("context_ids") or [])


def _hit_at_k(
    predicted_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if any(chunk_id in relevant for chunk_id in predicted_ids[:k]) else 0.0


def _precision_for_ids(predicted_ids: Sequence[str], relevant_ids: Sequence[str]) -> Optional[float]:
    predicted = [str(item).strip() for item in predicted_ids or [] if str(item).strip()]
    relevant = {str(item).strip() for item in relevant_ids or [] if str(item).strip()}
    if not relevant:
        return None
    if not predicted:
        return 0.0
    return len([chunk_id for chunk_id in predicted if chunk_id in relevant]) / len(predicted)


def _preview(text: str, limit: int = 220) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _stage_map(perf_doc: Dict[str, Any]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for item in perf_doc.get("stages") or []:
        stage_name = str(item.get("stage") or "").strip()
        duration = _num(item.get("duration_ms"))
        if stage_name and duration is not None:
            mapping[stage_name] = duration
    return mapping


def _run_single_case(
    case: Dict[str, Any],
    variant_name: str,
    cache_temperature: str = "",
) -> Dict[str, Any]:
    service_base_url = _evaluation_query_service_base_url()
    if service_base_url:
        return _run_single_case_via_service(
            case,
            variant_name,
            service_base_url,
            cache_temperature=cache_temperature,
        )

    case_id = str(case.get("case_id") or case.get("id") or uuid.uuid4().hex[:8])
    session_id = f"eval-{variant_name}-{case_id}-{uuid.uuid4().hex[:8]}"
    query = _query(case)
    declared_relevant_ids = _relevant_ids(case)
    resolved_relevant_ids = _resolved_relevant_ids(case)
    initial_state = {
        "session_id": session_id,
        "original_query": query,
        "is_stream": False,
        "evaluation_streaming_llm": True,
        "suppress_sse": True,
        "evaluation_mode": True,
        "evaluation_variant_name": variant_name,
        "evaluation_overrides": _build_overrides(variant_name, case),
    }
    final_state: Dict[str, Any] = {}
    error = ""
    cache_summary: Dict[str, Any] = {}
    with query_cache_request_context(initial_state):
        perf_start(session_id, query)
        try:
            result = query_app.invoke(initial_state)
            if isinstance(result, dict):
                final_state = result
        except Exception as exc:
            error = str(exc)
        cache_summary = get_current_request_cache_summary()
        if isinstance(final_state, dict):
            final_state["cache_summary"] = (
                final_state.get("cache_summary") or cache_summary
            )
        perf_doc = perf_finish(session_id, persist=False) or {}
    clear_task(session_id)
    reranked_docs = final_state.get("reranked_docs") or []
    retrieved_contexts = _extract_contexts(reranked_docs)
    final_context_summary = final_state.get("final_context_summary") or {}
    final_context_ids = normalize_ids(
        final_state.get("final_context_ids")
        or _context_ids_from_summary(final_context_summary)
    )
    final_context_titles = _string_list(
        final_state.get("final_context_titles")
        or final_context_summary.get("context_titles")
        or []
    )
    final_context_preview = _string_list(
        final_context_summary.get("context_previews") or []
    )
    answer_error = str(final_state.get("answer_error") or "").strip()
    item_name_extract_error = str(
        final_state.get("item_name_extract_error") or ""
    ).strip()
    runtime_error = " | ".join(
        value
        for value in [error.strip(), answer_error, item_name_extract_error]
        if value
    )
    return {
        "case_id": case_id,
        "query": query,
        "query_type": _query_type(case),
        "item_names": _item_names(case),
        "answerable": bool(case.get("answerable", True)),
        "required_facts": _string_list(case.get("required_facts")),
        "forbidden_facts": _string_list(case.get("forbidden_facts")),
        "evaluation_tags": _string_list(case.get("evaluation_tags")),
        "reference_answer": _reference(case),
        "declared_relevant_chunk_ids": declared_relevant_ids,
        "relevant_chunk_ids": resolved_relevant_ids,
        "retrieval_ground_truth": copy.deepcopy(
            case.get("_evaluation_ground_truth") or {}
        ),
        "response": str(final_state.get("answer") or "").strip(),
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": _extract_context_ids(reranked_docs),
        "retrieved_context_preview": [
            _preview(text) for text in retrieved_contexts[:3]
        ],
        "retrieved_context_titles": [
            str(doc.get("title") or "") for doc in reranked_docs[:5]
        ],
        "retrieval_plan": final_state.get("retrieval_plan") or {},
        "runtime_query_type": str(final_state.get("query_type") or _query_type(case)),
        "query_complexity": str(final_state.get("query_complexity") or "simple"),
        "query_complexity_reason": str(
            final_state.get("query_complexity_reason") or ""
        ),
        "router_decision": str(final_state.get("router_decision") or "default_path"),
        "router_query_family": str(final_state.get("router_query_family") or "general"),
        "query_anchor_targets": final_state.get("query_anchor_targets") or [],
        "anchor_hits": final_state.get("anchor_hits") or [],
        "target_coverage": final_state.get("target_coverage") or {},
        "evidence_pack_summary": final_state.get("evidence_pack_summary") or {},
        "context_budget_chars": int(final_state.get("context_budget_chars") or 0),
        "router_deep_search_enabled": bool(
            final_state.get("router_deep_search_enabled", False)
        ),
        "crag_router_enabled": bool(final_state.get("crag_router_enabled", False)),
        "grounded_mode": bool(final_state.get("grounded_mode", False)),
        "sub_query_routes": final_state.get("sub_query_routes") or [],
        "sub_query_results": final_state.get("sub_query_results") or [],
        "context_expansion_summary": final_state.get("context_expansion_summary") or {},
        "final_context_summary": final_context_summary,
        "final_context_ids": final_context_ids,
        "final_context_titles": final_context_titles,
        "final_context_preview": final_context_preview,
        "final_context_chars": int(
            final_state.get("final_context_chars")
            or final_context_summary.get("used_chars")
            or 0
        ),
        "final_context_doc_count": int(
            final_state.get("final_context_doc_count")
            or final_context_summary.get("included_docs")
            or 0
        ),
        "evidence_coverage_summary": final_state.get("evidence_coverage_summary") or {},
        "rerank_diagnostics": final_state.get("rerank_diagnostics") or {},
        "rescue_plan": final_state.get("rescue_plan") or {},
        "answer_plan": final_state.get("answer_plan") or {},
        "clarification_reason": str(final_state.get("clarification_reason") or ""),
        "judge_skipped_reason": str(final_state.get("judge_skipped_reason") or ""),
        "retrieval_judge_skipped_reason": str(
            final_state.get("retrieval_judge_skipped_reason") or ""
        ),
        "hallucination_judge_skipped_reason": str(
            final_state.get("hallucination_judge_skipped_reason") or ""
        ),
        "retrieval_grader_strategy": str(
            final_state.get("retrieval_grader_strategy") or ""
        ),
        "hallucination_check_strategy": str(
            final_state.get("hallucination_check_strategy") or ""
        ),
        "answer_error": answer_error,
        "item_name_extract_error": item_name_extract_error,
        "agentic_features": (
            initial_state.get("evaluation_overrides", {}).get("agentic_features") or {}
        ),
        "retrieval_grade": str(final_state.get("retrieval_grade") or ""),
        "retry_count": int(final_state.get("retry_count") or 0),
        "hallucination_retry_count": int(
            final_state.get("hallucination_retry_count") or 0
        ),
        "hallucination_check_passed": bool(
            final_state.get("hallucination_check_passed", True)
        ),
        "need_rag": bool(final_state.get("need_rag", True)),
        "llm_requested_model": str(final_state.get("llm_requested_model") or ""),
        "llm_model_used": str(final_state.get("llm_model_used") or ""),
        "llm_base_url_used": str(final_state.get("llm_base_url_used") or ""),
        "llm_fallback_model": str(final_state.get("llm_fallback_model") or ""),
        "llm_fallback_used": bool(final_state.get("llm_fallback_used", False)),
        "llm_fallback_reason": str(final_state.get("llm_fallback_reason") or ""),
        "cache_summary": final_state.get("cache_summary") or cache_summary,
        "latency_ms": _num(perf_doc.get("total_duration_ms")),
        "first_token_ms": _num(perf_doc.get("first_token_ms")),
        "first_answer_ms": _num(perf_doc.get("first_answer_ms")),
        "stage_durations_ms": _stage_map(perf_doc),
        "cache_temperature": cache_temperature or "",
        "error": runtime_error,
    }


def _run_single_case_via_service(
    case: Dict[str, Any],
    variant_name: str,
    service_base_url: str,
    cache_temperature: str = "",
) -> Dict[str, Any]:
    case_id = str(case.get("case_id") or case.get("id") or uuid.uuid4().hex[:8])
    query = _query(case)
    declared_relevant_ids = _relevant_ids(case)
    resolved_relevant_ids = _resolved_relevant_ids(case)

    response = requests.post(
        f"{service_base_url}/evaluation/variants/test",
        json={
            "query": query,
            "streaming": True,
            **_variant_request_payload(variant_name),
        },
        timeout=_evaluation_query_service_timeout(),
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    metadata = payload.get("metadata") or {}
    runtime_fields = payload.get("runtime_state_excerpt") or {}
    retrieved_contexts = [
        str(text or "").strip()
        for text in (payload.get("retrieved_contexts") or payload.get("retrieved_context_preview") or [])
        if str(text or "").strip()
    ]
    answer_error = str(runtime_fields.get("answer_error") or "").strip()
    item_name_extract_error = str(runtime_fields.get("item_name_extract_error") or "").strip()
    runtime_error = " | ".join(
        value
        for value in [
            str(payload.get("error") or "").strip(),
            answer_error,
            item_name_extract_error,
        ]
        if value
    )
    final_context_summary = (
        metadata.get("final_context_summary")
        or runtime_fields.get("final_context_summary")
        or {}
    )
    final_context_ids = normalize_ids(
        payload.get("final_context_ids")
        or metadata.get("final_context_ids")
        or runtime_fields.get("final_context_ids")
        or _context_ids_from_summary(final_context_summary)
    )
    final_context_titles = _string_list(
        payload.get("final_context_titles")
        or metadata.get("final_context_titles")
        or runtime_fields.get("final_context_titles")
        or final_context_summary.get("context_titles")
        or []
    )
    final_context_preview = _string_list(
        final_context_summary.get("context_previews") or []
    )

    return {
        "case_id": case_id,
        "query": query,
        "query_type": _query_type(case),
        "item_names": _item_names(case),
        "answerable": bool(case.get("answerable", True)),
        "required_facts": _string_list(case.get("required_facts")),
        "forbidden_facts": _string_list(case.get("forbidden_facts")),
        "evaluation_tags": _string_list(case.get("evaluation_tags")),
        "reference_answer": _reference(case),
        "declared_relevant_chunk_ids": declared_relevant_ids,
        "relevant_chunk_ids": resolved_relevant_ids,
        "retrieval_ground_truth": copy.deepcopy(
            case.get("_evaluation_ground_truth") or {}
        ),
        "response": str(payload.get("answer") or "").strip(),
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": payload.get("retrieved_context_ids") or [],
        "retrieved_context_preview": payload.get("retrieved_context_preview") or [],
        "retrieved_context_titles": payload.get("retrieved_context_titles") or [],
        "retrieval_plan": metadata.get("retrieval_plan")
        or runtime_fields.get("retrieval_plan")
        or {},
        "runtime_query_type": str(
            runtime_fields.get("query_type") or metadata.get("query_type") or _query_type(case)
        ),
        "query_complexity": str(
            metadata.get("query_complexity") or runtime_fields.get("query_complexity") or "simple"
        ),
        "query_complexity_reason": str(
            metadata.get("query_complexity_reason")
            or runtime_fields.get("query_complexity_reason")
            or ""
        ),
        "router_decision": str(
            metadata.get("router_decision") or runtime_fields.get("router_decision") or "default_path"
        ),
        "router_query_family": str(
            metadata.get("router_query_family")
            or runtime_fields.get("router_query_family")
            or "general"
        ),
        "query_anchor_targets": metadata.get("query_anchor_targets")
        or runtime_fields.get("query_anchor_targets")
        or [],
        "anchor_hits": metadata.get("anchor_hits")
        or runtime_fields.get("anchor_hits")
        or [],
        "target_coverage": metadata.get("target_coverage")
        or runtime_fields.get("target_coverage")
        or {},
        "evidence_pack_summary": metadata.get("evidence_pack_summary")
        or runtime_fields.get("evidence_pack_summary")
        or {},
        "context_budget_chars": int(
            metadata.get("context_budget_chars")
            or runtime_fields.get("context_budget_chars")
            or 0
        ),
        "router_deep_search_enabled": bool(
            metadata.get("router_deep_search_enabled", runtime_fields.get("router_deep_search_enabled", False))
        ),
        "crag_router_enabled": bool(
            metadata.get("crag_router_enabled", runtime_fields.get("crag_router_enabled", False))
        ),
        "grounded_mode": bool(
            metadata.get("grounded_mode", runtime_fields.get("grounded_mode", False))
        ),
        "sub_query_routes": runtime_fields.get("sub_query_routes") or [],
        "sub_query_results": runtime_fields.get("sub_query_results") or [],
        "context_expansion_summary": metadata.get("context_expansion_summary")
        or runtime_fields.get("context_expansion_summary")
        or {},
        "final_context_summary": final_context_summary,
        "final_context_ids": final_context_ids,
        "final_context_titles": final_context_titles,
        "final_context_preview": final_context_preview,
        "final_context_chars": int(
            payload.get("final_context_chars")
            or metadata.get("final_context_chars")
            or runtime_fields.get("final_context_chars")
            or final_context_summary.get("used_chars")
            or 0
        ),
        "final_context_doc_count": int(
            payload.get("final_context_doc_count")
            or metadata.get("final_context_doc_count")
            or runtime_fields.get("final_context_doc_count")
            or final_context_summary.get("included_docs")
            or 0
        ),
        "evidence_coverage_summary": metadata.get("evidence_coverage_summary")
        or runtime_fields.get("evidence_coverage_summary")
        or {},
        "rerank_diagnostics": metadata.get("rerank_diagnostics")
        or runtime_fields.get("rerank_diagnostics")
        or {},
        "rescue_plan": metadata.get("rescue_plan") or runtime_fields.get("rescue_plan") or {},
        "answer_plan": metadata.get("answer_plan") or runtime_fields.get("answer_plan") or {},
        "clarification_reason": str(
            metadata.get("clarification_reason") or runtime_fields.get("clarification_reason") or ""
        ),
        "judge_skipped_reason": str(
            metadata.get("judge_skipped_reason")
            or runtime_fields.get("judge_skipped_reason")
            or ""
        ),
        "retrieval_judge_skipped_reason": str(
            metadata.get("retrieval_judge_skipped_reason")
            or runtime_fields.get("retrieval_judge_skipped_reason")
            or ""
        ),
        "hallucination_judge_skipped_reason": str(
            metadata.get("hallucination_judge_skipped_reason")
            or runtime_fields.get("hallucination_judge_skipped_reason")
            or ""
        ),
        "retrieval_grader_strategy": str(
            metadata.get("retrieval_grader_strategy")
            or runtime_fields.get("retrieval_grader_strategy")
            or ""
        ),
        "hallucination_check_strategy": str(
            metadata.get("hallucination_check_strategy")
            or runtime_fields.get("hallucination_check_strategy")
            or ""
        ),
        "answer_error": answer_error,
        "item_name_extract_error": item_name_extract_error,
        "agentic_features": metadata.get("agentic_features")
        or _build_overrides(variant_name, case).get("agentic_features")
        or {},
        "retrieval_grade": str(runtime_fields.get("retrieval_grade") or ""),
        "retry_count": int(runtime_fields.get("retry_count") or 0),
        "hallucination_retry_count": int(
            runtime_fields.get("hallucination_retry_count") or 0
        ),
        "hallucination_check_passed": bool(
            runtime_fields.get("hallucination_check_passed", True)
        ),
        "need_rag": bool(runtime_fields.get("need_rag", True)),
        "llm_requested_model": str(runtime_fields.get("llm_requested_model") or ""),
        "llm_model_used": str(runtime_fields.get("llm_model_used") or ""),
        "llm_base_url_used": str(runtime_fields.get("llm_base_url_used") or ""),
        "llm_fallback_model": str(runtime_fields.get("llm_fallback_model") or ""),
        "llm_fallback_used": bool(runtime_fields.get("llm_fallback_used", False)),
        "llm_fallback_reason": str(runtime_fields.get("llm_fallback_reason") or ""),
        "cache_summary": metadata.get("cache_summary")
        or runtime_fields.get("cache_summary")
        or {},
        "latency_ms": _num(payload.get("latency_ms")),
        "first_token_ms": _num(payload.get("first_token_ms")),
        "first_answer_ms": _num(payload.get("first_answer_ms")),
        "stage_durations_ms": payload.get("stage_durations_ms") or {},
        "cache_temperature": cache_temperature or "",
        "error": runtime_error,
    }


def _build_ragas_row(case_result: Dict[str, Any]) -> Dict[str, Any]:
    prompt_contexts = (
        case_result.get("final_context_preview")
        or (case_result.get("final_context_summary") or {}).get("context_previews")
        or case_result.get("retrieved_contexts")
        or []
    )
    return {
        "user_input": case_result.get("query") or "",
        "query_type": case_result.get("query_type") or "general",
        "item_names": _string_list(case_result.get("item_names")),
        "answerable": bool(case_result.get("answerable", True)),
        "response": case_result.get("response") or "",
        "retrieved_contexts": prompt_contexts,
        "reference": case_result.get("reference_answer") or None,
        "retrieved_context_ids": case_result.get("retrieved_context_ids") or [],
        "final_context_ids": case_result.get("final_context_ids")
        or _context_ids_from_summary(case_result.get("final_context_summary") or {}),
        "reference_context_ids": _resolved_relevant_ids(case_result),
    }


def _clamp_score(value: Any) -> Optional[float]:
    casted = _num(value)
    if casted is None:
        return None
    return round(max(0.0, min(1.0, casted)), 4)


def _normalize_judge_score(value: Any) -> Optional[float]:
    score = _clamp_score(value)
    if score is None:
        return None
    return min(QUALITY_JUDGE_SCORE_BUCKETS, key=lambda bucket: abs(bucket - score))


def _metric_tokens(text: str) -> List[str]:
    return [token for token in tokenize_text(str(text or "")) if token]


def _token_recall(reference: str, candidate: str) -> float:
    reference_tokens = set(_metric_tokens(reference))
    if not reference_tokens:
        return 0.0
    candidate_tokens = set(_metric_tokens(candidate))
    if not candidate_tokens:
        return 0.0
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def _token_precision(reference: str, candidate: str) -> float:
    candidate_tokens = set(_metric_tokens(candidate))
    if not candidate_tokens:
        return 0.0
    reference_tokens = set(_metric_tokens(reference))
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & candidate_tokens) / len(candidate_tokens)


def _token_f1(reference: str, candidate: str) -> float:
    precision = _token_precision(reference, candidate)
    recall = _token_recall(reference, candidate)
    if precision <= 0 or recall <= 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _join_contexts(
    contexts: Sequence[str], limit: int = QUALITY_JUDGE_CONTEXT_CHARS
) -> str:
    text = "\n\n".join(
        str(item or "").strip() for item in contexts if str(item or "").strip()
    )
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _quality_mode() -> str:
    mode = str(os.environ.get("EVAL_QUALITY_MODE") or "hybrid").strip().lower()
    if mode in {"llm", "judge", "llm_judge"}:
        return "llm"
    if mode in {"lexical", "heuristic", "local"}:
        return "lexical"
    if mode in {"off", "disabled", "none"}:
        return "disabled"
    enabled = str(os.environ.get("EVAL_LLM_JUDGE_ENABLED") or "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return "lexical"
    return "hybrid"


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _lexical_quality_scores(row: Dict[str, Any]) -> Dict[str, float]:
    query = str(row.get("user_input") or "")
    response = str(row.get("response") or "")
    reference = str(row.get("reference") or "")
    context_text = _join_contexts(row.get("retrieved_contexts") or [])
    reference_answer_basis = reference or query

    factual = _token_f1(reference_answer_basis, response) if response else 0.0
    faithfulness = _token_recall(response, context_text) if response else 0.0
    relevancy = max(
        _token_recall(query, response) if response else 0.0,
        _token_recall(query, reference) if reference else 0.0,
    )
    context_recall = _token_recall(reference_answer_basis, context_text)

    return {
        "factual_correctness": round(factual, 4),
        "faithfulness": round(faithfulness, 4),
        "response_relevancy": round(relevancy, 4),
        "llm_context_recall": round(context_recall, 4),
    }


def _llm_quality_scores(row: Dict[str, Any]) -> Dict[str, float]:
    prompt_payload = {
        "query_type": str(row.get("query_type") or "general")[:80],
        "item_names": _string_list(row.get("item_names"))[:5],
        "answerable": bool(row.get("answerable", True)),
        "question": str(row.get("user_input") or "")[:QUALITY_JUDGE_TEXT_CHARS],
        "reference_answer": str(row.get("reference") or "")[
            :QUALITY_JUDGE_TEXT_CHARS
        ],
        "model_answer": str(row.get("response") or "")[:QUALITY_JUDGE_TEXT_CHARS],
        "retrieved_contexts": _join_contexts(row.get("retrieved_contexts") or []),
    }
    prompt = (
        "你是企业产品手册 RAG 评测员。请严格按 rubric 给分，只使用以下固定档位："
        "0.0、0.2、0.4、0.6、0.8、1.0。必须返回严格 JSON，不要输出 Markdown。\n"
        "通用规则：\n"
        "- question 是用户问题，query_type 是问题类型，item_names 是目标产品。\n"
        "- answerable=false 表示当前资料无法给出确定答案；合理拒答或要求澄清可以是高相关，编造具体答案必须扣分。\n"
        "- 不要因为措辞相似就给高分；只回答了部分关键结论时不得给满分。\n"
        "- 参考答案和检索上下文冲突时，faithfulness 只看检索上下文支撑，factual_correctness 只看参考答案一致性。\n"
        "指标 rubric：\n"
        "1. factual_correctness：模型答案与 reference_answer 的事实一致程度。"
        "1.0=完整且无事实错误；0.8=基本正确但遗漏次要条件；0.6=只覆盖部分关键事实；"
        "0.4=有明显遗漏或轻度错误；0.2=主要结论错误但有少量相关内容；0.0=空答案、错误拒答或核心事实相反。\n"
        "2. faithfulness：模型答案中的事实是否能被 retrieved_contexts 支撑。"
        "1.0=所有实质事实均有上下文依据；0.8=少量非关键表述无依据；0.6=部分关键事实有依据；"
        "0.4=依据不足且夹杂推测；0.2=大部分具体事实无依据；0.0=空答案或明显编造。\n"
        "3. response_relevancy：模型答案是否直接解决 question。"
        "1.0=直接且完整回答；0.8=回答方向正确但不够完整；0.6=部分相关；"
        "0.4=泛泛而谈或偏题较多；0.2=只有少量关键词相关；0.0=未回答问题。"
        "answerable=false 时，合理拒答/澄清可给 0.8-1.0。\n"
        "4. llm_context_recall：retrieved_contexts 是否覆盖 reference_answer 所需信息。"
        "1.0=覆盖全部关键证据；0.8=覆盖主要证据但缺少次要条件；0.6=覆盖部分关键证据；"
        "0.4=只覆盖少量背景；0.2=几乎没有可用证据；0.0=无上下文或完全不相关。\n"
        '返回格式：{"factual_correctness":0.0,"faithfulness":0.0,'
        '"response_relevancy":0.0,"llm_context_recall":0.0,'
        '"diagnostics":{"factual":"短句","faithfulness":"短句","relevancy":"短句","context_recall":"短句"}}\n'
        f"输入：{json.dumps(prompt_payload, ensure_ascii=False)}"
    )
    judge_model = str(os.environ.get("EVAL_JUDGE_MODEL") or "").strip() or None
    llm = get_llm_client(model=judge_model, json_mode=True)
    response = llm.invoke(prompt)
    parsed = _extract_json_object(
        coerce_llm_content(getattr(response, "content", response))
    )
    scores: Dict[str, float] = {}
    for key in QUALITY_METRIC_KEYS:
        score = _normalize_judge_score(parsed.get(key))
        if score is None:
            raise ValueError(f"LLM 评估返回缺失或非法字段：{key}")
        scores[key] = score
    return scores


def _id_context_scores(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    retrieved_ids = normalize_ids(row.get("retrieved_context_ids") or [])
    reference_ids = normalize_ids(row.get("reference_context_ids") or [])
    if not reference_ids:
        return {
            "id_based_context_precision": None,
            "id_based_context_recall": None,
        }
    relevant = set(reference_ids)
    hit_count = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant)
    precision = hit_count / len(retrieved_ids) if retrieved_ids else 0.0
    recall = hit_count / len(relevant)
    return {
        "id_based_context_precision": round(precision, 4),
        "id_based_context_recall": round(recall, 4),
    }


def _score_quality_row(
    row: Dict[str, Any], mode: str
) -> tuple[Dict[str, Optional[float]], str, str]:
    scores: Dict[str, Optional[float]] = {}
    scores.update(_id_context_scores(row))
    if mode == "disabled":
        for key in QUALITY_METRIC_KEYS:
            scores[key] = None
        return scores, "disabled", ""

    if mode in {"hybrid", "llm"}:
        try:
            scores.update(_llm_quality_scores(row))
            return scores, "llm_judge", ""
        except Exception as exc:
            if mode == "llm":
                for key in QUALITY_METRIC_KEYS:
                    scores[key] = None
                return scores, "llm_judge_failed", str(exc)
            scores.update(_lexical_quality_scores(row))
            return scores, "lexical_fallback", str(exc)

    scores.update(_lexical_quality_scores(row))
    return scores, "lexical", ""


def _run_ragas(
    case_results: List[Dict[str, Any]],
    cancel_callback: CancelCallback = None,
) -> Dict[str, Any]:
    rows = [_build_ragas_row(case_result) for case_result in case_results]
    per_case_scores: List[Dict[str, Optional[float]]] = [dict() for _ in case_results]
    summary: Dict[str, Any] = {}
    coverage: Dict[str, int] = {}
    errors: Dict[str, str] = {}
    mode = _quality_mode()
    method_counts: Counter[str] = Counter()
    llm_errors: Counter[str] = Counter()

    for index, row in enumerate(rows):
        _raise_if_cancelled(cancel_callback)
        scores, method, error = _score_quality_row(row, mode)
        per_case_scores[index] = scores
        method_counts[method] += 1
        if error:
            llm_errors[_preview(error, 220)] += 1

    for metric_key in [*QUALITY_METRIC_KEYS, *ID_CONTEXT_METRIC_KEYS]:
        values = [
            per_case_score.get(metric_key)
            for per_case_score in per_case_scores
            if per_case_score.get(metric_key) is not None
        ]
        coverage[metric_key] = len(values)
        summary_value = _avg(values)
        if summary_value is not None:
            summary[metric_key] = summary_value

    if llm_errors:
        errors["quality_judge"] = (
            f"LLM 评估失败 {sum(llm_errors.values())} 次，已使用词项覆盖评分兜底；"
            f"最常见错误：{llm_errors.most_common(1)[0][0]}"
        )
    return {
        "summary": summary,
        "coverage": coverage,
        "errors": errors,
        "per_case_scores": per_case_scores,
        "metadata": {
            "quality_mode": mode,
            "judge_prompt_version": QUALITY_JUDGE_PROMPT_VERSION,
            "score_buckets": list(QUALITY_JUDGE_SCORE_BUCKETS),
            "method_counts": dict(sorted(method_counts.items())),
            "llm_error_count": sum(llm_errors.values()),
        },
    }


def _merge_ragas_scores(
    case_results: List[Dict[str, Any]],
    per_case_scores: List[Dict[str, Optional[float]]],
) -> None:
    for case_result, scores in zip(case_results, per_case_scores):
        case_result["ragas_scores"] = scores


def _summarize_retrieval(
    case_results: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, int], Dict[str, Any], List[str]]:
    summary: Dict[str, Any] = {
        "empty_retrieval_rate": _avg(
            [0.0 if item.get("retrieved_context_ids") else 1.0 for item in case_results]
        )
        or 0.0,
    }
    eligible_cases = [item for item in case_results if _resolved_relevant_ids(item)]
    coverage: Dict[str, int] = {}
    retrieved_context_precision_values = [
        _precision_for_ids(
            item.get("retrieved_context_ids") or [],
            _resolved_relevant_ids(item),
        )
        for item in eligible_cases
    ]
    prompt_context_precision_values = []
    for item in eligible_cases:
        final_ids = normalize_ids(
            item.get("final_context_ids")
            or _context_ids_from_summary(item.get("final_context_summary") or {})
        )
        if not final_ids:
            prompt_context_precision_values.append(None)
        else:
            prompt_context_precision_values.append(
                _precision_for_ids(final_ids, _resolved_relevant_ids(item))
            )

    retrieved_context_precision = _avg(retrieved_context_precision_values)
    prompt_context_precision = _avg(prompt_context_precision_values)
    summary["retrieved_context_precision"] = (
        retrieved_context_precision if retrieved_context_precision is not None else 0.0
    )
    summary["prompt_context_precision"] = prompt_context_precision
    summary["final_context_precision"] = prompt_context_precision
    coverage["retrieved_context_precision"] = len(
        [value for value in retrieved_context_precision_values if value is not None]
    )
    coverage["prompt_context_precision"] = len(
        [value for value in prompt_context_precision_values if value is not None]
    )
    coverage["final_context_precision"] = len(
        [value for value in prompt_context_precision_values if value is not None]
    )
    for k in DEFAULT_K_VALUES:
        summary[f"hit@{k}"] = _avg(
            [
                _hit_at_k(
                    item.get("retrieved_context_ids") or [],
                    _resolved_relevant_ids(item),
                    k,
                )
                for item in eligible_cases
            ]
        )
        summary[f"recall@{k}"] = _avg(
            [
                recall_at_k(
                    item.get("retrieved_context_ids") or [],
                    _resolved_relevant_ids(item),
                    k,
                )
                for item in eligible_cases
            ]
        )
        summary[f"mrr@{k}"] = _avg(
            [
                mrr_at_k(
                    item.get("retrieved_context_ids") or [],
                    _resolved_relevant_ids(item),
                    k,
                )
                for item in eligible_cases
            ]
        )
        coverage[f"hit@{k}"] = len(eligible_cases)
        coverage[f"recall@{k}"] = len(eligible_cases)
        coverage[f"mrr@{k}"] = len(eligible_cases)

    ground_truth_records = _ground_truth_records_from_cases(case_results)
    ground_truth_summary, warnings = _build_ground_truth_alignment(
        ground_truth_records, len(eligible_cases)
    )
    return summary, coverage, ground_truth_summary, warnings


def _summarize_pipeline(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence_scores = [
        (item.get("evidence_coverage_summary") or {}).get("coverage_score")
        for item in case_results
    ]
    rerank_diagnostics = [
        item.get("rerank_diagnostics") or {} for item in case_results
    ]
    final_context_summaries = []
    for item in case_results:
        summary = dict(item.get("final_context_summary") or {})
        if "included_docs" not in summary and item.get("final_context_doc_count") is not None:
            summary["included_docs"] = item.get("final_context_doc_count")
        if "used_chars" not in summary and item.get("final_context_chars") is not None:
            summary["used_chars"] = item.get("final_context_chars")
        final_context_summaries.append(summary)
    cache_overall = [
        (item.get("cache_summary") or {}).get("overall") or {} for item in case_results
    ]
    answer_cache_hits = [
        (
            1.0
            if int(
                (
                    (
                        (item.get("cache_summary") or {})
                        .get("namespaces_breakdown", {})
                        .get("answer", {})
                    ).get("hits")
                    or 0
                )
            )
            > 0
            else 0.0
        )
        for item in case_results
    ]
    retrieval_cache_hits = [
        (
            1.0
            if any(
                int(
                    (
                        (
                            (item.get("cache_summary") or {})
                            .get("namespaces_breakdown", {})
                            .get(namespace, {})
                        ).get("hits")
                        or 0
                    )
                )
                > 0
                for namespace in (
                    "retrieval_embedding",
                    "retrieval_bm25",
                    "retrieval_kg",
                    "hyde_doc",
                    "web_search",
                    "rerank",
                    "embedding",
                )
            )
            else 0.0
        )
        for item in case_results
    ]
    return {
        "empty_answer_rate": _avg(
            [0.0 if item.get("response") else 1.0 for item in case_results]
        )
        or 0.0,
        "crag_retry_rate": _avg(
            [
                1.0 if int(item.get("retry_count") or 0) > 0 else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "hallucination_retry_rate": _avg(
            [
                1.0 if int(item.get("hallucination_retry_count") or 0) > 0 else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "need_rag_rate": _avg(
            [1.0 if item.get("need_rag") else 0.0 for item in case_results]
        )
        or 0.0,
        "clarification_rate": _avg(
            [
                1.0 if str(item.get("clarification_reason") or "").strip() else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "rescue_retry_rate": _avg(
            [
                1.0 if (item.get("rescue_plan") or {}).get("action") == "retry" else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "context_expansion_rate": _avg(
            [
                (
                    1.0
                    if int(
                        (
                            (item.get("context_expansion_summary") or {}).get(
                                "expanded_docs"
                            )
                            or 0
                        )
                    )
                    > 0
                    else 0.0
                )
                for item in case_results
            ]
        )
        or 0.0,
        "router_simple_rate": _avg(
            [
                1.0
                if str(item.get("query_complexity") or "simple") == "simple"
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "hyde_enabled_rate": _avg(
            [
                1.0
                if bool((item.get("retrieval_plan") or {}).get("run_hyde"))
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "anchor_enabled_rate": _avg(
            [
                1.0
                if bool((item.get("retrieval_plan") or {}).get("run_anchor"))
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "avg_target_coverage_rate": _avg(
            [
                _num((item.get("target_coverage") or {}).get("coverage_rate"))
                for item in case_results
                if (item.get("target_coverage") or {}).get("target_count")
            ]
        )
        or 0.0,
        "target_coverage_case_rate": _avg(
            [
                1.0
                if (item.get("target_coverage") or {}).get("target_count")
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "crag_router_enabled_rate": _avg(
            [
                1.0 if bool(item.get("crag_router_enabled", False)) else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "structured_answer_rate": _avg(
            [
                1.0 if (item.get("answer_plan") or {}).get("structured_output") else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "subquery_routing_rate": _avg(
            [
                1.0 if (item.get("sub_query_routes") or []) else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "retrieval_judge_skipped_rate": _avg(
            [
                1.0
                if str(item.get("retrieval_judge_skipped_reason") or "").strip()
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "hallucination_judge_skipped_rate": _avg(
            [
                1.0
                if str(item.get("hallucination_judge_skipped_reason") or "").strip()
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "rerank_fallback_rate": _avg(
            [1.0 if bool(item.get("fallback")) else 0.0 for item in rerank_diagnostics]
        )
        or 0.0,
        "rerank_heuristic_rate": _avg(
            [1.0 if bool(item.get("heuristic")) else 0.0 for item in rerank_diagnostics]
        )
        or 0.0,
        "rerank_cache_hit_rate": _avg(
            [1.0 if bool(item.get("cache_hit")) else 0.0 for item in rerank_diagnostics]
        )
        or 0.0,
        "avg_rerank_candidate_count": _avg(
            [item.get("candidate_count") for item in rerank_diagnostics]
        )
        or 0.0,
        "avg_rerank_selected_count": _avg(
            [item.get("selected_count") for item in rerank_diagnostics]
        )
        or 0.0,
        "avg_final_context_docs": _avg(
            [item.get("included_docs") for item in final_context_summaries]
        )
        or 0.0,
        "avg_final_context_used_chars": _avg(
            [item.get("used_chars") for item in final_context_summaries]
        )
        or 0.0,
        "avg_evidence_coverage_score": _avg(evidence_scores) or 0.0,
        "low_evidence_coverage_rate": _avg(
            [
                (
                    1.0
                    if (
                        _num(
                            (item.get("evidence_coverage_summary") or {}).get(
                                "coverage_score"
                            )
                        )
                        or 0.0
                    )
                    < 0.55
                    else 0.0
                )
                for item in case_results
            ]
        )
        or 0.0,
        "error_rate": _avg([1.0 if item.get("error") else 0.0 for item in case_results])
        or 0.0,
        "llm_fallback_rate": _avg(
            [1.0 if item.get("llm_fallback_used") else 0.0 for item in case_results]
        )
        or 0.0,
        "cache_hit_rate": _avg([bucket.get("hit_rate") for bucket in cache_overall])
        or 0.0,
        "l0_cache_hit_rate": _avg(
            [bucket.get("l0_hit_rate") for bucket in cache_overall]
        )
        or 0.0,
        "l1_cache_hit_rate": _avg(
            [bucket.get("l1_hit_rate") for bucket in cache_overall]
        )
        or 0.0,
        "l2_cache_hit_rate": _avg(
            [bucket.get("l2_hit_rate") for bucket in cache_overall]
        )
        or 0.0,
        "answer_cache_rate": _avg(answer_cache_hits) or 0.0,
        "retrieval_cache_rate": _avg(retrieval_cache_hits) or 0.0,
        "avg_cache_writes": _avg([bucket.get("writes") for bucket in cache_overall])
        or 0.0,
    }


def _summarize_stage_durations(case_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stage_bucket: Dict[str, List[float]] = defaultdict(list)
    for item in case_results:
        for stage_name, duration in (item.get("stage_durations_ms") or {}).items():
            casted = _num(duration)
            if casted is not None:
                stage_bucket[stage_name].append(casted)
    stage_summary: List[Dict[str, Any]] = []
    for stage_name, durations in sorted(stage_bucket.items()):
        stage_summary.append(
            {
                "stage": stage_name,
                "avg_duration_ms": _avg(durations),
                "p50_duration_ms": _pct(durations, 0.50),
                "p95_duration_ms": _pct(durations, 0.95),
                "count": len(durations),
            }
        )
    return stage_summary


def _duration_stats(prefix: str, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = [item.get("latency_ms") for item in case_results]
    first_token = [item.get("first_token_ms") for item in case_results]
    first_answer = [item.get("first_answer_ms") for item in case_results]
    return {
        f"{prefix}_avg_total_duration_ms": _avg(total),
        f"{prefix}_p50_total_duration_ms": _pct(total, 0.50),
        f"{prefix}_p95_total_duration_ms": _pct(total, 0.95),
        f"{prefix}_avg_first_token_ms": _avg(first_token),
        f"{prefix}_p50_first_token_ms": _pct(first_token, 0.50),
        f"{prefix}_p95_first_token_ms": _pct(first_token, 0.95),
        f"{prefix}_avg_first_answer_ms": _avg(first_answer),
        f"{prefix}_p50_first_answer_ms": _pct(first_answer, 0.50),
        f"{prefix}_p95_first_answer_ms": _pct(first_answer, 0.95),
    }


def _summarize_performance(
    case_results: List[Dict[str, Any]],
    cold_case_results: Optional[List[Dict[str, Any]]] = None,
    hot_case_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cold_results = (
        cold_case_results
        if cold_case_results is not None
        else [item for item in case_results if item.get("cache_temperature") == "cold"]
    )
    hot_results = (
        hot_case_results
        if hot_case_results is not None
        else [item for item in case_results if item.get("cache_temperature") == "hot"]
    )
    main_results = hot_results or case_results
    total = [item.get("latency_ms") for item in main_results]
    first_token = [item.get("first_token_ms") for item in main_results]
    first_answer = [item.get("first_answer_ms") for item in main_results]
    return {
        "avg_total_duration_ms": _avg(total),
        "p50_total_duration_ms": _pct(total, 0.50),
        "p95_total_duration_ms": _pct(total, 0.95),
        "avg_first_token_ms": _avg(first_token),
        "p50_first_token_ms": _pct(first_token, 0.50),
        "p95_first_token_ms": _pct(first_token, 0.95),
        "avg_first_answer_ms": _avg(first_answer),
        "p50_first_answer_ms": _pct(first_answer, 0.50),
        "p95_first_answer_ms": _pct(first_answer, 0.95),
        **_duration_stats("cold", cold_results),
        **_duration_stats("hot", hot_results),
        "stages": _summarize_stage_durations(main_results),
    }


def _summarize_ragas_from_cases(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = set()
    for item in case_results:
        metric_keys.update((item.get("ragas_scores") or {}).keys())
    summary: Dict[str, Any] = {}
    for metric_key in sorted(metric_keys):
        summary[metric_key] = _avg(
            [(item.get("ragas_scores") or {}).get(metric_key) for item in case_results]
        )
    return summary


def _build_headline_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    ragas_metrics = summary.get("ragas_metrics") or {}
    retrieval_metrics = summary.get("retrieval_metrics") or {}
    performance_metrics = summary.get("performance_metrics") or {}
    pipeline_metrics = summary.get("pipeline_metrics") or {}
    result: Dict[str, Any] = {}
    for key in ("factual_correctness", "faithfulness", "response_relevancy"):
        if key in ragas_metrics:
            result[key] = ragas_metrics.get(key)
    for key in (
        "retrieved_context_precision",
        "prompt_context_precision",
        "final_context_precision",
        "recall@5",
        "mrr@5",
        "hit@5",
        "recall@3",
        "mrr@3",
        "hit@3",
    ):
        if key in retrieval_metrics:
            result[key] = retrieval_metrics.get(key)
    for key in ("retrieval_cache_rate",):
        if key in pipeline_metrics:
            result[key] = pipeline_metrics.get(key)
    for key in ("avg_total_duration_ms", "p95_total_duration_ms", "p95_first_token_ms"):
        if key in performance_metrics:
            result[key] = performance_metrics.get(key)
    return result


def _summarize_variant(
    variant_name: str,
    case_results: List[Dict[str, Any]],
    ragas_bundle: Optional[Dict[str, Any]] = None,
    description: str = "",
    technique: str = "",
    cold_case_results: Optional[List[Dict[str, Any]]] = None,
    hot_case_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    config = VARIANTS.get(variant_name, {})
    retrieval_metrics, retrieval_coverage, retrieval_ground_truth, warnings = (
        _summarize_retrieval(case_results)
    )
    summary = {
        "variant": variant_name,
        "description": description or config.get("description") or variant_name,
        "technique": technique or config.get("technique") or variant_name,
        "case_count": len(case_results),
        "ragas_metrics": {},
        "ragas_coverage": {},
        "ragas_errors": {},
        "ragas_metadata": {},
        "retrieval_metrics": retrieval_metrics,
        "retrieval_coverage": retrieval_coverage,
        "retrieval_ground_truth": retrieval_ground_truth,
        "pipeline_metrics": _summarize_pipeline(case_results),
        "performance_metrics": _summarize_performance(
            case_results,
            cold_case_results=cold_case_results,
            hot_case_results=hot_case_results,
        ),
        "warnings": warnings,
    }
    if ragas_bundle is None:
        summary["ragas_metrics"] = _summarize_ragas_from_cases(case_results)
    else:
        summary["ragas_metrics"] = ragas_bundle.get("summary") or {}
        summary["ragas_coverage"] = ragas_bundle.get("coverage") or {}
        summary["ragas_errors"] = ragas_bundle.get("errors") or {}
        summary["ragas_metadata"] = ragas_bundle.get("metadata") or {}
    summary["headline_metrics"] = _build_headline_metrics(summary)
    return summary


def _group_by_query_type(
    case_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in case_results:
        grouped[str(item.get("query_type") or "general")].append(item)
    return dict(sorted(grouped.items()))


def _numeric_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    sections = [
        summary.get("ragas_metrics") or {},
        summary.get("retrieval_metrics") or {},
        summary.get("pipeline_metrics") or {},
        {
            key: value
            for key, value in (summary.get("performance_metrics") or {}).items()
            if key != "stages"
        },
    ]
    for section in sections:
        for key, value in section.items():
            casted = _num(value)
            if casted is not None:
                numeric[key] = round(casted, 4)
    return numeric


def _delta(current: Any, baseline: Any) -> Dict[str, Any]:
    current_value = _num(current)
    baseline_value = _num(baseline)
    if current_value is None or baseline_value is None:
        return {
            "current": current,
            "baseline": baseline,
            "delta": None,
            "relative_pct": None,
        }
    absolute = round(current_value - baseline_value, 4)
    relative_pct = None
    if baseline_value != 0:
        relative_pct = round((absolute / abs(baseline_value)) * 100, 2)
    return {
        "current": round(current_value, 4),
        "baseline": round(baseline_value, 4),
        "delta": absolute,
        "relative_pct": relative_pct,
    }


def _build_comparison_report(
    variants: Dict[str, Any],
    pairwise_order: Sequence[str] | None = None,
    pairwise_current_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    pairwise_current_set = (
        {str(name) for name in pairwise_current_names}
        if pairwise_current_names is not None
        else None
    )

    def add_comparison(variant_name: str, compare_to: str) -> None:
        if variant_name == compare_to:
            return
        if variant_name not in variants or compare_to not in variants:
            return
        comparison_key = f"{variant_name}_vs_{compare_to}"
        if comparison_key in report:
            return
        payload = variants[variant_name] or {}
        variant_config = VARIANTS.get(variant_name, {})
        current_summary = payload.get("summary") or {}
        baseline_summary = variants[compare_to].get("summary") or {}
        current_numeric = _numeric_metrics(current_summary)
        baseline_numeric = _numeric_metrics(baseline_summary)
        metric_keys = sorted(set(current_numeric.keys()) | set(baseline_numeric.keys()))
        overall = {
            metric_key: _delta(
                current_numeric.get(metric_key), baseline_numeric.get(metric_key)
            )
            for metric_key in metric_keys
        }
        current_groups = payload.get("by_query_type") or {}
        baseline_groups = variants[compare_to].get("by_query_type") or {}
        group_report: Dict[str, Any] = {}
        for query_type in sorted(
            set(current_groups.keys()) | set(baseline_groups.keys())
        ):
            current_group_numeric = _numeric_metrics(
                current_groups.get(query_type) or {}
            )
            baseline_group_numeric = _numeric_metrics(
                baseline_groups.get(query_type) or {}
            )
            group_keys = sorted(
                set(current_group_numeric.keys()) | set(baseline_group_numeric.keys())
            )
            group_report[query_type] = {
                metric_key: _delta(
                    current_group_numeric.get(metric_key),
                    baseline_group_numeric.get(metric_key),
                )
                for metric_key in group_keys
            }
        report[comparison_key] = {
            "variant": variant_name,
            "compare_to": compare_to,
            "technique": payload.get("technique") or variant_config.get("technique"),
            "overall": overall,
            "by_query_type": group_report,
        }

    for variant_name, payload in variants.items():
        variant_config = VARIANTS.get(variant_name, {})
        compare_to = payload.get("compare_to") or variant_config.get("compare_to")
        if compare_to:
            add_comparison(variant_name, str(compare_to))

    if pairwise_order:
        ordered = [
            str(variant_name)
            for variant_name in pairwise_order
            if str(variant_name) in variants
        ]
        for index, variant_name in enumerate(ordered):
            if pairwise_current_set is not None and variant_name not in pairwise_current_set:
                continue
            for compare_to in ordered[:index]:
                add_comparison(variant_name, compare_to)
    return report


def evaluate_variants(
    dataset_path: str,
    variant_names: Sequence[str],
    feature_variant_specs: Sequence[Dict[str, Any]] | None = None,
    progress_callback: ProgressCallback = None,
    cancel_callback: CancelCallback = None,
) -> Dict[str, Any]:
    _raise_if_cancelled(cancel_callback)
    cases = _prepare_cases(load_cases(dataset_path))
    _, _, dataset_payload, _ = _load_dataset_payload(dataset_path)
    requested_variant_names = [str(name or "").strip() for name in variant_names or [] if str(name or "").strip()]
    registered_feature_variants: List[Dict[str, Any]] = []
    if feature_variant_specs:
        registered_feature_variants.extend(register_feature_variants(feature_variant_specs))
        requested_variant_names.extend(
            str(item.get("name") or "").strip()
            for item in registered_feature_variants
            if str(item.get("name") or "").strip()
        )
    deduped_variant_names: List[str] = []
    for variant_name in requested_variant_names:
        if variant_name not in deduped_variant_names:
            deduped_variant_names.append(variant_name)
    resolved_variants = _resolve_variants(deduped_variant_names)
    variants_payload: Dict[str, Any] = {}
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "dataset_loaded",
                "dataset_path": str(Path(dataset_path).resolve()),
                "case_count": len(cases),
                "total_variants": len(resolved_variants),
            }
        )

    total_variants = len(resolved_variants)
    service_base_url = _evaluation_query_service_base_url()
    for index, variant_name in enumerate(resolved_variants, start=1):
        _raise_if_cancelled(cancel_callback)
        total_cases = len(cases)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "variant_started",
                    "variant_name": variant_name,
                    "variant_index": index,
                    "total_variants": total_variants,
                    "total_cases": total_cases,
                }
            )

        _reset_query_cache_for_evaluation(
            service_base_url,
            reason=f"evaluation:{variant_name}:cold",
        )
        cold_case_results: List[Dict[str, Any]] = []
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "warmup_started",
                    "variant_name": variant_name,
                    "variant_index": index,
                    "total_variants": total_variants,
                    "warmup_round": 1,
                    "warmup_rounds": 1,
                    "total_cases": total_cases,
                    "cache_temperature": "cold",
                }
            )
        for case_index, case in enumerate(cases, start=1):
            _raise_if_cancelled(cancel_callback)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "warmup_case_started",
                        "variant_name": variant_name,
                        "variant_index": index,
                        "total_variants": total_variants,
                        "warmup_round": 1,
                        "warmup_rounds": 1,
                        "case_id": str(case.get("case_id") or case.get("id") or ""),
                        "case_index": case_index,
                        "total_cases": total_cases,
                        "query": _query(case),
                        "cache_temperature": "cold",
                    }
                )
            cold_case_result = _run_single_case(case, variant_name, "cold")
            cold_case_results.append(cold_case_result)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "warmup_case_completed",
                        "variant_name": variant_name,
                        "variant_index": index,
                        "total_variants": total_variants,
                        "warmup_round": 1,
                        "warmup_rounds": 1,
                        "case_id": str(cold_case_result.get("case_id") or ""),
                        "case_index": case_index,
                        "total_cases": total_cases,
                        "query": str(cold_case_result.get("query") or ""),
                        "cache_temperature": "cold",
                        "error": str(cold_case_result.get("error") or ""),
                    }
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "warmup_completed",
                    "variant_name": variant_name,
                    "variant_index": index,
                    "total_variants": total_variants,
                    "warmup_round": 1,
                    "warmup_rounds": 1,
                    "total_cases": total_cases,
                    "cache_temperature": "cold",
                }
            )

        case_results = []
        for case_index, case in enumerate(cases, start=1):
            _raise_if_cancelled(cancel_callback)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "case_started",
                        "variant_name": variant_name,
                        "variant_index": index,
                        "total_variants": total_variants,
                        "case_id": str(case.get("case_id") or case.get("id") or ""),
                        "case_index": case_index,
                        "total_cases": total_cases,
                        "query": _query(case),
                        "cache_temperature": "hot",
                    }
                )
            case_result = _run_single_case(case, variant_name, "hot")
            case_results.append(case_result)
            _raise_if_cancelled(cancel_callback)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "case_completed",
                        "variant_name": variant_name,
                        "variant_index": index,
                        "total_variants": total_variants,
                        "case_id": str(case_result.get("case_id") or ""),
                        "case_index": case_index,
                        "total_cases": total_cases,
                        "query": str(case_result.get("query") or ""),
                        "cache_temperature": "hot",
                        "error": str(case_result.get("error") or ""),
                    }
                )
        _raise_if_cancelled(cancel_callback)
        ragas_bundle = _run_ragas(case_results, cancel_callback=cancel_callback)
        _merge_ragas_scores(case_results, ragas_bundle.get("per_case_scores") or [])
        cold_by_query_type = _group_by_query_type(cold_case_results)
        grouped_summaries = {
            query_type: _summarize_variant(
                variant_name=query_type,
                case_results=grouped_cases,
                description=f"query_type={query_type}",
                technique="QueryType Slice",
                cold_case_results=cold_by_query_type.get(query_type, []),
                hot_case_results=grouped_cases,
            )
            for query_type, grouped_cases in _group_by_query_type(case_results).items()
        }
        variants_payload[variant_name] = {
            "description": VARIANTS.get(variant_name, {}).get("description"),
            "technique": VARIANTS.get(variant_name, {}).get("technique"),
            "compare_to": VARIANTS.get(variant_name, {}).get("compare_to"),
            "feature_variant": copy.deepcopy(
                VARIANTS.get(variant_name, {}).get("feature_variant") or {}
            ),
            "summary": _summarize_variant(
                variant_name,
                case_results,
                ragas_bundle,
                cold_case_results=cold_case_results,
                hot_case_results=case_results,
            ),
            "by_query_type": grouped_summaries,
            "case_results": case_results,
            "performance_case_results": cold_case_results + case_results,
        }
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "variant_completed",
                    "variant_name": variant_name,
                    "variant_index": index,
                    "total_variants": total_variants,
                    "total_cases": total_cases,
                }
            )

    _raise_if_cancelled(cancel_callback)
    final_variant = resolved_variants[-1]
    for preferred_variant in (
        "router_anchor_rescue_structured_cached",
        "router_anchor_contextual_grounded_cached",
        "router_hybrid_grounded_cached",
        "agentic_enhanced_system_cached",
        "agentic_enhanced_system",
        "final_system",
    ):
        if preferred_variant in variants_payload:
            final_variant = preferred_variant
            break
    has_feature_variants = any(
        bool(VARIANTS.get(variant_name, {}).get("is_feature_variant"))
        for variant_name in resolved_variants
    )
    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_name": (
            str(dataset_payload.get("dataset_name") or "").strip()
            if isinstance(dataset_payload, dict)
            else ""
        ),
        "case_count": len(cases),
        "evaluation_method": {
            "mode": "controlled_ablation" if has_feature_variants else "static_variants",
            "case_count": len(cases),
            "query_type_source": "dataset.query_type",
            "item_name_source": "dataset.item_names",
            "execution_order": resolved_variants,
            "cache_policy": (
                "each variant resets cache before cold pass, then reuses that cache for hot pass"
            ),
            "feature_variants": [
                copy.deepcopy(VARIANTS.get(variant_name, {}).get("feature_variant"))
                for variant_name in resolved_variants
                if VARIANTS.get(variant_name, {}).get("feature_variant")
            ],
            "performance_sampling": {
                "cache_temperature": "explicit_cold_hot",
                "quality_scoring_source": "hot",
            },
        },
        "variants": variants_payload,
        "final_variant": final_variant,
        "final_system_metrics": variants_payload.get(final_variant, {}).get(
            "summary", {}
        ),
        "comparisons": _build_comparison_report(
            variants_payload,
            pairwise_order=resolved_variants if has_feature_variants else None,
        ),
    }
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "report_ready",
                "final_variant": final_variant,
                "case_count": len(cases),
                "total_variants": total_variants,
            }
        )
    return report


def evaluate_variants_to_file(
    dataset_path: str,
    variant_names: Sequence[str],
    output_path: str | None = None,
    feature_variant_specs: Sequence[Dict[str, Any]] | None = None,
    progress_callback: ProgressCallback = None,
    cancel_callback: CancelCallback = None,
) -> Dict[str, Any]:
    report = evaluate_variants(
        dataset_path,
        variant_names,
        feature_variant_specs=feature_variant_specs,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    resolved_output_path = Path(output_path) if output_path else _default_output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "report_saved",
                "output_path": str(resolved_output_path.resolve()),
            }
        )
    return {
        "report": report,
        "output_path": str(resolved_output_path.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一 RAG / RAGAS 评测入口")
    parser.add_argument("dataset", help="评测数据集路径，支持 JSON / JSONL")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(DEFAULT_VARIANTS),
        help="要执行的评测变体，默认运行内置统一对比集合",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_path()),
        help="输出 JSON 报告路径",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = evaluate_variants_to_file(args.dataset, args.variants, args.output)
    report = result["report"]
    output_path = Path(result["output_path"])
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "final_variant": report.get("final_variant"),
                "headline_metrics": report.get("final_system_metrics", {}).get(
                    "headline_metrics", {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
