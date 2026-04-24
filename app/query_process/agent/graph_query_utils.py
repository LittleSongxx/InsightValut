import re
from typing import Any, Dict, List, Sequence
from app.conf.query_threshold_config import query_threshold_config
from app.utils.anchor_context_utils import (
    classify_router_query_family,
    extract_query_anchor_targets,
)

GRAPH_PREFERRED_QUERY_TYPES = {
    "navigation",
    "comparison",
    "relation",
    "constraint",
    "explain",
}

_QUERY_TYPE_KEYWORDS = {
    "navigation": [
        "上一步",
        "下一步",
        "前一步",
        "后一步",
        "哪一步",
        "步骤",
        "顺序",
        "对应哪步",
        "对应哪一步",
        "属于哪个章节",
        "属于哪一章",
        "属于哪个部分",
        "在哪一节",
        "导航",
        "流程",
        "图属于",
        "警告对应",
    ],
    "comparison": [
        "区别",
        "差异",
        "比较",
        "对比",
        "相比",
        "不同",
        "兼容",
        "适配",
        "参数差异",
        "参数对比",
        "哪个更",
        "和",
    ],
    "relation": [
        "关联",
        "关系",
        "相关",
        "原因",
        "导致",
        "故障",
        "排查",
        "排除",
        "部件",
        "组件",
        "影响",
        "连带",
        "可能是",
    ],
    "constraint": [
        "满足",
        "条件",
        "必须",
        "同时",
        "组合",
        "约束",
        "符合",
        "支持",
        "适合",
        "可用于",
        "需要",
        "要求",
    ],
    "explain": [
        "为什么",
        "依据",
        "证据",
        "来自哪里",
        "来自哪",
        "实体",
        "关系",
        "可解释",
        "来源",
        "怎么得出",
        "从哪几个",
    ],
}

_STOP_TERMS = {
    "这个",
    "那个",
    "一下",
    "一下子",
    "什么",
    "怎么",
    "如何",
    "是否",
    "可以",
    "哪些",
    "哪个",
    "以及",
    "还有",
    "关于",
    "对于",
    "请问",
    "一下子",
    "里面",
    "对应",
    "说明",
    "手册",
    "产品",
    "商品",
    "设备",
    "机器",
}

_QUERY_TYPE_PRIORITY = [
    "explain",
    "comparison",
    "constraint",
    "relation",
    "navigation",
    "general",
]

_COMPLEXITY_MARKERS = (
    "并且",
    "以及",
    "同时",
    "分别",
    "对比",
    "比较",
    "区别",
    "差异",
    "原因",
    "为什么",
    "是否",
    "如果",
    "怎么得出",
    "哪一步",
    "上一步",
    "下一步",
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _keyword_score(query: str, keywords: Sequence[str]) -> int:
    lowered = query.lower()
    return sum(1 for keyword in keywords if keyword and keyword.lower() in lowered)


def extract_focus_terms(
    query: str, item_names: Sequence[str] | None = None
) -> List[str]:
    text = _clean_text(query)
    for item_name in item_names or []:
        cleaned = _clean_text(str(item_name))
        if cleaned:
            text = text.replace(cleaned, " ")

    candidates: List[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*|[\u4e00-\u9fff]{2,16}", text):
        normalized = token.strip().strip("，。；：、,.!?！？（）()[]【】")
        if not normalized or normalized in _STOP_TERMS:
            continue
        if normalized.isdigit():
            continue
        if normalized not in candidates:
            candidates.append(normalized)

    return candidates[:8]


def classify_query_type(
    query: str, item_names: Sequence[str] | None = None
) -> Dict[str, Any]:
    normalized_query = _clean_text(query)
    if not normalized_query:
        return {
            "query_type": "general",
            "graph_preferred": False,
            "focus_terms": [],
            "reason": "empty_query",
        }

    scores = {
        query_type: _keyword_score(normalized_query, keywords)
        for query_type, keywords in _QUERY_TYPE_KEYWORDS.items()
    }

    query_type = "general"
    best_score = 0
    for candidate in _QUERY_TYPE_PRIORITY:
        if candidate == "general":
            continue
        current_score = scores.get(candidate, 0)
        if current_score > best_score:
            query_type = candidate
            best_score = current_score

    if query_type == "comparison":
        item_name_count = len(
            [name for name in item_names or [] if _clean_text(str(name))]
        )
        if item_name_count < 2 and not any(
            keyword in normalized_query
            for keyword in ("区别", "差异", "比较", "对比", "兼容", "适配")
        ):
            query_type = "general"
            best_score = 0

    graph_preferred = query_type in GRAPH_PREFERRED_QUERY_TYPES
    focus_terms = extract_focus_terms(normalized_query, item_names)

    if query_type == "general" and any(
        term in normalized_query for term in ("故障", "原因", "步骤", "区别", "证据")
    ):
        query_type = (
            "relation"
            if "故障" in normalized_query or "原因" in normalized_query
            else "navigation"
        )
        graph_preferred = True

    reason = f"keyword_score={best_score}" if best_score > 0 else "fallback_general"
    return {
        "query_type": query_type,
        "graph_preferred": graph_preferred,
        "focus_terms": focus_terms,
        "reason": reason,
    }


def build_retrieval_plan(
    query_type: str, graph_preferred: bool, bm25_enabled: bool | None = None
) -> Dict[str, Any]:
    resolved_bm25_enabled = (
        query_threshold_config.bm25_enabled
        if bm25_enabled is None
        else bool(bm25_enabled)
    )
    plan: Dict[str, Any] = {
        "graph_first": graph_preferred,
        "run_kg": True,
        "run_embedding": True,
        "run_anchor": False,
        "run_bm25": resolved_bm25_enabled,
        "run_hyde": not graph_preferred,
        "run_web": not graph_preferred,
        "graph_limit": 6,
        "kg_weight_multiplier": 1.0,
        "embedding_weight_multiplier": 1.0,
        "anchor_weight_multiplier": 1.0,
        "bm25_weight_multiplier": 1.0,
        "hyde_weight_multiplier": 1.0,
    }

    if not graph_preferred:
        return plan

    plan.update(
        {
            "run_web": False,
            "run_hyde": False,
            "graph_limit": 8,
            "kg_weight_multiplier": 1.8,
            "embedding_weight_multiplier": 0.9,
            "bm25_weight_multiplier": 0.9,
            "hyde_weight_multiplier": 0.4,
        }
    )

    if query_type == "navigation":
        plan.update(
            {
                "graph_limit": 10,
                "kg_weight_multiplier": 2.2,
                "embedding_weight_multiplier": 0.8,
                "bm25_weight_multiplier": 0.8,
            }
        )
    elif query_type == "comparison":
        plan.update(
            {
                "graph_limit": 12,
                "kg_weight_multiplier": 2.0,
                "embedding_weight_multiplier": 0.85,
                "bm25_weight_multiplier": 1.0,
            }
        )
    elif query_type == "relation":
        plan.update(
            {
                "graph_limit": 10,
                "kg_weight_multiplier": 2.1,
                "embedding_weight_multiplier": 0.8,
                "bm25_weight_multiplier": 0.9,
            }
        )
    elif query_type == "constraint":
        plan.update(
            {
                "graph_limit": 12,
                "kg_weight_multiplier": 2.0,
                "embedding_weight_multiplier": 0.85,
                "bm25_weight_multiplier": 1.0,
            }
        )
    elif query_type == "explain":
        plan.update(
            {
                "graph_limit": 12,
                "kg_weight_multiplier": 2.3,
                "embedding_weight_multiplier": 0.8,
                "bm25_weight_multiplier": 0.8,
            }
        )

    return plan


def classify_query_complexity(
    query: str,
    item_names: Sequence[str] | None = None,
    *,
    query_type: str = "general",
    focus_terms: Sequence[str] | None = None,
) -> Dict[str, Any]:
    normalized_query = _clean_text(query)
    normalized_item_names = [
        _clean_text(str(name)) for name in (item_names or []) if _clean_text(str(name))
    ]
    resolved_focus_terms = [
        _clean_text(str(term)) for term in (focus_terms or []) if _clean_text(str(term))
    ]
    token_count = len(
        re.findall(
            r"[A-Za-z0-9][A-Za-z0-9._/-]*|[\u4e00-\u9fff]{1,16}",
            normalized_query,
        )
    )
    marker_hits = [
        marker for marker in _COMPLEXITY_MARKERS if marker in normalized_query
    ]
    has_multi_clause = bool(
        re.search(r"[，；;、]|(并且|以及|同时|分别|如果|是否)", normalized_query)
    )

    if not normalized_query:
        return {
            "query_complexity": "simple",
            "reason": "empty_query",
        }

    complexity = "simple"
    reason = "single_factoid"
    if query_type in GRAPH_PREFERRED_QUERY_TYPES:
        complexity = "complex"
        reason = f"query_type={query_type}"
    elif len(normalized_item_names) >= 2:
        complexity = "complex"
        reason = "multiple_item_targets"
    elif len(resolved_focus_terms) >= 5:
        complexity = "complex"
        reason = "many_focus_terms"
    elif token_count >= 12:
        complexity = "complex"
        reason = "long_query"
    elif has_multi_clause and marker_hits:
        complexity = "complex"
        reason = f"markers={','.join(marker_hits[:3])}"
    elif has_multi_clause and token_count >= 8:
        complexity = "complex"
        reason = "multi_clause_query"

    return {
        "query_complexity": complexity,
        "reason": reason,
    }


def build_query_route(
    query: str, item_names: Sequence[str] | None = None
) -> Dict[str, Any]:
    route = classify_query_type(query, item_names=item_names)
    complexity = classify_query_complexity(
        query,
        item_names=item_names,
        query_type=str(route.get("query_type") or "general"),
        focus_terms=route.get("focus_terms") or [],
    )
    route["retrieval_plan"] = build_retrieval_plan(
        route["query_type"], route["graph_preferred"]
    )
    anchor_targets = extract_query_anchor_targets(query, item_names=item_names)
    route["query_anchor_targets"] = anchor_targets
    route["router_query_family"] = classify_router_query_family(
        query,
        str(route.get("query_type") or "general"),
        anchor_targets,
    )
    if route["router_query_family"] in {
        "section_summary",
        "section_lookup",
        "procedure_lookup",
    }:
        route["graph_preferred"] = False
        route["retrieval_plan"] = build_retrieval_plan(
            route["query_type"],
            False,
        )
        complexity["query_complexity"] = "simple"
        complexity["reason"] = route["router_query_family"]
    route["query_complexity"] = complexity["query_complexity"]
    route["query_complexity_reason"] = complexity["reason"]
    return route


def _get_evaluation_overrides(state: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    merged: Dict[str, Any] = {}
    runtime_overrides = state.get("route_overrides")
    if isinstance(runtime_overrides, dict):
        merged.update(runtime_overrides)
    overrides = state.get("evaluation_overrides")
    if isinstance(overrides, dict):
        merged.update(overrides)
    return merged


def get_bm25_enabled(state: Dict[str, Any] | None = None) -> bool:
    overrides = _get_evaluation_overrides(state)
    if "bm25_enabled" in overrides:
        return bool(overrides.get("bm25_enabled"))
    return bool(query_threshold_config.bm25_enabled)


def get_grounded_mode(state: Dict[str, Any] | None = None) -> bool:
    overrides = _get_evaluation_overrides(state)
    if "grounded_answer_enabled" in overrides:
        return bool(overrides.get("grounded_answer_enabled"))
    return bool((state or {}).get("grounded_mode", False))


def is_router_deep_search_enabled(state: Dict[str, Any] | None = None) -> bool:
    overrides = _get_evaluation_overrides(state)
    if "router_deep_search_enabled" in overrides:
        return bool(overrides.get("router_deep_search_enabled"))
    return bool((state or {}).get("router_deep_search_enabled", False))


def apply_route_overrides(
    route: Dict[str, Any], state: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    overrides = _get_evaluation_overrides(state)
    if not overrides:
        return route

    query_type = (
        str(
            overrides.get("force_query_type") or route.get("query_type") or "general"
        ).strip()
        or "general"
    )
    if "force_graph_preferred" in overrides:
        graph_preferred = bool(overrides.get("force_graph_preferred"))
    else:
        graph_preferred = bool(route.get("graph_preferred", False))

    updated_route = dict(route)
    updated_route["query_type"] = query_type
    updated_route["graph_preferred"] = graph_preferred
    if "force_focus_terms" in overrides and isinstance(
        overrides.get("force_focus_terms"), list
    ):
        updated_route["focus_terms"] = overrides.get("force_focus_terms") or []
    updated_route["reason"] = str(
        overrides.get("route_reason") or updated_route.get("reason") or "evaluation"
    )
    updated_route["retrieval_plan"] = build_retrieval_plan(
        query_type,
        graph_preferred,
        bm25_enabled=get_bm25_enabled(state),
    )
    state_query_original = str((state or {}).get("original_query") or "")
    state_query_rewritten = str((state or {}).get("rewritten_query") or "")
    state_item_names = (state or {}).get("item_names") or []
    anchor_targets = list(
        updated_route.get("query_anchor_targets")
        or route.get("query_anchor_targets")
        or extract_query_anchor_targets(state_query_original, state_item_names)
        or extract_query_anchor_targets(state_query_rewritten, state_item_names)
    )
    updated_route["query_anchor_targets"] = anchor_targets
    router_query_family = str(
        updated_route.get("router_query_family")
        or route.get("router_query_family")
        or "general"
    )
    if router_query_family == "general" and anchor_targets:
        router_query_family = classify_router_query_family(
            state_query_original or state_query_rewritten,
            query_type,
            anchor_targets,
        )
    updated_route["router_query_family"] = router_query_family

    retrieval_plan_overrides = overrides.get("retrieval_plan_overrides") or {}
    if isinstance(retrieval_plan_overrides, dict):
        updated_plan = dict(updated_route.get("retrieval_plan") or {})
        updated_plan.update(retrieval_plan_overrides)
        updated_route["retrieval_plan"] = updated_plan

    query_complexity = str(
        updated_route.get("query_complexity")
        or route.get("query_complexity")
        or "simple"
    ).strip() or "simple"
    updated_route["query_complexity"] = query_complexity
    updated_route["query_complexity_reason"] = str(
        updated_route.get("query_complexity_reason")
        or route.get("query_complexity_reason")
        or "unknown"
    )

    router_enabled = is_router_deep_search_enabled(state)
    grounded_mode = get_grounded_mode(state)
    anchor_context_enabled = bool(overrides.get("anchor_context_enabled", False))
    router_decision = "default_path"
    crag_router_enabled = bool(overrides.get("retrieval_grader_enabled", True))

    if router_enabled:
        updated_plan = dict(updated_route.get("retrieval_plan") or {})
        if anchor_context_enabled:
            updated_plan["run_anchor"] = True
            if router_query_family in {
                "section_summary",
                "section_lookup",
                "procedure_lookup",
                "comparison",
            }:
                updated_plan["run_hyde"] = False
                crag_router_enabled = False
                router_decision = "anchor_grounded_path"
        if query_complexity == "simple":
            updated_plan["run_hyde"] = False
            router_decision = (
                "anchor_grounded_path"
                if anchor_context_enabled and updated_plan.get("run_anchor")
                else "simple_fast_path"
            )
            crag_router_enabled = False
        elif router_decision == "default_path":
            router_decision = "complex_deep_path"
            updated_plan["run_hyde"] = bool(
                retrieval_plan_overrides.get("run_hyde", True)
                if isinstance(retrieval_plan_overrides, dict)
                else True
            )
            crag_router_enabled = bool(overrides.get("retrieval_grader_enabled", True))
        updated_route["retrieval_plan"] = updated_plan

    updated_route["router_deep_search_enabled"] = router_enabled
    updated_route["router_decision"] = router_decision
    updated_route["crag_router_enabled"] = crag_router_enabled
    updated_route["grounded_mode"] = grounded_mode

    return updated_route


def should_run_retriever(state: Dict[str, Any], source: str) -> bool:
    if source == "bm25" and not get_bm25_enabled(state):
        return False

    plan = state.get("retrieval_plan") or {}
    key_map = {
        "kg": "run_kg",
        "anchor": "run_anchor",
        "embedding": "run_embedding",
        "bm25": "run_bm25",
        "hyde": "run_hyde",
        "web": "run_web",
    }
    plan_key = key_map.get(source)
    if not plan_key:
        return True
    return bool(plan.get(plan_key, True))


def get_rrf_weight_multipliers(state: Dict[str, Any]) -> Dict[str, float]:
    plan = state.get("retrieval_plan") or {}
    return {
        "embedding": float(plan.get("embedding_weight_multiplier", 1.0) or 1.0),
        "anchor": float(plan.get("anchor_weight_multiplier", 1.0) or 1.0),
        "hyde": float(plan.get("hyde_weight_multiplier", 1.0) or 1.0),
        "bm25": float(plan.get("bm25_weight_multiplier", 1.0) or 1.0),
        "kg": float(plan.get("kg_weight_multiplier", 1.0) or 1.0),
    }
