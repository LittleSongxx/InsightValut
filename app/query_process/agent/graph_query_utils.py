import re
from typing import Any, Dict, List, Sequence

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


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _keyword_score(query: str, keywords: Sequence[str]) -> int:
    lowered = query.lower()
    return sum(1 for keyword in keywords if keyword and keyword.lower() in lowered)


def extract_focus_terms(query: str, item_names: Sequence[str] | None = None) -> List[str]:
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


def classify_query_type(query: str, item_names: Sequence[str] | None = None) -> Dict[str, Any]:
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
        item_name_count = len([name for name in item_names or [] if _clean_text(str(name))])
        if item_name_count < 2 and not any(
            keyword in normalized_query for keyword in ("区别", "差异", "比较", "对比", "兼容", "适配")
        ):
            query_type = "general"
            best_score = 0

    graph_preferred = query_type in GRAPH_PREFERRED_QUERY_TYPES
    focus_terms = extract_focus_terms(normalized_query, item_names)

    if query_type == "general" and any(term in normalized_query for term in ("故障", "原因", "步骤", "区别", "证据")):
        query_type = "relation" if "故障" in normalized_query or "原因" in normalized_query else "navigation"
        graph_preferred = True

    reason = f"keyword_score={best_score}" if best_score > 0 else "fallback_general"
    return {
        "query_type": query_type,
        "graph_preferred": graph_preferred,
        "focus_terms": focus_terms,
        "reason": reason,
    }


def build_retrieval_plan(query_type: str, graph_preferred: bool) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "graph_first": graph_preferred,
        "run_kg": True,
        "run_embedding": True,
        "run_bm25": True,
        "run_hyde": not graph_preferred,
        "run_web": not graph_preferred,
        "graph_limit": 6,
        "kg_weight_multiplier": 1.0,
        "embedding_weight_multiplier": 1.0,
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
        plan.update({
            "graph_limit": 10,
            "kg_weight_multiplier": 2.2,
            "embedding_weight_multiplier": 0.8,
            "bm25_weight_multiplier": 0.8,
        })
    elif query_type == "comparison":
        plan.update({
            "graph_limit": 12,
            "kg_weight_multiplier": 2.0,
            "embedding_weight_multiplier": 0.85,
            "bm25_weight_multiplier": 1.0,
        })
    elif query_type == "relation":
        plan.update({
            "graph_limit": 10,
            "kg_weight_multiplier": 2.1,
            "embedding_weight_multiplier": 0.8,
            "bm25_weight_multiplier": 0.9,
        })
    elif query_type == "constraint":
        plan.update({
            "graph_limit": 12,
            "kg_weight_multiplier": 2.0,
            "embedding_weight_multiplier": 0.85,
            "bm25_weight_multiplier": 1.0,
        })
    elif query_type == "explain":
        plan.update({
            "graph_limit": 12,
            "kg_weight_multiplier": 2.3,
            "embedding_weight_multiplier": 0.8,
            "bm25_weight_multiplier": 0.8,
        })

    return plan


def build_query_route(query: str, item_names: Sequence[str] | None = None) -> Dict[str, Any]:
    route = classify_query_type(query, item_names=item_names)
    route["retrieval_plan"] = build_retrieval_plan(
        route["query_type"], route["graph_preferred"]
    )
    return route


def should_run_retriever(state: Dict[str, Any], source: str) -> bool:
    plan = state.get("retrieval_plan") or {}
    key_map = {
        "kg": "run_kg",
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
        "hyde": float(plan.get("hyde_weight_multiplier", 1.0) or 1.0),
        "bm25": float(plan.get("bm25_weight_multiplier", 1.0) or 1.0),
        "kg": float(plan.get("kg_weight_multiplier", 1.0) or 1.0),
    }
