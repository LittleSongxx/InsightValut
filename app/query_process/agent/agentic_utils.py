import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from app.conf.query_threshold_config import query_threshold_config

DEFAULT_AGENTIC_FEATURES: Dict[str, bool] = {
    "subquery_routing": True,
    "context_expansion": True,
    "evidence_coverage": True,
    "retrieval_rescue": True,
    "structured_answer": True,
    "clarification_guard": True,
}

_GENERIC_FOCUS_TERMS = {
    "步骤",
    "流程",
    "说明",
    "产品",
    "设备",
    "机器",
    "参数",
    "为什么",
    "证据",
    "哪里",
    "多少",
}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_term(value: Any) -> str:
    return _clean_text(value).lower()


def _doc_blob(doc: Dict[str, Any]) -> str:
    parts = [
        doc.get("title"),
        doc.get("text"),
        doc.get("content"),
        doc.get("graph_fact"),
        doc.get("evidence_text"),
    ]
    return "\n".join(_clean_text(part).lower() for part in parts if _clean_text(part))


def _extract_terms(text: str) -> List[str]:
    tokens: List[str] = []
    for token in re.findall(
        r"[A-Za-z0-9][A-Za-z0-9._/-]*|[\u4e00-\u9fff]{2,16}", _clean_text(text)
    ):
        normalized = _normalize_term(token)
        if not normalized or normalized in _GENERIC_FOCUS_TERMS or normalized.isdigit():
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens[:8]


def _question_has_explicit_product_reference(question: str) -> bool:
    normalized = _clean_text(question)
    if not normalized:
        return False
    explicit_patterns = (
        r"\b[A-Za-z]{1,10}[A-Za-z0-9]*[-_][A-Za-z0-9._/-]+\b",
        r"\b[A-Za-z]+[A-Za-z0-9]*\d+[A-Za-z0-9._/-]*\b",
        r"[\u4e00-\u9fff]{2,12}[A-Za-z0-9]+[-_][A-Za-z0-9._/-]+",
    )
    return any(re.search(pattern, normalized) for pattern in explicit_patterns)


def get_agentic_features(state: Dict[str, Any] | None) -> Dict[str, bool]:
    features = dict(DEFAULT_AGENTIC_FEATURES)
    if not isinstance(state, dict):
        return features

    for container in (
        state,
        state.get("route_overrides") or {},
        state.get("evaluation_overrides") or {},
    ):
        config = container.get("agentic_features") if isinstance(container, dict) else {}
        if not isinstance(config, dict):
            continue
        for key, value in config.items():
            if key in features:
                features[key] = bool(value)
    return features


def is_agentic_feature_enabled(state: Dict[str, Any] | None, feature_name: str) -> bool:
    return bool(get_agentic_features(state).get(feature_name, True))


def build_clarification_request(state: Dict[str, Any]) -> Dict[str, Any]:
    query_type = str(state.get("query_type") or "general")
    question = _clean_text(state.get("rewritten_query") or state.get("original_query"))
    item_names = [_clean_text(name) for name in (state.get("item_names") or []) if _clean_text(name)]
    focus_terms = [
        _clean_text(term)
        for term in (state.get("query_focus_terms") or _extract_terms(question))
        if _clean_text(term)
    ]

    if query_type == "comparison" and len(item_names) < 2 and len(focus_terms) < 2:
        return {
            "required": True,
            "reason": "missing_comparison_targets",
            "question": "您想对比哪两个产品或型号？请至少明确提供两个型号，我再帮您做差异对比。",
            "options": item_names,
        }

    if (
        query_type == "navigation"
        and not item_names
        and not _question_has_explicit_product_reference(question)
    ):
        return {
            "required": True,
            "reason": "missing_navigation_product",
            "question": "您想查询哪一款产品的步骤导航？请补充具体产品或型号，以及您关心的步骤/章节。",
            "options": focus_terms[:3],
        }

    if query_type == "constraint" and len(focus_terms) < 2:
        return {
            "required": True,
            "reason": "missing_constraint_terms",
            "question": "您希望系统按哪些条件筛选？请尽量明确 2 个以上约束条件，例如参数、部件、适配关系或使用场景。",
            "options": item_names[:2],
        }

    if query_type == "relation" and not item_names and len(focus_terms) < 1:
        return {
            "required": True,
            "reason": "missing_relation_focus",
            "question": "您想分析哪一款产品、哪个部件或哪种故障现象之间的关系？请补充更具体的对象。",
            "options": [],
        }

    if query_type == "explain" and len(focus_terms) < 1:
        return {
            "required": True,
            "reason": "missing_explain_target",
            "question": "您希望我解释哪条结论、哪一步骤，或者哪组实体关系？请补充您想追问的具体对象。",
            "options": item_names[:2],
        }

    if not item_names and any(keyword in question for keyword in ("这款", "这个", "它", "该设备")):
        return {
            "required": True,
            "reason": "missing_referent_product",
            "question": "您当前的问题还缺少明确的产品或型号。请告诉我具体是哪一款设备，我再继续检索。",
            "options": [],
        }

    return {"required": False, "reason": "", "question": "", "options": []}


def analyze_evidence_coverage(state: Dict[str, Any]) -> Dict[str, Any]:
    reranked_docs = state.get("reranked_docs") or []
    question = _clean_text(state.get("rewritten_query") or state.get("original_query"))
    query_type = str(state.get("query_type") or "general")
    focus_terms = [
        _normalize_term(term)
        for term in (state.get("query_focus_terms") or _extract_terms(question))
        if _normalize_term(term)
    ]
    item_names = [
        _normalize_term(name) for name in (state.get("item_names") or []) if _normalize_term(name)
    ]
    sub_queries = [_clean_text(item) for item in (state.get("sub_queries") or []) if _clean_text(item)]
    doc_blobs = [_doc_blob(doc) for doc in reranked_docs if isinstance(doc, dict)]
    source_counts = Counter(
        _clean_text(doc.get("source") or "local") for doc in reranked_docs if isinstance(doc, dict)
    )

    covered_terms: List[str] = []
    missing_terms: List[str] = []
    for term in focus_terms:
        if any(term in blob for blob in doc_blobs):
            covered_terms.append(term)
        else:
            missing_terms.append(term)

    covered_items: List[str] = []
    missing_items: List[str] = []
    for item_name in item_names:
        if any(item_name in blob for blob in doc_blobs):
            covered_items.append(item_name)
        else:
            missing_items.append(item_name)

    covered_sub_queries: List[str] = []
    missing_sub_queries: List[str] = []
    for sub_query in sub_queries:
        sub_terms = _extract_terms(sub_query) or [_normalize_term(sub_query)]
        hit_count = sum(
            1
            for term in sub_terms
            if term and any(term in blob for blob in doc_blobs)
        )
        if sub_terms and hit_count / len(sub_terms) >= 0.5:
            covered_sub_queries.append(sub_query)
        else:
            missing_sub_queries.append(sub_query)

    doc_count = len(reranked_docs)
    coverage_terms_ratio = (
        len(covered_terms) / len(focus_terms) if focus_terms else 1.0
    )
    coverage_item_ratio = (
        len(covered_items) / len(item_names) if item_names else 1.0
    )
    coverage_sub_ratio = (
        len(covered_sub_queries) / len(sub_queries) if sub_queries else 1.0
    )
    doc_ratio = min(
        doc_count / max(query_threshold_config.evidence_min_docs, 1),
        1.0,
    )
    coverage_score = round(
        coverage_terms_ratio * 0.35
        + coverage_item_ratio * 0.20
        + coverage_sub_ratio * 0.25
        + doc_ratio * 0.20,
        4,
    )

    clarification = build_clarification_request(state)
    needs_rescue = (
        coverage_score < query_threshold_config.evidence_min_coverage_score
        or doc_count < query_threshold_config.evidence_min_docs
        or bool(missing_sub_queries)
        or bool(missing_items)
    )

    return {
        "query_type": query_type,
        "question": question,
        "focus_terms": focus_terms,
        "covered_focus_terms": covered_terms,
        "missing_focus_terms": missing_terms,
        "covered_item_names": covered_items,
        "missing_item_names": missing_items,
        "covered_sub_queries": covered_sub_queries,
        "missing_sub_queries": missing_sub_queries,
        "doc_count": doc_count,
        "source_counts": dict(source_counts),
        "coverage_terms_ratio": round(coverage_terms_ratio, 4),
        "coverage_item_ratio": round(coverage_item_ratio, 4),
        "coverage_sub_ratio": round(coverage_sub_ratio, 4),
        "coverage_score": coverage_score,
        "needs_rescue": needs_rescue,
        "clarification_required": bool(clarification.get("required")),
        "clarification_reason": clarification.get("reason", ""),
        "clarification_question": clarification.get("question", ""),
        "clarification_options": clarification.get("options", []),
    }


def build_retrieval_rescue_plan(
    state: Dict[str, Any], grade_result: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    grade_result = grade_result or {}
    coverage = state.get("evidence_coverage_summary") or {}
    query_type = str(state.get("query_type") or "general")
    item_names = [_clean_text(name) for name in (state.get("item_names") or []) if _clean_text(name)]
    retrieval_plan = dict(state.get("retrieval_plan") or {})
    kg_summary = state.get("kg_query_summary") or {}
    question = _clean_text(state.get("rewritten_query") or state.get("original_query"))
    features = get_agentic_features(state)

    if not features.get("retrieval_rescue", True):
        return {
            "action": "none",
            "reason": "retrieval_rescue_disabled",
            "route_overrides": {},
            "question": question,
            "steps": [],
        }

    if coverage.get("clarification_required") and features.get(
        "clarification_guard", True
    ):
        return {
            "action": "clarify",
            "reason": coverage.get("clarification_reason") or "clarification_required",
            "route_overrides": {},
            "question": question,
            "clarification_question": coverage.get("clarification_question") or "",
            "clarification_options": coverage.get("clarification_options") or [],
            "steps": ["clarification_guard"],
        }

    route_overrides: Dict[str, Any] = {
        "route_reason": "agentic_rescue_retry",
        "retrieval_plan_overrides": {},
    }
    steps: List[str] = []
    plan_overrides = route_overrides["retrieval_plan_overrides"]

    suggested_query = _clean_text(grade_result.get("suggested_query") or "")
    next_query = suggested_query or question

    if query_type in {"navigation", "relation", "constraint", "comparison", "explain"}:
        route_overrides["force_query_type"] = query_type
        route_overrides["force_graph_preferred"] = True
        plan_overrides["run_kg"] = True
        plan_overrides["graph_limit"] = max(
            int(retrieval_plan.get("graph_limit", 8) or 8),
            query_threshold_config.context_expand_top_k + 6,
        )
        plan_overrides["kg_weight_multiplier"] = max(
            float(retrieval_plan.get("kg_weight_multiplier", 1.0) or 1.0),
            2.2,
        )
        steps.append("widen_graph_context")

    if not coverage.get("doc_count"):
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
        plan_overrides["run_hyde"] = True
        if query_type == "general":
            plan_overrides["run_web"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("enable_multi_retriever")

    if coverage.get("missing_sub_queries"):
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("cover_missing_sub_queries")

    if item_names and coverage.get("missing_item_names") == [name.lower() for name in item_names]:
        route_overrides["drop_item_names"] = True
        plan_overrides["run_web"] = query_type == "general"
        steps.append("drop_item_filter")

    if kg_summary.get("result_count", 0) <= 1 and query_type in {"navigation", "relation", "explain"}:
        plan_overrides["graph_limit"] = max(
            int(plan_overrides.get("graph_limit") or retrieval_plan.get("graph_limit", 8) or 8),
            12,
        )
        plan_overrides["run_bm25"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("boost_graph_with_bm25")

    if coverage.get("coverage_score", 0.0) < 0.35 and not steps:
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
        plan_overrides["run_hyde"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("broad_retry")

    if not steps:
        steps.append("query_rewrite_retry")

    return {
        "action": "retry",
        "reason": grade_result.get("reason") or "insufficient_coverage",
        "question": next_query,
        "route_overrides": route_overrides,
        "steps": steps,
        "coverage_score": coverage.get("coverage_score"),
        "missing_focus_terms": coverage.get("missing_focus_terms") or [],
        "missing_sub_queries": coverage.get("missing_sub_queries") or [],
    }


def build_answer_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    query_type = str(state.get("query_type") or "general")
    structured_enabled = is_agentic_feature_enabled(state, "structured_answer")
    item_names = [_clean_text(name) for name in (state.get("item_names") or []) if _clean_text(name)]
    coverage = state.get("evidence_coverage_summary") or {}
    kg_summary = state.get("kg_query_summary") or {}
    reranked_docs = state.get("reranked_docs") or []
    top_titles = [
        _clean_text(doc.get("title") or "")
        for doc in reranked_docs[:4]
        if _clean_text(doc.get("title") or "")
    ]
    top_graph_facts = [
        _clean_text(doc.get("graph_fact") or "")
        for doc in reranked_docs[:3]
        if _clean_text(doc.get("graph_fact") or "")
    ]

    plan = {
        "query_type": query_type,
        "structured_output": structured_enabled
        and query_type
        in {"navigation", "comparison", "relation", "constraint", "explain"},
        "response_format": "paragraph",
        "sections": ["直接回答", "补充说明"],
        "style_instructions": ["优先基于证据给出直接结论。"],
        "must_cover": item_names + list(coverage.get("covered_focus_terms") or [])[:4],
        "top_titles": top_titles,
        "top_graph_facts": top_graph_facts,
        "coverage_score": coverage.get("coverage_score"),
        "graph_summary": kg_summary,
    }

    if not structured_enabled:
        return plan

    if query_type == "navigation":
        plan.update(
            {
                "response_format": "ordered_list",
                "sections": ["当前步骤", "上一步与下一步", "注意事项"],
                "style_instructions": [
                    "先回答用户当前想找的步骤，再交代相邻步骤和章节定位。",
                    "如果存在顺序关系，优先使用编号列表。",
                ],
            }
        )
    elif query_type == "comparison":
        plan.update(
            {
                "response_format": "table_plus_summary",
                "sections": ["对比结论", "关键差异", "适配/兼容性说明"],
                "style_instructions": [
                    "先给出一句总结，再给出 Markdown 表格。",
                    "只对比证据中明确出现的参数和关系。",
                ],
            }
        )
    elif query_type == "relation":
        plan.update(
            {
                "response_format": "bullet_chain",
                "sections": ["关系结论", "可能原因/部件/步骤", "证据链"],
                "style_instructions": [
                    "把关系链拆成要点，避免一段话堆砌。",
                    "如果有因果或解决关系，显式写出箭头或先后关系。",
                ],
            }
        )
    elif query_type == "constraint":
        plan.update(
            {
                "response_format": "decision_matrix",
                "sections": ["是否满足", "满足证据", "待确认项"],
                "style_instructions": [
                    "先判断是否满足，再列出每条条件的对应证据。",
                    "缺证据的条件要明确标注待确认。",
                ],
            }
        )
    elif query_type == "explain":
        plan.update(
            {
                "response_format": "evidence_chain",
                "sections": ["结论", "证据链", "涉及实体与关系"],
                "style_instructions": [
                    "解释型问题必须先给结论，再说明结论来自哪些实体和关系。",
                    "尽量引用 chunk_id 或标题作为证据锚点。",
                ],
            }
        )

    return plan


def build_agentic_response_metadata(
    state: Dict[str, Any], image_urls: Sequence[str] | None = None
) -> Dict[str, Any]:
    answer_plan = state.get("answer_plan") or {}
    coverage = state.get("evidence_coverage_summary") or {}
    rescue_plan = state.get("rescue_plan") or {}
    context_expansion = state.get("context_expansion_summary") or {}
    return {
        "query_type": state.get("query_type") or "general",
        "query_complexity": state.get("query_complexity") or "simple",
        "query_complexity_reason": state.get("query_complexity_reason") or "",
        "graph_preferred": bool(state.get("graph_preferred", False)),
        "router_decision": state.get("router_decision") or "default_path",
        "router_deep_search_enabled": bool(
            state.get("router_deep_search_enabled", False)
        ),
        "crag_router_enabled": bool(state.get("crag_router_enabled", False)),
        "grounded_mode": bool(state.get("grounded_mode", False)),
        "query_focus_terms": state.get("query_focus_terms") or [],
        "query_route_reason": state.get("query_route_reason") or "",
        "retrieval_plan": state.get("retrieval_plan") or {},
        "kg_query_summary": state.get("kg_query_summary") or {},
        "evidence_coverage_summary": coverage,
        "rescue_plan": rescue_plan,
        "context_expansion_summary": context_expansion,
        "answer_plan": answer_plan,
        "clarification_reason": coverage.get("clarification_reason") or state.get("clarification_reason") or "",
        "image_urls": list(image_urls or []),
        "agentic_features": get_agentic_features(state),
        "cache_summary": state.get("cache_summary") or {},
    }
