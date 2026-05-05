import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from app.conf.query_threshold_config import query_threshold_config
from app.utils.anchor_context_utils import build_target_coverage

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


def _state_override(state: Dict[str, Any], key: str, default: Any = None) -> Any:
    for container in (
        state,
        state.get("route_overrides") or {},
        state.get("evaluation_overrides") or {},
    ):
        if isinstance(container, dict) and key in container:
            return container.get(key)
    return default


def _rescue_min_coverage_score(state: Dict[str, Any]) -> float:
    value = _state_override(
        state,
        "rescue_min_coverage_score",
        query_threshold_config.evidence_min_coverage_score,
    )
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return query_threshold_config.evidence_min_coverage_score


def _rescue_min_docs(state: Dict[str, Any]) -> int:
    value = _state_override(
        state,
        "rescue_min_docs",
        query_threshold_config.evidence_min_docs,
    )
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return query_threshold_config.evidence_min_docs


def _structured_answer_high_risk_only(state: Dict[str, Any]) -> bool:
    return bool(_state_override(state, "structured_answer_high_risk_only", False))


def _force_target_coverage(state: Dict[str, Any]) -> bool:
    return bool(_state_override(state, "rescue_force_target_coverage", False))


def _query_family(state: Dict[str, Any]) -> str:
    return str(state.get("router_query_family") or "general")


def _is_fast_section_family(query_family: str) -> bool:
    return query_family in {"section_summary", "section_lookup", "procedure_lookup"}


def _is_quality_rescue_path(query_type: str, query_family: str) -> bool:
    if query_family == "comparison":
        return True
    if query_family == "multi_hop_relation":
        return True
    if _is_fast_section_family(query_family):
        return False
    return query_type in {"comparison", "relation", "constraint", "explain"}


def _doc_evidence_id(doc: Dict[str, Any], index: int) -> str:
    for key in ("chunk_id", "doc_id", "id", "url"):
        value = _clean_text(doc.get(key))
        if value:
            return value
    return f"doc_{index + 1}"


def _doc_evidence_title(doc: Dict[str, Any]) -> str:
    for key in ("section_path", "title", "document_title", "file_title"):
        value = _clean_text(doc.get(key))
        if value:
            return value
    return ""


def _supporting_docs_for_terms(
    terms: Sequence[str],
    docs: Sequence[Dict[str, Any]],
    doc_blobs: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for term in terms or []:
        normalized = _normalize_term(term)
        evidence = []
        if normalized:
            for index, doc in enumerate(docs or []):
                blob = doc_blobs[index] if index < len(doc_blobs) else ""
                if normalized not in blob:
                    continue
                evidence.append(
                    {
                        "id": _doc_evidence_id(doc, index),
                        "title": _doc_evidence_title(doc),
                        "source": _clean_text(doc.get("source") or "local"),
                    }
                )
        rows.append(
            {
                "term": normalized,
                "covered": bool(evidence),
                "evidence": evidence[:3],
            }
        )
    return rows


def _supporting_docs_for_sub_queries(
    sub_queries: Sequence[str],
    docs: Sequence[Dict[str, Any]],
    doc_blobs: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sub_query in sub_queries or []:
        query_text = _clean_text(sub_query)
        sub_terms = _extract_terms(query_text) or [_normalize_term(query_text)]
        hit_terms = [
            term
            for term in sub_terms
            if term and any(term in blob for blob in doc_blobs)
        ]
        evidence = []
        for index, doc in enumerate(docs or []):
            blob = doc_blobs[index] if index < len(doc_blobs) else ""
            if not any(term and term in blob for term in sub_terms):
                continue
            evidence.append(
                {
                    "id": _doc_evidence_id(doc, index),
                    "title": _doc_evidence_title(doc),
                    "source": _clean_text(doc.get("source") or "local"),
                }
            )
        coverage_ratio = len(hit_terms) / len(sub_terms) if sub_terms else 0.0
        rows.append(
            {
                "query": query_text,
                "terms": sub_terms,
                "covered_terms": hit_terms,
                "coverage_ratio": round(coverage_ratio, 4),
                "covered": bool(sub_terms) and coverage_ratio >= 0.5,
                "evidence": evidence[:3],
            }
        )
    return rows


def _target_support_rows(target_coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in target_coverage.get("targets") or []:
        chunks = item.get("chunks") or []
        rows.append(
            {
                "target": _clean_text(item.get("target")),
                "covered": bool(item.get("covered")),
                "evidence": [
                    {
                        "id": _clean_text(chunk.get("chunk_id")),
                        "title": _clean_text(chunk.get("title")),
                        "source": "target_coverage",
                    }
                    for chunk in chunks[:3]
                    if isinstance(chunk, dict)
                ],
            }
        )
    return rows


def _coverage_gap(
    dimension: str,
    missing: Sequence[Any],
    severity: str,
    reason: str,
    suggested_action: str,
) -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "missing": [_clean_text(item) for item in missing or [] if _clean_text(item)],
        "severity": severity,
        "reason": reason,
        "suggested_action": suggested_action,
    }


def _coverage_answerability(
    doc_count: int,
    needs_rescue: bool,
    clarification_required: bool,
    high_gap_count: int,
) -> str:
    if clarification_required:
        return "needs_clarification"
    if doc_count <= 0:
        return "unanswerable_no_evidence"
    if needs_rescue or high_gap_count > 0:
        return "partial"
    return "answerable"


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
    answerable = bool(state.get("answerable", True))
    has_product_reference = bool(item_names) or _question_has_explicit_product_reference(question)
    constraint_markers = (
        "限制",
        "安全",
        "注意",
        "条件",
        "要求",
        "适配",
        "不能",
        "禁止",
        "必须",
        "充电",
        "约束",
    )
    has_constraint_marker = any(marker in question for marker in constraint_markers)

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

    if (
        query_type == "constraint"
        and not (answerable and has_product_reference and has_constraint_marker)
        and (not has_product_reference or not has_constraint_marker or len(focus_terms) < 1)
    ):
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

    if query_type == "explain" and len(focus_terms) < 1 and not (answerable and has_product_reference and question):
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
    evidence_docs = [doc for doc in reranked_docs if isinstance(doc, dict)]
    question = _clean_text(state.get("rewritten_query") or state.get("original_query"))
    query_type = str(state.get("query_type") or "general")
    query_family = _query_family(state)
    focus_terms = [
        _normalize_term(term)
        for term in (state.get("query_focus_terms") or _extract_terms(question))
        if _normalize_term(term)
    ]
    item_names = [
        _normalize_term(name) for name in (state.get("item_names") or []) if _normalize_term(name)
    ]
    sub_queries = [_clean_text(item) for item in (state.get("sub_queries") or []) if _clean_text(item)]
    query_anchor_targets = [
        _clean_text(target)
        for target in (state.get("query_anchor_targets") or [])
        if _clean_text(target)
    ]
    doc_blobs = [_doc_blob(doc) for doc in evidence_docs]
    source_counts = Counter(
        _clean_text(doc.get("source") or "local") for doc in evidence_docs
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
    target_coverage = build_target_coverage(evidence_docs, query_anchor_targets)
    coverage_target_ratio = target_coverage.get("coverage_rate", 1.0)
    doc_ratio = min(
        doc_count / max(_rescue_min_docs(state), 1),
        1.0,
    )
    coverage_score = round(
        coverage_terms_ratio * 0.30
        + coverage_item_ratio * 0.15
        + coverage_sub_ratio * 0.20
        + doc_ratio * 0.20,
        4,
    )
    coverage_score = round(
        coverage_score + float(coverage_target_ratio or 0.0) * 0.15,
        4,
    )

    clarification = build_clarification_request(state)
    target_coverage_required = _force_target_coverage(state) or query_type == "comparison"
    missing_targets = target_coverage.get("missing_targets") or []
    needs_rescue = (
        coverage_score < _rescue_min_coverage_score(state)
        or doc_count < _rescue_min_docs(state)
        or bool(missing_sub_queries)
        or bool(missing_items)
        or (target_coverage_required and bool(missing_targets))
    )
    coverage_gaps: List[Dict[str, Any]] = []
    if doc_count <= 0:
        coverage_gaps.append(
            _coverage_gap(
                "documents",
                ["no_retrieved_documents"],
                "high",
                "no_evidence_available",
                "expand_retrieval_sources",
            )
        )
    elif doc_count < _rescue_min_docs(state):
        coverage_gaps.append(
            _coverage_gap(
                "documents",
                [f"{doc_count}/{_rescue_min_docs(state)}"],
                "medium",
                "too_few_evidence_documents",
                "expand_lexical_retrieval",
            )
        )
    if missing_targets:
        coverage_gaps.append(
            _coverage_gap(
                "targets",
                missing_targets,
                "high" if target_coverage_required else "medium",
                "missing_anchor_targets",
                "cover_missing_targets",
            )
        )
    if missing_sub_queries:
        coverage_gaps.append(
            _coverage_gap(
                "sub_queries",
                missing_sub_queries,
                "high",
                "missing_sub_query_evidence",
                "cover_missing_sub_queries",
            )
        )
    if missing_items:
        coverage_gaps.append(
            _coverage_gap(
                "item_names",
                missing_items,
                "high",
                "missing_item_evidence",
                "broaden_or_drop_item_filter",
            )
        )
    if missing_terms:
        all_focus_missing = bool(focus_terms) and len(missing_terms) == len(focus_terms)
        coverage_gaps.append(
            _coverage_gap(
                "focus_terms",
                missing_terms,
                "high" if all_focus_missing and _is_quality_rescue_path(query_type, query_family) else "medium",
                "missing_focus_term_evidence",
                "expand_lexical_retrieval",
            )
        )
    if coverage_score < _rescue_min_coverage_score(state) and not coverage_gaps:
        coverage_gaps.append(
            _coverage_gap(
                "coverage_score",
                [coverage_score],
                "medium",
                "coverage_score_below_threshold",
                "query_rewrite_retry",
            )
        )
    high_gap_count = len(
        [gap for gap in coverage_gaps if gap.get("severity") == "high"]
    )
    clarification_required = bool(clarification.get("required"))
    if clarification_required:
        coverage_gaps.append(
            _coverage_gap(
                "query",
                [clarification.get("reason") or "clarification_required"],
                "high",
                "query_missing_required_constraints",
                "ask_clarification",
            )
        )
        high_gap_count += 1
    if doc_count <= 0 or clarification_required or high_gap_count > 0:
        coverage_risk_level = "high"
    elif needs_rescue or coverage_gaps:
        coverage_risk_level = "medium"
    else:
        coverage_risk_level = "low"
    answerability = _coverage_answerability(
        doc_count,
        needs_rescue,
        clarification_required,
        high_gap_count,
    )
    coverage_support = {
        "focus_terms": _supporting_docs_for_terms(focus_terms, evidence_docs, doc_blobs),
        "item_names": _supporting_docs_for_terms(item_names, evidence_docs, doc_blobs),
        "sub_queries": _supporting_docs_for_sub_queries(
            sub_queries,
            evidence_docs,
            doc_blobs,
        ),
        "targets": _target_support_rows(target_coverage),
    }

    return {
        "coverage_version": "v2",
        "query_type": query_type,
        "router_query_family": query_family,
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
        "coverage_target_ratio": round(float(coverage_target_ratio or 0.0), 4),
        "target_coverage": target_coverage,
        "missing_targets": missing_targets,
        "coverage_score": coverage_score,
        "coverage_risk_level": coverage_risk_level,
        "coverage_gaps": coverage_gaps,
        "coverage_support": coverage_support,
        "answerability": answerability,
        "needs_rescue": needs_rescue,
        "clarification_required": clarification_required,
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
    query_family = _query_family(state)
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
    target_coverage = coverage.get("target_coverage") or state.get("target_coverage") or {}
    missing_targets = list(coverage.get("missing_targets") or target_coverage.get("missing_targets") or [])
    missing_sub_queries = list(coverage.get("missing_sub_queries") or [])
    coverage_score = float(coverage.get("coverage_score") or 0.0)
    doc_count = int(coverage.get("doc_count") or 0)
    rescue_min_score = _rescue_min_coverage_score(state)
    min_docs = _rescue_min_docs(state)
    quality_rescue_path = _is_quality_rescue_path(query_type, query_family)
    severe_shortage = doc_count <= 0 or coverage_score < max(0.45, rescue_min_score - 0.2)

    if quality_rescue_path:
        route_overrides["force_query_type"] = query_type
        route_overrides["force_graph_preferred"] = query_type in {"relation", "explain"}

    if missing_targets:
        plan_overrides["run_anchor"] = True
        plan_overrides["run_bm25"] = True
        plan_overrides["run_embedding"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("cover_missing_targets")

    if missing_sub_queries:
        plan_overrides["run_anchor"] = True
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("cover_missing_sub_queries")

    if doc_count < min_docs and "expand_lexical_retrieval" not in steps:
        plan_overrides["run_anchor"] = True
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("expand_lexical_retrieval")

    if item_names and coverage.get("missing_item_names") == [name.lower() for name in item_names]:
        route_overrides["drop_item_names"] = True
        plan_overrides["run_web"] = query_type == "general"
        steps.append("drop_item_filter")

    if (
        quality_rescue_path
        and query_type in {"relation", "explain"}
        and (
            coverage_score < rescue_min_score
            or doc_count < min_docs
            or kg_summary.get("result_count", 0) <= 1
            or bool(missing_sub_queries)
            or bool(missing_targets)
        )
    ):
        plan_overrides["run_kg"] = True
        plan_overrides["graph_limit"] = max(
            int(plan_overrides.get("graph_limit") or retrieval_plan.get("graph_limit", 8) or 8),
            12,
        )
        plan_overrides["kg_weight_multiplier"] = max(
            float(plan_overrides.get("kg_weight_multiplier") or retrieval_plan.get("kg_weight_multiplier", 1.0) or 1.0),
            2.2,
        )
        steps.append("widen_graph_context")

    if severe_shortage and quality_rescue_path:
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
        plan_overrides["run_hyde"] = True
        route_overrides["bm25_enabled"] = True
        steps.append("enable_hyde_rescue")
    elif severe_shortage and not steps:
        plan_overrides["run_embedding"] = True
        plan_overrides["run_bm25"] = True
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
        "coverage_score": coverage_score,
        "missing_focus_terms": coverage.get("missing_focus_terms") or [],
        "missing_sub_queries": missing_sub_queries,
        "missing_targets": missing_targets,
        "router_query_family": query_family,
    }


def build_answer_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    query_type = str(state.get("query_type") or "general")
    query_family = _query_family(state)
    structured_enabled = is_agentic_feature_enabled(state, "structured_answer")
    item_names = [_clean_text(name) for name in (state.get("item_names") or []) if _clean_text(name)]
    coverage = state.get("evidence_coverage_summary") or {}
    kg_summary = state.get("kg_query_summary") or {}
    reranked_docs = state.get("reranked_docs") or []
    retrieval_grade = str(state.get("retrieval_grade") or "")
    coverage_score = coverage.get("coverage_score")
    coverage_answerability = str(coverage.get("answerability") or "")
    coverage_risk_level = str(coverage.get("coverage_risk_level") or "")
    coverage_incomplete = (
        bool(coverage.get("needs_rescue"))
        or retrieval_grade == "insufficient"
        or coverage_risk_level in {"medium", "high"}
        or coverage_answerability in {
            "partial",
            "unanswerable_no_evidence",
            "needs_clarification",
        }
    )
    query_anchor_targets = [
        _clean_text(target)
        for target in (state.get("query_anchor_targets") or [])
        if _clean_text(target)
    ]
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
    high_risk_structured = (
        query_family in {"comparison", "multi_hop_relation"}
        or (
            query_type in {"comparison", "relation", "constraint", "explain"}
            and not _is_fast_section_family(query_family)
        )
    )
    if not _structured_answer_high_risk_only(state):
        high_risk_structured = query_type in {
            "navigation",
            "comparison",
            "relation",
            "constraint",
            "explain",
        }
    must_cover: List[str] = []
    for value in item_names + query_anchor_targets + list(coverage.get("covered_focus_terms") or [])[:4]:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in must_cover:
            must_cover.append(cleaned)

    plan = {
        "query_type": query_type,
        "router_query_family": query_family,
        "structured_output": structured_enabled and high_risk_structured,
        "response_format": "paragraph",
        "sections": ["直接回答", "补充说明"],
        "style_instructions": ["优先基于证据给出直接结论。"],
        "must_cover": must_cover,
        "top_titles": top_titles,
        "top_graph_facts": top_graph_facts,
        "coverage_score": coverage_score,
        "graph_summary": kg_summary,
        "risk_level": "high" if coverage_incomplete else "normal",
        "answerability": coverage_answerability
        or ("partial_or_unknown" if coverage_incomplete else "answerable"),
        "must_refuse_missing_parts": coverage_incomplete,
    }

    if coverage_incomplete:
        plan["style_instructions"] = list(plan.get("style_instructions") or []) + [
            "如果当前证据不足，只回答证据直接支持的部分。",
            "对缺少证据的参数、步骤、条件或结论必须明确说明资料中未提供。",
        ]

    if not structured_enabled or not high_risk_structured:
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

    if coverage_incomplete:
        for instruction in (
            "如果当前证据不足，只回答证据直接支持的部分。",
            "对缺少证据的参数、步骤、条件或结论必须明确说明资料中未提供。",
        ):
            if instruction not in plan.get("style_instructions", []):
                plan["style_instructions"].append(instruction)

    return plan


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_flag(flags: List[str], flag: str) -> None:
    if flag and flag not in flags:
        flags.append(flag)


def _retrieval_source_counts(state: Dict[str, Any]) -> Dict[str, int]:
    return {
        "embedding": len(state.get("embedding_chunks") or []),
        "hyde": len(state.get("hyde_embedding_chunks") or []),
        "bm25": len(state.get("bm25_chunks") or []),
        "anchor": len(state.get("anchor_chunks") or []),
        "kg": len(state.get("kg_chunks") or []),
        "web": len(state.get("web_search_docs") or []),
        "rrf": len(state.get("rrf_chunks") or []),
        "reranked": len(state.get("reranked_docs") or []),
    }


def _has_judge_failure_reason(reason: Any) -> bool:
    text = _clean_text(reason).lower()
    return any(marker in text for marker in ("failed", "失败", "error", "exception", "降级通过"))


def build_quality_trace(state: Dict[str, Any]) -> Dict[str, Any]:
    coverage = state.get("evidence_coverage_summary") or {}
    rerank_diagnostics = state.get("rerank_diagnostics") or {}
    rescue_plan = state.get("rescue_plan") or {}
    answer_plan = state.get("answer_plan") or {}
    claim_summary = state.get("claim_verification_summary") or {}
    retrieval_grade = str(state.get("retrieval_grade") or "")
    retrieval_strategy = str(state.get("retrieval_grader_strategy") or "")
    retrieval_skip_reason = str(state.get("retrieval_judge_skipped_reason") or "")
    hallucination_strategy = str(state.get("hallucination_check_strategy") or "")
    hallucination_skip_reason = str(state.get("hallucination_judge_skipped_reason") or "")
    retry_count = int(state.get("retry_count") or 0)
    hallucination_retry_count = int(state.get("hallucination_retry_count") or 0)
    coverage_score = _safe_float(coverage.get("coverage_score"), 1.0)
    min_coverage_score = _rescue_min_coverage_score(state)
    coverage_risk_level = str(coverage.get("coverage_risk_level") or "").lower()
    coverage_answerability = str(coverage.get("answerability") or "").lower()
    target_coverage = state.get("target_coverage") or {}
    missing_targets = coverage.get("missing_targets") or target_coverage.get("missing_targets") or []
    missing_sub_queries = coverage.get("missing_sub_queries") or []
    missing_items = coverage.get("missing_item_names") or []
    missing_focus_terms = coverage.get("missing_focus_terms") or []
    coverage_gaps = coverage.get("coverage_gaps") or []
    flags: List[str] = []

    if bool(state.get("need_rag", False)) and not state.get("reranked_docs"):
        _append_flag(flags, "empty_retrieval")
    if retrieval_grade in {"insufficient", "partial", "irrelevant", "conflicting", "unanswerable"}:
        _append_flag(flags, "low_retrieval_confidence")
    if retrieval_grade == "retry":
        _append_flag(flags, "crag_retry_pending")
    if retry_count > 0:
        _append_flag(flags, "crag_retry_used")
    if retrieval_grade == "insufficient" and retry_count >= int(query_threshold_config.crag_max_retries):
        _append_flag(flags, "crag_retry_exhausted")
    if bool(coverage.get("needs_rescue")) or coverage_score < min_coverage_score:
        _append_flag(flags, "coverage_incomplete")
    if coverage_risk_level == "high":
        _append_flag(flags, "coverage_high_risk")
    elif coverage_risk_level == "medium":
        _append_flag(flags, "coverage_medium_risk")
    if coverage_answerability in {"unanswerable_no_evidence", "needs_clarification"}:
        _append_flag(flags, "coverage_unanswerable")
    elif coverage_answerability == "partial":
        _append_flag(flags, "coverage_partial")
    if missing_targets:
        _append_flag(flags, "target_coverage_incomplete")
    if missing_sub_queries:
        _append_flag(flags, "subquery_coverage_incomplete")
    if missing_items:
        _append_flag(flags, "item_coverage_incomplete")
    if missing_focus_terms:
        _append_flag(flags, "focus_term_coverage_incomplete")
    if rescue_plan.get("action") == "retry":
        _append_flag(flags, "retrieval_rescue_triggered")
    if rescue_plan.get("action") == "clarify":
        _append_flag(flags, "clarification_required")
    if (
        bool(rerank_diagnostics.get("fallback"))
        or bool(rerank_diagnostics.get("heuristic"))
        or bool(rerank_diagnostics.get("error"))
        or str(rerank_diagnostics.get("status") or "").lower().startswith("error")
    ):
        _append_flag(flags, "rerank_untrusted")
    if retrieval_strategy == "failed" or _has_judge_failure_reason(retrieval_skip_reason):
        _append_flag(flags, "retrieval_judge_failed")
    elif retrieval_skip_reason:
        _append_flag(flags, "retrieval_judge_skipped")
    if hallucination_strategy == "failed" or _has_judge_failure_reason(hallucination_skip_reason):
        _append_flag(flags, "hallucination_judge_failed")
    elif hallucination_skip_reason:
        _append_flag(flags, "hallucination_judge_skipped")
    if not bool(state.get("hallucination_check_passed", True)):
        _append_flag(flags, "hallucination_detected")
    if hallucination_retry_count > 0:
        _append_flag(flags, "hallucination_retry_used")
    if bool(state.get("is_stream")) and bool(state.get("need_rag", False)) and not hallucination_strategy:
        _append_flag(flags, "streaming_unverified_answer")
    if bool(state.get("answer")) and (
        retrieval_grade == "insufficient" or bool(coverage.get("needs_rescue"))
    ):
        _append_flag(flags, "answer_generated_with_insufficient_evidence")
    if bool(answer_plan.get("must_refuse_missing_parts")):
        _append_flag(flags, "answer_requires_partial_refusal")
    if bool(state.get("crag_safe_generation_required")):
        _append_flag(flags, "crag_safe_generation_required")
    if bool(state.get("citation_required")):
        _append_flag(flags, "citation_required")
    if int(claim_summary.get("unsupported_claim_count") or 0) > 0:
        _append_flag(flags, "unsupported_claims_detected")
    if int(claim_summary.get("weak_claim_count") or 0) > 0:
        _append_flag(flags, "weakly_supported_claims_detected")
    claim_risk_level = str(claim_summary.get("risk_level") or "").lower()
    if claim_risk_level == "high":
        _append_flag(flags, "claim_verification_high_risk")
    elif claim_risk_level == "medium":
        _append_flag(flags, "claim_verification_medium_risk")

    severe_flags = {
        "empty_retrieval",
        "low_retrieval_confidence",
        "crag_retry_exhausted",
        "retrieval_judge_failed",
        "hallucination_judge_failed",
        "hallucination_detected",
        "answer_generated_with_insufficient_evidence",
        "coverage_high_risk",
        "coverage_unanswerable",
        "unsupported_claims_detected",
        "claim_verification_high_risk",
    }
    medium_flags = {
        "coverage_incomplete",
        "coverage_medium_risk",
        "coverage_partial",
        "target_coverage_incomplete",
        "subquery_coverage_incomplete",
        "item_coverage_incomplete",
        "focus_term_coverage_incomplete",
        "weakly_supported_claims_detected",
        "claim_verification_medium_risk",
        "rerank_untrusted",
        "streaming_unverified_answer",
    }
    if any(flag in severe_flags for flag in flags):
        risk_level = "high"
    elif any(flag in medium_flags for flag in flags):
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "flags": flags,
        "retrieval": {
            "grade": retrieval_grade,
            "retry_count": retry_count,
            "strategy": retrieval_strategy,
            "judge_skipped_reason": retrieval_skip_reason,
            "rescue_action": rescue_plan.get("action") or "",
            "rescue_reason": rescue_plan.get("reason") or "",
            "rescue_steps": rescue_plan.get("steps") or [],
            "source_counts": _retrieval_source_counts(state),
            "rerank_diagnostics": rerank_diagnostics,
        },
        "coverage": {
            "score": coverage_score,
            "min_score": min_coverage_score,
            "needs_rescue": bool(coverage.get("needs_rescue")),
            "risk_level": coverage_risk_level,
            "answerability": coverage_answerability,
            "gaps": coverage_gaps,
            "missing_targets": missing_targets,
            "missing_sub_queries": missing_sub_queries,
            "missing_item_names": missing_items,
            "missing_focus_terms": missing_focus_terms,
        },
        "generation": {
            "grounded_mode": bool(state.get("grounded_mode", False)),
            "answer_plan_risk_level": answer_plan.get("risk_level") or "",
            "answerability": answer_plan.get("answerability") or "",
            "must_refuse_missing_parts": bool(answer_plan.get("must_refuse_missing_parts")),
            "crag_safe_generation_required": bool(
                state.get("crag_safe_generation_required")
            ),
            "crag_safe_reason": state.get("crag_safe_reason") or "",
            "citation_required": bool(state.get("citation_required", False)),
            "citation_targets": state.get("citation_targets") or [],
            "final_context_doc_count": int(state.get("final_context_doc_count") or 0),
            "final_context_chars": int(state.get("final_context_chars") or 0),
        },
        "verification": {
            "hallucination_check_passed": bool(
                state.get("hallucination_check_passed", True)
            ),
            "hallucination_check_strategy": hallucination_strategy,
            "hallucination_judge_skipped_reason": hallucination_skip_reason,
            "hallucination_retry_count": hallucination_retry_count,
            "feedback": state.get("hallucination_feedback") or "",
            "claim_verification": {
                "enabled": bool(claim_summary.get("enabled", False)),
                "strategy": claim_summary.get("strategy") or "",
                "risk_level": claim_summary.get("risk_level") or "",
                "claim_count": int(claim_summary.get("claim_count") or 0),
                "unsupported_claim_count": int(
                    claim_summary.get("unsupported_claim_count") or 0
                ),
                "weak_claim_count": int(claim_summary.get("weak_claim_count") or 0),
                "support_rate": _safe_float(claim_summary.get("support_rate"), 1.0),
            },
        },
    }


def build_agentic_response_metadata(
    state: Dict[str, Any], image_urls: Sequence[str] | None = None
) -> Dict[str, Any]:
    answer_plan = state.get("answer_plan") or {}
    coverage = state.get("evidence_coverage_summary") or {}
    rescue_plan = state.get("rescue_plan") or {}
    context_expansion = state.get("context_expansion_summary") or {}
    rerank_diagnostics = state.get("rerank_diagnostics") or {}
    quality_trace = build_quality_trace(state)
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
        "query_anchor_targets": state.get("query_anchor_targets") or [],
        "router_query_family": state.get("router_query_family") or "general",
        "query_route_reason": state.get("query_route_reason") or "",
        "retrieval_plan": state.get("retrieval_plan") or {},
        "anchor_hits": state.get("anchor_hits") or [],
        "target_coverage": state.get("target_coverage") or {},
        "evidence_pack_summary": state.get("evidence_pack_summary") or {},
        "context_budget_chars": state.get("context_budget_chars") or 0,
        "kg_query_summary": state.get("kg_query_summary") or {},
        "evidence_coverage_summary": coverage,
        "rescue_plan": rescue_plan,
        "context_expansion_summary": context_expansion,
        "final_context_summary": state.get("final_context_summary") or {},
        "final_context_ids": state.get("final_context_ids") or [],
        "final_context_titles": state.get("final_context_titles") or [],
        "final_context_chars": int(state.get("final_context_chars") or 0),
        "final_context_doc_count": int(state.get("final_context_doc_count") or 0),
        "rerank_diagnostics": rerank_diagnostics,
        "judge_skipped_reason": state.get("judge_skipped_reason") or "",
        "retrieval_judge_skipped_reason": state.get("retrieval_judge_skipped_reason")
        or "",
        "hallucination_judge_skipped_reason": state.get(
            "hallucination_judge_skipped_reason"
        )
        or "",
        "retrieval_grader_strategy": state.get("retrieval_grader_strategy") or "",
        "hallucination_check_strategy": state.get("hallucination_check_strategy") or "",
        "claim_verification_summary": state.get("claim_verification_summary") or {},
        "answer_plan": answer_plan,
        "crag_safe_generation_required": bool(
            state.get("crag_safe_generation_required", False)
        ),
        "crag_safe_reason": state.get("crag_safe_reason") or "",
        "citation_required": bool(state.get("citation_required", False)),
        "citation_targets": state.get("citation_targets") or [],
        "clarification_reason": coverage.get("clarification_reason") or state.get("clarification_reason") or "",
        "image_urls": list(image_urls or []),
        "agentic_features": get_agentic_features(state),
        "quality_trace": quality_trace,
        "quality_flags": quality_trace.get("flags") or [],
        "quality_risk_level": quality_trace.get("risk_level") or "low",
        "cache_summary": state.get("cache_summary") or {},
    }
