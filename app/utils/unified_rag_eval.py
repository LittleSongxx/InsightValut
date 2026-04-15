import argparse
import copy
import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from langchain_core.embeddings import Embeddings
except ImportError:

    class Embeddings:
        pass


from app.clients.milvus_utils import get_milvus_client, query_chunks_by_filter
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.agentic_utils import DEFAULT_AGENTIC_FEATURES
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.graph_query_utils import GRAPH_PREFERRED_QUERY_TYPES
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.retrieval_utils import build_item_name_filter_expr
from app.utils.bm25_utils import rank_documents_bm25, tokenize_text
from app.utils.path_util import PROJECT_ROOT
from app.utils.perf_tracker import perf_finish, perf_start
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
    "item_name",
    "title",
    "parent_title",
    "part",
    "file_title",
    "content",
]
GROUND_TRUTH_SEARCH_LIMIT = 80
DEFAULT_VARIANTS = [
    "baseline_rag",
    "bm25_hybrid",
    "hyde_hybrid",
    "kg_hybrid",
    "neo4j_graph_first",
    "final_system",
    "agentic_context_expansion",
    "agentic_rescue_system",
    "agentic_enhanced_system",
]

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _agentic_features(**overrides: bool) -> Dict[str, bool]:
    features = {key: False for key in DEFAULT_AGENTIC_FEATURES.keys()}
    features.update(overrides)
    return features


def _ensure_ragas_ready() -> None:
    if _RAGAS_IMPORT_ERROR is not None:
        raise ImportError(
            "RAGAS 未安装，请先安装项目依赖后再运行统一评测。"
        ) from _RAGAS_IMPORT_ERROR


VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline_rag": {
        "description": "最原始基线：仅 embedding 检索 + rerank + answer",
        "technique": "Baseline RAG",
        "use_case_query_type": False,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_query_type": "general",
            "force_graph_preferred": False,
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
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "force_graph_preferred": False,
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
    "neo4j_graph_first": {
        "description": "按题型启用 graph-first，但关闭其它额外增强器以隔离图路由收益",
        "technique": "Neo4j Graph-First",
        "compare_to": "kg_hybrid",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "route_reason": "eval_neo4j_graph_first",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {
                "run_embedding": True,
                "run_kg": True,
                "run_bm25": False,
                "run_hyde": False,
                "run_web": False,
                "bm25_weight_multiplier": 0.0,
                "hyde_weight_multiplier": 0.0,
            },
        },
    },
    "final_system": {
        "description": "当前最终版统一系统：保留原有题型路由与图优先策略，关闭新增 Agentic 增强",
        "technique": "Final Unified RAG",
        "compare_to": "neo4j_graph_first",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "route_reason": "eval_final_system",
            "agentic_features": _agentic_features(),
        },
    },
    "final_system_with_bm25": {
        "description": "在关闭新增 Agentic 增强的最终版系统上显式开启 BM25，用于量化 BM25 增益/代价",
        "technique": "BM25 on Final System",
        "compare_to": "final_system",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "bm25_enabled": True,
            "route_reason": "eval_final_system_with_bm25",
            "agentic_features": _agentic_features(),
            "retrieval_plan_overrides": {"run_bm25": True},
        },
    },
    "agentic_context_expansion": {
        "description": "在原最终版上只开启命中上下文扩展，用于量化章节/相邻步骤补文效果",
        "technique": "Agentic Context Expansion",
        "compare_to": "final_system",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "route_reason": "eval_agentic_context_expansion",
            "agentic_features": _agentic_features(context_expansion=True),
        },
    },
    "agentic_rescue_system": {
        "description": "在原最终版上开启子问题级路由、证据覆盖和检索补救，用于量化复杂问题的自适应检索收益",
        "technique": "Agentic Retrieval Rescue",
        "compare_to": "final_system",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "route_reason": "eval_agentic_rescue_system",
            "agentic_features": _agentic_features(
                subquery_routing=True,
                evidence_coverage=True,
                retrieval_rescue=True,
                clarification_guard=True,
            ),
        },
    },
    "agentic_enhanced_system": {
        "description": "完整 Agentic 增强系统：子问题级路由、上下文扩展、证据覆盖、检索补救和结构化回答全部开启",
        "technique": "Agentic Enhanced RAG",
        "compare_to": "final_system",
        "use_case_query_type": True,
        "evaluation_overrides": {
            "force_need_rag": True,
            "route_reason": "eval_agentic_enhanced_system",
            "agentic_features": dict(DEFAULT_AGENTIC_FEATURES),
        },
    },
}


def get_variant_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for name, config in VARIANTS.items():
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


def _fetch_ground_truth_candidates(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = get_milvus_client()
    collection_name = _chunks_collection_name()
    if not client or not collection_name:
        return []

    item_filter = build_item_name_filter_expr(_item_names(case))
    docs = query_chunks_by_filter(
        client=client,
        collection_name=collection_name,
        filter_expr=item_filter,
        output_fields=list(GROUND_TRUTH_OUTPUT_FIELDS),
        limit=GROUND_TRUTH_SEARCH_LIMIT,
    )
    if docs or item_filter:
        return docs

    return query_chunks_by_filter(
        client=client,
        collection_name=collection_name,
        filter_expr="",
        output_fields=list(GROUND_TRUTH_OUTPUT_FIELDS),
        limit=GROUND_TRUTH_SEARCH_LIMIT,
    )


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
    }
    if not eligible:
        result["reason"] = "no_retrieval_ground_truth"
        return result

    candidates = _fetch_ground_truth_candidates(case)
    result["candidate_count"] = len(candidates)
    candidate_ids = {str(doc.get("chunk_id")) for doc in candidates if doc.get("chunk_id") is not None}

    if declared_ids:
        matched_ids = [chunk_id for chunk_id in declared_ids if chunk_id in candidate_ids]
        if matched_ids:
            result["resolved_ids"] = matched_ids
            result["source"] = "declared_ids"
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
    for doc, score in ranked_docs:
        if score <= 0:
            continue
        doc_text = _ground_truth_doc_text(doc)
        overlap = _token_overlap_ratio(reference_answer or _query(case), doc_text)
        chunk_id = doc.get("chunk_id")
        if chunk_id is None:
            continue
        if overlap >= 0.2 or (score >= top_score * 0.92 and overlap >= 0.08):
            selected_ids.append(str(chunk_id))
        if len(selected_ids) >= 3:
            break

    if not selected_ids:
        result["reason"] = "reference_match_too_weak"
        result["source"] = "unresolved"
        return result

    result["resolved_ids"] = selected_ids
    result["source"] = "reference_answer_bm25"
    if result["declared_ids_stale"]:
        result["reason"] = "declared_ids_stale"
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


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
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
    index = min(int(len(nums) * ratio), len(nums) - 1)
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
    if config.get("use_case_query_type"):
        query_type = _query_type(case)
        overrides["force_query_type"] = query_type
        overrides.setdefault(
            "force_graph_preferred", query_type in GRAPH_PREFERRED_QUERY_TYPES
        )
    item_names = _item_names(case)
    if item_names:
        overrides.setdefault("force_item_names", item_names)
    return overrides


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
        chunk_id = doc.get("chunk_id") or doc.get("doc_id") or doc.get("id")
        if chunk_id is None:
            continue
        value = str(chunk_id).strip()
        if value:
            chunk_ids.append(value)
    return chunk_ids


def _hit_at_k(
    predicted_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if any(chunk_id in relevant for chunk_id in predicted_ids[:k]) else 0.0


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


def _run_single_case(case: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    case_id = str(case.get("case_id") or case.get("id") or uuid.uuid4().hex[:8])
    session_id = f"eval-{variant_name}-{case_id}-{uuid.uuid4().hex[:8]}"
    query = _query(case)
    declared_relevant_ids = _relevant_ids(case)
    resolved_relevant_ids = _resolved_relevant_ids(case)
    initial_state = {
        "session_id": session_id,
        "original_query": query,
        "is_stream": False,
        "evaluation_mode": True,
        "evaluation_overrides": _build_overrides(variant_name, case),
    }
    final_state: Dict[str, Any] = {}
    error = ""
    perf_start(session_id, query)
    try:
        result = query_app.invoke(initial_state)
        if isinstance(result, dict):
            final_state = result
    except Exception as exc:
        error = str(exc)
    perf_doc = perf_finish(session_id, persist=False) or {}
    clear_task(session_id)
    reranked_docs = final_state.get("reranked_docs") or []
    retrieved_contexts = _extract_contexts(reranked_docs)
    return {
        "case_id": case_id,
        "query": query,
        "query_type": _query_type(case),
        "item_names": _item_names(case),
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
        "graph_preferred": bool(final_state.get("graph_preferred", False)),
        "sub_query_routes": final_state.get("sub_query_routes") or [],
        "sub_query_results": final_state.get("sub_query_results") or [],
        "context_expansion_summary": final_state.get("context_expansion_summary") or {},
        "evidence_coverage_summary": final_state.get("evidence_coverage_summary")
        or {},
        "rescue_plan": final_state.get("rescue_plan") or {},
        "answer_plan": final_state.get("answer_plan") or {},
        "clarification_reason": str(final_state.get("clarification_reason") or ""),
        "agentic_features": (
            initial_state.get("evaluation_overrides", {}).get("agentic_features")
            or {}
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
        "latency_ms": _num(perf_doc.get("total_duration_ms")),
        "first_answer_ms": _num(perf_doc.get("first_answer_ms")),
        "stage_durations_ms": _stage_map(perf_doc),
        "error": error,
    }


def _build_ragas_row(case_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_input": case_result.get("query") or "",
        "response": case_result.get("response") or "",
        "retrieved_contexts": case_result.get("retrieved_contexts") or [],
        "reference": case_result.get("reference_answer") or None,
        "retrieved_context_ids": case_result.get("retrieved_context_ids") or [],
        "reference_context_ids": _resolved_relevant_ids(case_result),
    }


def _metric_column(frame, aliases: Sequence[str]) -> Optional[str]:
    columns = [str(column) for column in getattr(frame, "columns", [])]
    for alias in aliases:
        if alias in columns:
            return alias
    normalized_columns = {
        column: "".join(ch for ch in column.lower() if ch.isalnum()) for column in columns
    }
    for alias in aliases:
        alias_norm = "".join(ch for ch in alias.lower() if ch.isalnum())
        for column in columns:
            if column.startswith(f"{alias}(") or column.startswith(f"{alias}["):
                return column
            normalized = normalized_columns[column]
            if normalized == alias_norm or normalized.startswith(alias_norm):
                return column
    return None


def _run_ragas(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    _ensure_ragas_ready()
    llm = get_llm_client()
    embeddings = BgeM3LangChainEmbeddings()
    rows = [_build_ragas_row(case_result) for case_result in case_results]
    per_case_scores: List[Dict[str, Optional[float]]] = [dict() for _ in case_results]
    summary: Dict[str, Any] = {}
    coverage: Dict[str, int] = {}
    errors: Dict[str, str] = {}

    for metric in RAGAS_METRICS:
        eligible_indices = [
            index for index, row in enumerate(rows) if metric["requires"](row)
        ]
        coverage[metric["name"]] = len(eligible_indices)
        if not eligible_indices:
            continue
        dataset = EvaluationDataset.from_list(
            [rows[index] for index in eligible_indices]
        )
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[metric["factory"]()],
                llm=llm,
                embeddings=embeddings,
            )
            frame = result.to_pandas()
            column = _metric_column(frame, metric["aliases"])
            if not column:
                errors[metric["name"]] = "未在 RAGAS 输出中找到指标列"
                continue
            values = [_num(value) for value in frame[column].tolist()]
            for offset, case_index in enumerate(eligible_indices):
                per_case_scores[case_index][metric["name"]] = values[offset]
            summary[metric["name"]] = _avg(values)
        except Exception as exc:
            errors[metric["name"]] = str(exc)
    return {
        "summary": summary,
        "coverage": coverage,
        "errors": errors,
        "per_case_scores": per_case_scores,
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
    eligible_cases = [
        item for item in case_results if _resolved_relevant_ids(item)
    ]
    coverage: Dict[str, int] = {}
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

    ground_truth_records = [
        (item.get("retrieval_ground_truth") or {})
        for item in case_results
        if (item.get("retrieval_ground_truth") or {}).get("eligible")
    ]
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
    warnings: List[str] = []
    if ground_truth_records and len(eligible_cases) < len(ground_truth_records):
        warnings.append(
            f"检索金标仅对齐了 {len(eligible_cases)}/{len(ground_truth_records)} 个带金标样本；未对齐样本已从 Recall/MRR 统计中剔除。"
        )
    if unresolved_reasons.get("item_name_not_in_corpus"):
        warnings.append(
            f"{unresolved_reasons['item_name_not_in_corpus']} 个样本的 item_names 在当前知识库中不存在，说明评测集与当前入库内容未完全对齐。"
        )
    if stale_declared_cases:
        warnings.append(
            f"{stale_declared_cases} 个样本的 declared chunk_id 已失效，系统已按 reference_answer 自动重映射当前 chunk_id。"
        )

    ground_truth_summary = {
        "eligible_cases": len(ground_truth_records),
        "resolved_cases": len(eligible_cases),
        "unresolved_cases": max(len(ground_truth_records) - len(eligible_cases), 0),
        "source_breakdown": dict(sorted(source_breakdown.items())),
        "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        "stale_declared_cases": stale_declared_cases,
    }
    return summary, coverage, ground_truth_summary, warnings


def _summarize_pipeline(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence_scores = [
        (item.get("evidence_coverage_summary") or {}).get("coverage_score")
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
        "graph_preferred_rate": _avg(
            [1.0 if item.get("graph_preferred") else 0.0 for item in case_results]
        )
        or 0.0,
        "need_rag_rate": _avg(
            [1.0 if item.get("need_rag") else 0.0 for item in case_results]
        )
        or 0.0,
        "clarification_rate": _avg(
            [
                1.0
                if str(item.get("clarification_reason") or "").strip()
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "rescue_retry_rate": _avg(
            [
                1.0
                if (item.get("rescue_plan") or {}).get("action") == "retry"
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "context_expansion_rate": _avg(
            [
                1.0
                if int(
                    ((item.get("context_expansion_summary") or {}).get("expanded_docs") or 0)
                )
                > 0
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "structured_answer_rate": _avg(
            [
                1.0
                if (item.get("answer_plan") or {}).get("structured_output")
                else 0.0
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
        "avg_evidence_coverage_score": _avg(evidence_scores) or 0.0,
        "low_evidence_coverage_rate": _avg(
            [
                1.0
                if (_num((item.get("evidence_coverage_summary") or {}).get("coverage_score")) or 0.0)
                < 0.55
                else 0.0
                for item in case_results
            ]
        )
        or 0.0,
        "error_rate": _avg([1.0 if item.get("error") else 0.0 for item in case_results])
        or 0.0,
    }


def _summarize_performance(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = [item.get("latency_ms") for item in case_results]
    first_answer = [item.get("first_answer_ms") for item in case_results]
    stage_bucket: Dict[str, List[float]] = defaultdict(list)
    for item in case_results:
        for stage_name, duration in (item.get("stage_durations_ms") or {}).items():
            casted = _num(duration)
            if casted is not None:
                stage_bucket[stage_name].append(casted)
    stage_summary = []
    for stage_name, durations in sorted(stage_bucket.items()):
        stage_summary.append(
            {
                "stage": stage_name,
                "avg_duration_ms": _avg(durations),
                "p95_duration_ms": _pct(durations, 0.95),
                "count": len(durations),
            }
        )
    return {
        "avg_total_duration_ms": _avg(total),
        "p50_total_duration_ms": _pct(total, 0.50),
        "p95_total_duration_ms": _pct(total, 0.95),
        "avg_first_answer_ms": _avg(first_answer),
        "p50_first_answer_ms": _pct(first_answer, 0.50),
        "p95_first_answer_ms": _pct(first_answer, 0.95),
        "stages": stage_summary,
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
    result: Dict[str, Any] = {}
    for key in (
        "factual_correctness",
        "faithfulness",
        "response_relevancy",
        "llm_context_recall",
        "id_based_context_precision",
        "id_based_context_recall",
    ):
        if key in ragas_metrics:
            result[key] = ragas_metrics.get(key)
    for key in ("hit@1", "hit@3", "recall@3", "mrr@3"):
        if key in retrieval_metrics:
            result[key] = retrieval_metrics.get(key)
    for key in (
        "avg_evidence_coverage_score",
        "rescue_retry_rate",
        "context_expansion_rate",
        "structured_answer_rate",
    ):
        pipeline_metrics = summary.get("pipeline_metrics") or {}
        if key in pipeline_metrics:
            result[key] = pipeline_metrics.get(key)
    for key in ("avg_total_duration_ms", "p95_total_duration_ms"):
        if key in performance_metrics:
            result[key] = performance_metrics.get(key)
    return result


def _summarize_variant(
    variant_name: str,
    case_results: List[Dict[str, Any]],
    ragas_bundle: Optional[Dict[str, Any]] = None,
    description: str = "",
    technique: str = "",
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
        "retrieval_metrics": retrieval_metrics,
        "retrieval_coverage": retrieval_coverage,
        "retrieval_ground_truth": retrieval_ground_truth,
        "pipeline_metrics": _summarize_pipeline(case_results),
        "performance_metrics": _summarize_performance(case_results),
        "warnings": warnings,
    }
    if ragas_bundle is None:
        summary["ragas_metrics"] = _summarize_ragas_from_cases(case_results)
    else:
        summary["ragas_metrics"] = ragas_bundle.get("summary") or {}
        summary["ragas_coverage"] = ragas_bundle.get("coverage") or {}
        summary["ragas_errors"] = ragas_bundle.get("errors") or {}
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


def _build_comparison_report(variants: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for variant_name, payload in variants.items():
        compare_to = payload.get("compare_to") or VARIANTS[variant_name].get(
            "compare_to"
        )
        if not compare_to or compare_to not in variants:
            continue
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
        report[f"{variant_name}_vs_{compare_to}"] = {
            "variant": variant_name,
            "compare_to": compare_to,
            "technique": VARIANTS[variant_name].get("technique"),
            "overall": overall,
            "by_query_type": group_report,
        }
    return report


def evaluate_variants(
    dataset_path: str,
    variant_names: Sequence[str],
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    cases = _prepare_cases(load_cases(dataset_path))
    resolved_variants = _resolve_variants(variant_names)
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
    for index, variant_name in enumerate(resolved_variants, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "variant_started",
                    "variant_name": variant_name,
                    "variant_index": index,
                    "total_variants": total_variants,
                }
            )
        case_results = [_run_single_case(case, variant_name) for case in cases]
        ragas_bundle = _run_ragas(case_results)
        _merge_ragas_scores(case_results, ragas_bundle.get("per_case_scores") or [])
        grouped_summaries = {
            query_type: _summarize_variant(
                variant_name=query_type,
                case_results=grouped_cases,
                description=f"query_type={query_type}",
                technique="QueryType Slice",
            )
            for query_type, grouped_cases in _group_by_query_type(case_results).items()
        }
        variants_payload[variant_name] = {
            "description": VARIANTS[variant_name].get("description"),
            "technique": VARIANTS[variant_name].get("technique"),
            "compare_to": VARIANTS[variant_name].get("compare_to"),
            "summary": _summarize_variant(variant_name, case_results, ragas_bundle),
            "by_query_type": grouped_summaries,
            "case_results": case_results,
        }
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "variant_completed",
                    "variant_name": variant_name,
                    "variant_index": index,
                    "total_variants": total_variants,
                }
            )

    final_variant = (
        "agentic_enhanced_system"
        if "agentic_enhanced_system" in variants_payload
        else "final_system"
        if "final_system" in variants_payload
        else resolved_variants[-1]
    )
    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()),
        "case_count": len(cases),
        "variants": variants_payload,
        "final_variant": final_variant,
        "final_system_metrics": variants_payload.get(final_variant, {}).get(
            "summary", {}
        ),
        "comparisons": _build_comparison_report(variants_payload),
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
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    report = evaluate_variants(
        dataset_path,
        variant_names,
        progress_callback=progress_callback,
    )
    resolved_output_path = (
        Path(output_path) if output_path else _default_output_path()
    )
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
