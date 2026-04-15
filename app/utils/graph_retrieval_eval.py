import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from app.clients.milvus_schema import extract_chunk_id
from app.clients.neo4j_graph_utils import query_graph_context
from app.conf.query_threshold_config import query_threshold_config
from app.query_process.agent.graph_query_utils import (
    GRAPH_PREFERRED_QUERY_TYPES,
    build_query_route,
    build_retrieval_plan,
    get_rrf_weight_multipliers,
)
from app.query_process.agent.nodes.node_rrf import _as_entity_list, reciprocal_rank_fusion
from app.query_process.agent.retrieval_utils import run_bm25_search, run_embedding_hybrid_search
from app.utils.path_util import PROJECT_ROOT
from app.utils.retrieval_eval import (
    OUTPUT_FIELDS,
    load_cases,
    mrr_at_k,
    normalize_ids,
    recall_at_k,
    run_mode as run_baseline_mode,
)

GRAPH_EVAL_MODES = [
    "embedding",
    "bm25",
    "embedding_bm25_rrf",
    "kg",
    "embedding_bm25_kg_rrf",
    "graph_first_hybrid",
]
DEFAULT_K_VALUES = [1, 3, 5, 10]
BASELINE_MODES = {"embedding", "bm25", "embedding_bm25_rrf"}


def _query_text(case: Dict[str, Any]) -> str:
    return (case.get("query") or case.get("rewritten_query") or "").strip()


def _item_names(case: Dict[str, Any]) -> List[str]:
    return [str(name).strip() for name in (case.get("item_names") or []) if str(name).strip()]


def _resolve_route(case: Dict[str, Any]) -> Dict[str, Any]:
    query = _query_text(case)
    item_names = _item_names(case)
    route = build_query_route(query, item_names)
    case_query_type = str(case.get("query_type") or "").strip()
    if case_query_type:
        graph_preferred = case_query_type in GRAPH_PREFERRED_QUERY_TYPES
        route["query_type"] = case_query_type
        route["graph_preferred"] = graph_preferred
        route["retrieval_plan"] = build_retrieval_plan(case_query_type, graph_preferred)
    return route


def _extract_ids(results: Sequence[Any]) -> List[str]:
    ids: List[str] = []
    for entity in _as_entity_list(results):
        chunk_id = extract_chunk_id(entity)
        if chunk_id is None:
            continue
        ids.append(str(chunk_id))
    return ids


def _run_kg(case: Dict[str, Any], route: Dict[str, Any], top_k: int) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    query = _query_text(case)
    item_names = _item_names(case)
    result = query_graph_context(
        query,
        item_names,
        query_type=route.get("query_type") or "general",
        focus_terms=route.get("focus_terms") or [],
        limit=top_k,
    )
    kg_chunks = result.get("kg_chunks") or []
    return _extract_ids(kg_chunks), result.get("summary") or {}, kg_chunks


def _fuse_results(source_weights: Sequence[Tuple[List[Dict[str, Any]], float]], top_k: int) -> List[str]:
    fused = reciprocal_rank_fusion(
        source_weights,
        k=query_threshold_config.rrf_k,
        max_results=top_k,
    )
    ids: List[str] = []
    for doc, _ in fused:
        chunk_id = extract_chunk_id(doc)
        if chunk_id is not None:
            ids.append(str(chunk_id))
    return ids


def run_graph_mode(case: Dict[str, Any], mode: str, top_k: int) -> Tuple[List[str], Dict[str, Any]]:
    route = _resolve_route(case)
    query = _query_text(case)
    item_names = _item_names(case)
    cfg = query_threshold_config

    if mode in BASELINE_MODES:
        predicted_ids = run_baseline_mode(case, mode, top_k)
        return predicted_ids, {"route": route, "kg_query_summary": {}, "retriever_counts": {}}

    kg_ids: List[str] = []
    kg_summary: Dict[str, Any] = {}
    kg_chunks: List[Dict[str, Any]] = []
    embedding_hits: List[Dict[str, Any]] = []
    bm25_hits: List[Dict[str, Any]] = []

    if mode in {"kg", "embedding_bm25_kg_rrf", "graph_first_hybrid"}:
        kg_ids, kg_summary, kg_chunks = _run_kg(case, route, max(top_k, int((route.get("retrieval_plan") or {}).get("graph_limit", top_k) or top_k)))

    if mode in {"embedding_bm25_kg_rrf", "graph_first_hybrid"}:
        plan = route.get("retrieval_plan") or {}
        if mode == "embedding_bm25_kg_rrf" or plan.get("run_embedding", True):
            embedding_hits = run_embedding_hybrid_search(
                query_text=query,
                item_names=item_names,
                top_k=max(top_k, cfg.embedding_top_k),
                output_fields=list(OUTPUT_FIELDS),
            )
        if mode == "embedding_bm25_kg_rrf" or plan.get("run_bm25", True):
            bm25_hits = run_bm25_search(
                query_text=query,
                item_names=item_names,
                top_k=max(top_k, cfg.bm25_top_k),
                output_fields=list(OUTPUT_FIELDS),
            )

    if mode == "kg":
        return kg_ids[:top_k], {
            "route": route,
            "kg_query_summary": kg_summary,
            "retriever_counts": {
                "kg": len(kg_chunks),
                "embedding": 0,
                "bm25": 0,
            },
        }

    if mode == "embedding_bm25_kg_rrf":
        predicted_ids = _fuse_results(
            [
                (_as_entity_list(embedding_hits), cfg.rrf_weight_embedding),
                (_as_entity_list(bm25_hits), cfg.rrf_weight_bm25),
                (_as_entity_list(kg_chunks), cfg.rrf_weight_kg),
            ],
            top_k,
        )
        return predicted_ids, {
            "route": route,
            "kg_query_summary": kg_summary,
            "retriever_counts": {
                "kg": len(kg_chunks),
                "embedding": len(_as_entity_list(embedding_hits)),
                "bm25": len(_as_entity_list(bm25_hits)),
            },
        }

    if mode == "graph_first_hybrid":
        multipliers = get_rrf_weight_multipliers({"retrieval_plan": route.get("retrieval_plan") or {}})
        predicted_ids = _fuse_results(
            [
                (_as_entity_list(embedding_hits), cfg.rrf_weight_embedding * multipliers["embedding"]),
                (_as_entity_list(bm25_hits), cfg.rrf_weight_bm25 * multipliers["bm25"]),
                (_as_entity_list(kg_chunks), cfg.rrf_weight_kg * multipliers["kg"]),
            ],
            top_k,
        )
        return predicted_ids, {
            "route": route,
            "kg_query_summary": kg_summary,
            "retriever_counts": {
                "kg": len(kg_chunks),
                "embedding": len(_as_entity_list(embedding_hits)),
                "bm25": len(_as_entity_list(bm25_hits)),
            },
        }

    raise ValueError(f"不支持的评测模式: {mode}")


def _init_metric_bucket(k_values: Sequence[int]) -> Dict[str, float]:
    bucket: Dict[str, float] = {
        "evaluated_cases": 0,
        "empty_result_cases": 0,
    }
    for k in k_values:
        bucket[f"recall@{k}"] = 0.0
        bucket[f"mrr@{k}"] = 0.0
    return bucket


def _finalize_metric_bucket(bucket: Dict[str, float], k_values: Sequence[int]) -> Dict[str, Any]:
    evaluated_cases = int(bucket["evaluated_cases"])
    summary: Dict[str, Any] = {
        "evaluated_cases": evaluated_cases,
        "empty_result_rate": bucket["empty_result_cases"] / evaluated_cases if evaluated_cases else 0.0,
    }
    for k in k_values:
        summary[f"recall@{k}"] = bucket[f"recall@{k}"] / evaluated_cases if evaluated_cases else 0.0
        summary[f"mrr@{k}"] = bucket[f"mrr@{k}"] / evaluated_cases if evaluated_cases else 0.0
    return summary


def evaluate_cases(cases: Sequence[Dict[str, Any]], modes: Sequence[str], k_values: Sequence[int], dataset_path: str) -> Dict[str, Any]:
    max_k = max(k_values)
    overall_metrics = {mode: _init_metric_bucket(k_values) for mode in modes}
    by_type_metrics: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: {mode: _init_metric_bucket(k_values) for mode in modes}
    )
    details: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        relevant_ids = normalize_ids(
            case.get("relevant_chunk_ids")
            or case.get("relevant_ids")
            or case.get("chunk_ids")
            or []
        )
        if not relevant_ids:
            continue

        route = _resolve_route(case)
        query_type = route.get("query_type") or "general"
        case_detail: Dict[str, Any] = {
            "case_index": index,
            "case_id": case.get("case_id") or case.get("id") or str(index),
            "query": _query_text(case),
            "item_names": _item_names(case),
            "query_type": query_type,
            "relevant_chunk_ids": relevant_ids,
            "results": {},
        }

        for mode in modes:
            predicted_ids, meta = run_graph_mode(case, mode, max_k)
            overall_bucket = overall_metrics[mode]
            type_bucket = by_type_metrics[query_type][mode]
            overall_bucket["evaluated_cases"] += 1
            type_bucket["evaluated_cases"] += 1
            if not predicted_ids:
                overall_bucket["empty_result_cases"] += 1
                type_bucket["empty_result_cases"] += 1
            for k in k_values:
                recall_value = recall_at_k(predicted_ids, relevant_ids, k)
                mrr_value = mrr_at_k(predicted_ids, relevant_ids, k)
                overall_bucket[f"recall@{k}"] += recall_value
                overall_bucket[f"mrr@{k}"] += mrr_value
                type_bucket[f"recall@{k}"] += recall_value
                type_bucket[f"mrr@{k}"] += mrr_value
            case_detail["results"][mode] = {
                "predicted_ids": predicted_ids[:max_k],
                "route": meta.get("route") or {},
                "kg_query_summary": meta.get("kg_query_summary") or {},
                "retriever_counts": meta.get("retriever_counts") or {},
            }
        details.append(case_detail)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": dataset_path,
        "cases": len(cases),
        "modes": {},
        "by_query_type": {},
        "details": details,
    }
    for mode in modes:
        summary["modes"][mode] = _finalize_metric_bucket(overall_metrics[mode], k_values)
    for query_type, per_mode in by_type_metrics.items():
        summary["by_query_type"][query_type] = {
            mode: _finalize_metric_bucket(per_mode[mode], k_values) for mode in modes
        }
    return summary


def _default_output_path() -> Path:
    output_dir = Path(PROJECT_ROOT) / "output" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"graph_eval_{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSON/JSONL 数据集路径")
    parser.add_argument(
        "--modes",
        default=",".join(GRAPH_EVAL_MODES),
        help="逗号分隔的模式列表",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(value) for value in DEFAULT_K_VALUES),
        help="逗号分隔的 K 值，例如 1,3,5,10",
    )
    parser.add_argument("--output", default="", help="可选的 JSON 输出文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    cases = load_cases(args.dataset)
    summary = evaluate_cases(cases, modes, k_values, args.dataset)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[graph_eval_output] {output_path}")


if __name__ == "__main__":
    main()
