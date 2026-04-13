import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from app.conf.query_threshold_config import query_threshold_config
from app.query_process.agent.nodes.node_rrf import _as_entity_list, reciprocal_rank_fusion
from app.query_process.agent.retrieval_utils import run_bm25_search, run_embedding_hybrid_search

OUTPUT_FIELDS = [
    "chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    "image_urls",
]
DEFAULT_MODES = ["embedding", "bm25", "embedding_bm25_rrf"]
DEFAULT_K_VALUES = [1, 3, 5, 10]


def load_cases(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if path.suffix.lower() == ".jsonl":
        cases = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    raise ValueError("dataset 必须是 JSON 数组、带 cases 字段的 JSON 对象或 JSONL")


def normalize_ids(values: Sequence[Any]) -> List[str]:
    normalized = []
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def extract_chunk_ids(results: Sequence[Any]) -> List[str]:
    ids: List[str] = []
    for entity in _as_entity_list(results):
        chunk_id = entity.get("chunk_id") or entity.get("id")
        if chunk_id is None:
            continue
        ids.append(str(chunk_id))
    return ids


def run_mode(case: Dict[str, Any], mode: str, top_k: int) -> List[str]:
    query = (case.get("query") or case.get("rewritten_query") or "").strip()
    item_names = case.get("item_names") or []
    if mode == "embedding":
        hits = run_embedding_hybrid_search(
            query_text=query,
            item_names=item_names,
            top_k=top_k,
            output_fields=list(OUTPUT_FIELDS),
        )
        return extract_chunk_ids(hits)

    if mode == "bm25":
        hits = run_bm25_search(
            query_text=query,
            item_names=item_names,
            top_k=top_k,
            output_fields=list(OUTPUT_FIELDS),
        )
        return extract_chunk_ids(hits)

    if mode == "embedding_bm25_rrf":
        cfg = query_threshold_config
        embedding_hits = run_embedding_hybrid_search(
            query_text=query,
            item_names=item_names,
            top_k=max(top_k, cfg.embedding_top_k),
            output_fields=list(OUTPUT_FIELDS),
        )
        bm25_hits = run_bm25_search(
            query_text=query,
            item_names=item_names,
            top_k=max(top_k, cfg.bm25_top_k),
            output_fields=list(OUTPUT_FIELDS),
        )
        fused = reciprocal_rank_fusion(
            [
                (_as_entity_list(embedding_hits), cfg.rrf_weight_embedding),
                (_as_entity_list(bm25_hits), cfg.rrf_weight_bm25),
            ],
            k=cfg.rrf_k,
            max_results=top_k,
        )
        return [str((doc.get("chunk_id") or doc.get("id"))) for doc, _ in fused if doc.get("chunk_id") or doc.get("id")]

    raise ValueError(f"不支持的评测模式: {mode}")


def recall_at_k(predicted_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hit_count = sum(1 for chunk_id in predicted_ids[:k] if chunk_id in relevant)
    return hit_count / len(relevant)


def mrr_at_k(predicted_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    for index, chunk_id in enumerate(predicted_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def evaluate_cases(cases: Sequence[Dict[str, Any]], modes: Sequence[str], k_values: Sequence[int]) -> Dict[str, Any]:
    max_k = max(k_values)
    metrics: Dict[str, Dict[str, float]] = {}
    details: List[Dict[str, Any]] = []

    for mode in modes:
        metrics[mode] = {
            "evaluated_cases": 0,
            "empty_result_cases": 0,
        }
        for k in k_values:
            metrics[mode][f"recall@{k}"] = 0.0
            metrics[mode][f"mrr@{k}"] = 0.0

    for index, case in enumerate(cases, start=1):
        relevant_ids = normalize_ids(
            case.get("relevant_chunk_ids")
            or case.get("relevant_ids")
            or case.get("chunk_ids")
            or []
        )
        if not relevant_ids:
            continue

        case_detail = {
            "case_index": index,
            "case_id": case.get("case_id") or case.get("id") or str(index),
            "query": case.get("query") or case.get("rewritten_query") or "",
            "results": {},
        }

        for mode in modes:
            predicted_ids = run_mode(case, mode, max_k)
            metrics[mode]["evaluated_cases"] += 1
            if not predicted_ids:
                metrics[mode]["empty_result_cases"] += 1
            for k in k_values:
                metrics[mode][f"recall@{k}"] += recall_at_k(predicted_ids, relevant_ids, k)
                metrics[mode][f"mrr@{k}"] += mrr_at_k(predicted_ids, relevant_ids, k)
            case_detail["results"][mode] = predicted_ids[:max_k]

        details.append(case_detail)

    summary: Dict[str, Any] = {"cases": len(cases), "modes": {}, "details": details}
    for mode in modes:
        evaluated_cases = int(metrics[mode]["evaluated_cases"])
        mode_summary: Dict[str, Any] = {
            "evaluated_cases": evaluated_cases,
            "empty_result_rate": (
                metrics[mode]["empty_result_cases"] / evaluated_cases if evaluated_cases else 0.0
            ),
        }
        for k in k_values:
            mode_summary[f"recall@{k}"] = (
                metrics[mode][f"recall@{k}"] / evaluated_cases if evaluated_cases else 0.0
            )
            mode_summary[f"mrr@{k}"] = (
                metrics[mode][f"mrr@{k}"] / evaluated_cases if evaluated_cases else 0.0
            )
        summary["modes"][mode] = mode_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSON/JSONL 数据集路径")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="逗号分隔的模式列表: embedding,bm25,embedding_bm25_rrf",
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
    summary = evaluate_cases(cases, modes, k_values)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
