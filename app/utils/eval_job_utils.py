import json
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from app.utils.path_util import PROJECT_ROOT
from app.utils.eval_report_utils import load_evaluation_report_payload
from app.utils.unified_rag_eval import (
    DEFAULT_VARIANTS,
    EvaluationCancelledError,
    _build_comparison_report,
    evaluate_variants,
    evaluate_variants_to_file,
    get_feature_catalog,
    get_variant_catalog,
    register_feature_variants,
)

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_CANCELLING = "cancelling"
JOB_STATUS_CANCELLED = "cancelled"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

_jobs: Dict[str, Dict[str, Any]] = {}
_job_order: List[str] = []
_jobs_lock = Lock()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _template_dataset_path() -> str:
    candidates = [
        Path(PROJECT_ROOT) / "docs" / "graph_eval_cases.docs.json",
        Path(PROJECT_ROOT) / "test" / "graph_eval_cases.template.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def _dataset_catalog() -> List[Dict[str, Any]]:
    docs_dir = Path(PROJECT_ROOT) / "docs"
    items = [
        {
            "key": "smoke_eval_25",
            "label": "Smoke Eval 25",
            "dataset_name": "insightvault_25_case_eval",
            "description": "自动生成的 25 条 smoke test，用于快速检查评测链路和检索对齐。",
            "path": docs_dir / "graph_eval_cases.docs.json",
            "case_count": 25,
            "is_default": True,
        },
        {
            "key": "business_benchmark_30",
            "label": "Business Benchmark 30",
            "dataset_name": "business_benchmark_30",
            "description": "人工业务基准集，覆盖 6 类问题各 5 条，优先观察 LLM Judge、检索、时延和缓存口径。",
            "path": docs_dir / "business_benchmark_30.docs.json",
            "case_count": 30,
            "is_default": False,
        },
        {
            "key": "business_benchmark_100",
            "label": "Business Benchmark 100",
            "dataset_name": "business_benchmark_100",
            "description": "完整 100 样本业务基准集，覆盖 6 类问题、双产品各 50 条，适合正式对比评测。",
            "path": docs_dir / "business_benchmark_100.docs.json",
            "case_count": 100,
            "is_default": False,
        },
    ]
    catalog: List[Dict[str, Any]] = []
    for item in items:
        path = Path(item["path"])
        payload = dict(item)
        payload["path"] = str(path.resolve())
        payload["exists"] = path.exists()
        catalog.append(payload)
    return catalog


def _clone_job(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if job is None:
        return None
    return deepcopy(job)


def _default_eval_output_path() -> Path:
    output_dir = Path(PROJECT_ROOT) / "reports" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"unified_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = output_dir / f"{stem}.json"
    counter = 1
    while path.exists():
        path = output_dir / f"{stem}_{counter}.json"
        counter += 1
    return path


def _feature_signature(runtime: Dict[str, Any] | None) -> tuple[str, ...] | None:
    if not runtime:
        return None
    features = runtime.get("resolved_features")
    if not isinstance(features, list):
        features = runtime.get("requested_features")
    if not isinstance(features, list):
        return None
    return tuple(str(item).strip() for item in features if str(item).strip())


def _report_feature_signatures(report: Dict[str, Any]) -> set[tuple[str, ...]]:
    signatures: set[tuple[str, ...]] = set()
    for payload in (report.get("variants") or {}).values():
        if not isinstance(payload, dict):
            continue
        signature = _feature_signature(payload.get("feature_variant") or {})
        if signature is not None:
            signatures.add(signature)
    return signatures


def merge_appended_evaluation_report(
    base_report: Dict[str, Any],
    appended_report: Dict[str, Any],
    appended_variant_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    merged = deepcopy(base_report)
    base_variants = merged.setdefault("variants", {})
    appended_variants = appended_report.get("variants") or {}
    appended_names = [
        str(name)
        for name in (appended_variant_names or appended_variants.keys())
        if str(name) in appended_variants
    ]

    for variant_name in appended_names:
        base_variants[variant_name] = deepcopy(appended_variants[variant_name])

    evaluation_method = dict(merged.get("evaluation_method") or {})
    existing_order = [
        str(name)
        for name in (
            evaluation_method.get("execution_order")
            or list((base_report.get("variants") or {}).keys())
        )
        if str(name) in base_variants
    ]
    execution_order: List[str] = []
    for variant_name in [*existing_order, *appended_names]:
        if variant_name and variant_name not in execution_order:
            execution_order.append(variant_name)

    existing_feature_variants = list(evaluation_method.get("feature_variants") or [])
    existing_feature_names = {
        str(item.get("name") or "")
        for item in existing_feature_variants
        if isinstance(item, dict)
    }
    for variant_name in appended_names:
        feature_variant = (base_variants.get(variant_name) or {}).get("feature_variant")
        if isinstance(feature_variant, dict) and str(feature_variant.get("name") or "") not in existing_feature_names:
            existing_feature_variants.append(deepcopy(feature_variant))
            existing_feature_names.add(str(feature_variant.get("name") or ""))

    evaluation_method.update(
        {
            "execution_order": execution_order,
            "feature_variants": existing_feature_variants,
            "mode": "controlled_ablation",
        }
    )
    merged["evaluation_method"] = evaluation_method
    merged["generated_at"] = datetime.now().isoformat()
    merged["case_count"] = int(base_report.get("case_count") or appended_report.get("case_count") or 0)
    merged["dataset_path"] = base_report.get("dataset_path") or appended_report.get("dataset_path")
    merged["dataset_name"] = base_report.get("dataset_name") or appended_report.get("dataset_name")

    if appended_names:
        final_variant = appended_names[-1]
        merged["final_variant"] = final_variant
        merged["final_system_metrics"] = (base_variants.get(final_variant) or {}).get("summary") or {}

    merged_comparisons = dict(base_report.get("comparisons") or {})
    merged_comparisons.update(
        _build_comparison_report(
            base_variants,
            pairwise_order=execution_order,
            pairwise_current_names=appended_names,
        )
    )
    merged["comparisons"] = merged_comparisons
    merged["append_metadata"] = {
        "base_generated_at": base_report.get("generated_at"),
        "appended_at": merged["generated_at"],
        "appended_variants": appended_names,
    }
    return merged


def get_evaluation_config() -> Dict[str, Any]:
    return {
        "template_dataset_path": _template_dataset_path(),
        "default_variants": list(DEFAULT_VARIANTS),
        "variant_catalog": get_variant_catalog(),
        "feature_catalog": get_feature_catalog(),
        "dataset_catalog": _dataset_catalog(),
    }


def create_evaluation_job(
    dataset_path: str,
    variants: Sequence[str],
    feature_variants: Sequence[Dict[str, Any]] | None = None,
    output_path: str | None = None,
) -> Dict[str, Any]:
    resolved_dataset_path = Path(dataset_path).expanduser().resolve()
    if not resolved_dataset_path.exists() or not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"评测数据集不存在: {resolved_dataset_path}")
    job_id = f"eval-job-{uuid.uuid4().hex[:12]}"
    normalized_variants = [str(variant).strip() for variant in variants if str(variant).strip()]
    registered_feature_variants: List[Dict[str, Any]] = []
    if feature_variants:
        registered_feature_variants.extend(register_feature_variants(feature_variants))
        for item in registered_feature_variants:
            variant_name = str(item.get("name") or "").strip()
            if variant_name and variant_name not in normalized_variants:
                normalized_variants.append(variant_name)
    resolved_variants = normalized_variants or list(DEFAULT_VARIANTS)
    job = {
        "job_id": job_id,
        "job_mode": "standard",
        "append_base_report_id": "",
        "status": JOB_STATUS_PENDING,
        "dataset_path": str(resolved_dataset_path),
        "variants": resolved_variants,
        "feature_variants": registered_feature_variants,
        "feature_variant_specs": [dict(item or {}) for item in (feature_variants or [])],
        "output_path": (
            str(Path(output_path).expanduser().resolve()) if output_path else ""
        ),
        "progress_message": "等待开始",
        "phase": "pending",
        "current_variant": "",
        "completed_variants": 0,
        "total_variants": len(resolved_variants),
        "case_count": 0,
        "current_case_id": "",
        "current_case_query": "",
        "completed_cases": 0,
        "current_variant_total_cases": 0,
        "warmup_round": 0,
        "warmup_rounds": 0,
        "cancel_requested": False,
        "report_id": "",
        "report_path": "",
        "error": "",
        "created_at": _now_iso(),
        "started_at": "",
        "finished_at": "",
        "last_progress_at": _now_iso(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        _job_order.insert(0, job_id)
    return _clone_job(job) or {}


def create_append_evaluation_job(
    report_id: str,
    feature_variants: Sequence[Dict[str, Any]] | None = None,
    output_path: str | None = None,
) -> Dict[str, Any]:
    loaded = load_evaluation_report_payload(report_id)
    if loaded is None:
        raise FileNotFoundError("evaluation report not found")
    _, base_report = loaded
    dataset_path = str(base_report.get("dataset_path") or "").strip()
    if not dataset_path:
        raise ValueError("原报告缺少 dataset_path，无法追加测评")
    resolved_dataset_path = Path(dataset_path).expanduser().resolve()
    if not resolved_dataset_path.exists() or not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"评测数据集不存在: {resolved_dataset_path}")
    if not feature_variants:
        raise ValueError("请至少提供一个要追加的功能组合")

    existing_signatures = _report_feature_signatures(base_report)
    seen_signatures: set[tuple[str, ...]] = set()
    registered_feature_variants = register_feature_variants(feature_variants)
    normalized_variants: List[str] = []
    for item in registered_feature_variants:
        signature = _feature_signature(item)
        if signature in existing_signatures:
            label = str(item.get("label") or item.get("name") or "功能组合")
            raise ValueError(f"功能组合已存在，无法重复追加: {label}")
        if signature in seen_signatures:
            label = str(item.get("label") or item.get("name") or "功能组合")
            raise ValueError(f"本次追加中存在重复功能组合: {label}")
        if signature is not None:
            seen_signatures.add(signature)
        variant_name = str(item.get("name") or "").strip()
        if variant_name and variant_name not in normalized_variants:
            normalized_variants.append(variant_name)

    job_id = f"eval-job-{uuid.uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "job_mode": "append_report",
        "append_base_report_id": report_id,
        "status": JOB_STATUS_PENDING,
        "dataset_path": str(resolved_dataset_path),
        "variants": normalized_variants,
        "feature_variants": registered_feature_variants,
        "feature_variant_specs": [dict(item or {}) for item in (feature_variants or [])],
        "output_path": (
            str(Path(output_path).expanduser().resolve()) if output_path else ""
        ),
        "progress_message": "等待开始追加测评",
        "phase": "pending",
        "current_variant": "",
        "completed_variants": 0,
        "total_variants": len(normalized_variants),
        "case_count": int(base_report.get("case_count") or 0),
        "current_case_id": "",
        "current_case_query": "",
        "completed_cases": 0,
        "current_variant_total_cases": 0,
        "warmup_round": 0,
        "warmup_rounds": 0,
        "cancel_requested": False,
        "report_id": "",
        "report_path": "",
        "error": "",
        "created_at": _now_iso(),
        "started_at": "",
        "finished_at": "",
        "last_progress_at": _now_iso(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        _job_order.insert(0, job_id)
    return _clone_job(job) or {}


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        fields.setdefault("last_progress_at", _now_iso())
        job.update(fields)


def _is_cancel_requested(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id) or {}
        return bool(job.get("cancel_requested"))


def get_evaluation_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return _clone_job(_jobs.get(job_id))


def list_evaluation_jobs(limit: int = 10) -> Dict[str, Any]:
    with _jobs_lock:
        jobs = [_clone_job(_jobs.get(job_id)) for job_id in _job_order[:limit]]
    return {"jobs": [job for job in jobs if job is not None]}


def get_active_evaluation_job_ids() -> List[str]:
    with _jobs_lock:
        return [
            job_id
            for job_id in _job_order
            if (_jobs.get(job_id) or {}).get("status")
            in (JOB_STATUS_PENDING, JOB_STATUS_RUNNING, JOB_STATUS_CANCELLING)
        ]


def cancel_evaluation_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None

        status = str(job.get("status") or "")
        if status in (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED):
            return deepcopy(job)

        job["cancel_requested"] = True
        if status == JOB_STATUS_PENDING:
            job.update(
                {
                    "status": JOB_STATUS_CANCELLED,
                    "phase": "cancelled",
                    "progress_message": "评测已取消",
                    "finished_at": _now_iso(),
                    "current_variant": "",
                    "current_case_id": "",
                    "current_case_query": "",
                }
            )
        else:
            job.update(
                {
                    "status": JOB_STATUS_CANCELLING,
                    "phase": "cancelling",
                    "progress_message": "正在停止评测...",
                }
            )
        return deepcopy(job)


def _run_append_report_job(
    job: Dict[str, Any],
    progress_callback,
    cancel_callback,
) -> Dict[str, Any]:
    base_report_id = str(job.get("append_base_report_id") or "").strip()
    loaded = load_evaluation_report_payload(base_report_id)
    if loaded is None:
        raise FileNotFoundError("evaluation report not found")
    _, base_report = loaded
    dataset_path = str(base_report.get("dataset_path") or job.get("dataset_path") or "").strip()
    variants = list(job.get("variants") or [])
    feature_variant_specs = list(job.get("feature_variant_specs") or [])
    appended_report = evaluate_variants(
        dataset_path=dataset_path,
        variant_names=variants,
        feature_variant_specs=feature_variant_specs,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    merged_report = merge_appended_evaluation_report(
        base_report,
        appended_report,
        appended_variant_names=variants,
    )
    output_path = str(job.get("output_path") or "").strip()
    resolved_output_path = Path(output_path).expanduser().resolve() if output_path else _default_eval_output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(merged_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "report_saved",
                "output_path": str(resolved_output_path.resolve()),
            }
        )
    return {
        "report": merged_report,
        "output_path": str(resolved_output_path.resolve()),
    }


def run_evaluation_job(job_id: str) -> None:
    job = get_evaluation_job(job_id)
    if job is None:
        return

    if job.get("cancel_requested") or job.get("status") == JOB_STATUS_CANCELLED:
        _update_job(
            job_id,
            status=JOB_STATUS_CANCELLED,
            progress_message="评测已取消",
            finished_at=_now_iso(),
        )
        return

    dataset_path = str(job.get("dataset_path") or "").strip()
    variants = list(job.get("variants") or list(DEFAULT_VARIANTS))
    feature_variant_specs = list(job.get("feature_variant_specs") or [])
    output_path = str(job.get("output_path") or "").strip() or None
    job_mode = str(job.get("job_mode") or "standard")
    _update_job(
        job_id,
        status=JOB_STATUS_RUNNING,
        started_at=_now_iso(),
        progress_message="正在加载数据集" if job_mode == "standard" else "正在加载原报告与数据集",
        phase="loading_dataset",
        error="",
    )

    def on_progress(event: Dict[str, Any]) -> None:
        if _is_cancel_requested(job_id):
            raise EvaluationCancelledError("用户已取消评测任务")
        stage = str(event.get("stage") or "")
        if stage == "dataset_loaded":
            _update_job(
                job_id,
                progress_message=f"已加载数据集，共 {int(event.get('case_count') or 0)} 条样本",
                phase="dataset_loaded",
                case_count=int(event.get("case_count") or 0),
                total_variants=int(event.get("total_variants") or len(variants)),
            )
        elif stage == "variant_started":
            current_variant = str(event.get("variant_name") or "")
            index = int(event.get("variant_index") or 0)
            total = int(event.get("total_variants") or len(variants))
            _update_job(
                job_id,
                phase="variant_started",
                current_variant=current_variant,
                completed_variants=max(index - 1, 0),
                total_variants=total,
                current_case_id="",
                current_case_query="",
                completed_cases=0,
                current_variant_total_cases=int(event.get("total_cases") or 0),
                warmup_round=0,
                warmup_rounds=0,
                progress_message=f"正在评测 {current_variant} ({index}/{total})",
            )
        elif stage == "warmup_started":
            current_variant = str(event.get("variant_name") or "")
            warmup_round = int(event.get("warmup_round") or 0)
            warmup_rounds = int(event.get("warmup_rounds") or 0)
            total_cases = int(event.get("total_cases") or 0)
            cache_temperature = str(event.get("cache_temperature") or "")
            action_label = "冷缓存测量" if cache_temperature == "cold" else "预热缓存"
            _update_job(
                job_id,
                phase="warmup",
                current_variant=current_variant,
                current_case_id="",
                current_case_query="",
                completed_cases=0,
                current_variant_total_cases=total_cases,
                warmup_round=warmup_round,
                warmup_rounds=warmup_rounds,
                progress_message=(
                    f"正在{action_label} {current_variant} "
                    f"({warmup_round}/{max(warmup_rounds, 1)})"
                ),
            )
        elif stage == "warmup_case_started":
            current_variant = str(event.get("variant_name") or "")
            case_index = int(event.get("case_index") or 0)
            total_cases = int(event.get("total_cases") or 0)
            warmup_round = int(event.get("warmup_round") or 0)
            warmup_rounds = int(event.get("warmup_rounds") or 0)
            case_id = str(event.get("case_id") or "")
            case_query = str(event.get("query") or "")
            cache_temperature = str(event.get("cache_temperature") or "")
            action_label = "冷缓存测量" if cache_temperature == "cold" else "预热缓存"
            _update_job(
                job_id,
                phase="warmup",
                current_variant=current_variant,
                current_case_id=case_id,
                current_case_query=case_query,
                completed_cases=max(case_index - 1, 0),
                current_variant_total_cases=total_cases,
                warmup_round=warmup_round,
                warmup_rounds=warmup_rounds,
                progress_message=(
                    f"正在{action_label} {current_variant} 第 {warmup_round}/{max(warmup_rounds, 1)} 轮 "
                    f"样本 {case_index}/{total_cases}: {case_id or case_query}"
                ),
            )
        elif stage == "warmup_case_completed":
            current_variant = str(event.get("variant_name") or "")
            case_index = int(event.get("case_index") or 0)
            total_cases = int(event.get("total_cases") or 0)
            warmup_round = int(event.get("warmup_round") or 0)
            warmup_rounds = int(event.get("warmup_rounds") or 0)
            case_id = str(event.get("case_id") or "")
            case_query = str(event.get("query") or "")
            cache_temperature = str(event.get("cache_temperature") or "")
            action_label = "冷缓存测量" if cache_temperature == "cold" else "预热"
            _update_job(
                job_id,
                phase="warmup",
                current_variant=current_variant,
                current_case_id=case_id,
                current_case_query=case_query,
                completed_cases=case_index,
                current_variant_total_cases=total_cases,
                warmup_round=warmup_round,
                warmup_rounds=warmup_rounds,
                progress_message=(
                    f"已完成{action_label} {current_variant} 第 {warmup_round}/{max(warmup_rounds, 1)} 轮 "
                    f"样本 {case_index}/{total_cases}: {case_id or case_query}"
                ),
            )
        elif stage == "warmup_completed":
            current_variant = str(event.get("variant_name") or "")
            warmup_round = int(event.get("warmup_round") or 0)
            warmup_rounds = int(event.get("warmup_rounds") or 0)
            total_cases = int(event.get("total_cases") or 0)
            more_rounds = warmup_round < warmup_rounds
            cache_temperature = str(event.get("cache_temperature") or "")
            completed_label = "冷缓存测量完成" if cache_temperature == "cold" else "缓存预热完成"
            _update_job(
                job_id,
                phase="warmup" if more_rounds else "evaluation",
                current_variant=current_variant,
                current_case_id="",
                current_case_query="",
                completed_cases=total_cases if more_rounds else 0,
                current_variant_total_cases=total_cases,
                warmup_round=warmup_round,
                warmup_rounds=warmup_rounds,
                progress_message=(
                    f"{completed_label}，开始热缓存正式评测 {current_variant}"
                    if not more_rounds
                    else f"第 {warmup_round}/{warmup_rounds} 轮缓存预热完成"
                ),
            )
        elif stage == "case_started":
            current_variant = str(event.get("variant_name") or "")
            case_index = int(event.get("case_index") or 0)
            total_cases = int(event.get("total_cases") or 0)
            case_id = str(event.get("case_id") or "")
            case_query = str(event.get("query") or "")
            _update_job(
                job_id,
                phase="evaluation",
                current_variant=current_variant,
                current_case_id=case_id,
                current_case_query=case_query,
                completed_cases=max(case_index - 1, 0),
                current_variant_total_cases=total_cases,
                progress_message=f"正在评测 {current_variant} 样本 {case_index}/{total_cases}: {case_id or case_query}",
            )
        elif stage == "case_completed":
            current_variant = str(event.get("variant_name") or "")
            case_index = int(event.get("case_index") or 0)
            total_cases = int(event.get("total_cases") or 0)
            case_id = str(event.get("case_id") or "")
            case_query = str(event.get("query") or "")
            error = str(event.get("error") or "")
            suffix = f"（异常: {error}）" if error else ""
            _update_job(
                job_id,
                phase="evaluation",
                current_variant=current_variant,
                current_case_id=case_id,
                current_case_query=case_query,
                completed_cases=case_index,
                current_variant_total_cases=total_cases,
                progress_message=f"已完成 {current_variant} 样本 {case_index}/{total_cases}: {case_id or case_query}{suffix}",
            )
        elif stage == "variant_completed":
            current_variant = str(event.get("variant_name") or "")
            index = int(event.get("variant_index") or 0)
            total = int(event.get("total_variants") or len(variants))
            _update_job(
                job_id,
                phase="variant_completed",
                current_variant=current_variant,
                completed_variants=index,
                total_variants=total,
                completed_cases=int(event.get("total_cases") or 0),
                current_variant_total_cases=int(event.get("total_cases") or 0),
                warmup_round=0,
                warmup_rounds=0,
                progress_message=f"已完成 {current_variant} ({index}/{total})",
            )
        elif stage == "report_ready":
            _update_job(
                job_id,
                phase="saving_report",
                progress_message="评测完成，正在写入报告",
            )
        elif stage == "report_saved":
            report_path = str(event.get("output_path") or "")
            _update_job(
                job_id,
                phase="report_saved",
                progress_message="报告已生成",
                report_path=report_path,
                report_id=Path(report_path).name if report_path else "",
            )

    try:
        if job_mode == "append_report":
            result = _run_append_report_job(
                job,
                progress_callback=on_progress,
                cancel_callback=lambda: _is_cancel_requested(job_id),
            )
        else:
            result = evaluate_variants_to_file(
                dataset_path=dataset_path,
                variant_names=variants,
                output_path=output_path,
                feature_variant_specs=feature_variant_specs,
                progress_callback=on_progress,
                cancel_callback=lambda: _is_cancel_requested(job_id),
            )
        report_path = str(result.get("output_path") or "")
        _update_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            phase="completed",
            finished_at=_now_iso(),
            completed_variants=len(variants),
            total_variants=len(variants),
            current_variant="",
            current_case_id="",
            current_case_query="",
            warmup_round=0,
            warmup_rounds=0,
            progress_message="评测完成",
            report_path=report_path,
            report_id=Path(report_path).name if report_path else "",
        )
    except EvaluationCancelledError:
        _update_job(
            job_id,
            status=JOB_STATUS_CANCELLED,
            phase="cancelled",
            finished_at=_now_iso(),
            current_variant="",
            current_case_id="",
            current_case_query="",
            warmup_round=0,
            warmup_rounds=0,
            progress_message="评测已取消",
            error="",
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JOB_STATUS_FAILED,
            phase="failed",
            finished_at=_now_iso(),
            progress_message="评测失败",
            error=str(exc),
        )
