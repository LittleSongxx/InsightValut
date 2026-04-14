import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from app.utils.path_util import PROJECT_ROOT
from app.utils.unified_rag_eval import (
    DEFAULT_VARIANTS,
    evaluate_variants_to_file,
    get_variant_catalog,
)

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
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


def _clone_job(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if job is None:
        return None
    return deepcopy(job)


def get_evaluation_config() -> Dict[str, Any]:
    return {
        "template_dataset_path": _template_dataset_path(),
        "default_variants": list(DEFAULT_VARIANTS),
        "variant_catalog": get_variant_catalog(),
    }


def create_evaluation_job(
    dataset_path: str,
    variants: Sequence[str],
    output_path: str | None = None,
) -> Dict[str, Any]:
    resolved_dataset_path = Path(dataset_path).expanduser().resolve()
    if not resolved_dataset_path.exists() or not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"评测数据集不存在: {resolved_dataset_path}")
    job_id = f"eval-job-{uuid.uuid4().hex[:12]}"
    normalized_variants = [str(variant).strip() for variant in variants if str(variant).strip()]
    job = {
        "job_id": job_id,
        "status": JOB_STATUS_PENDING,
        "dataset_path": str(resolved_dataset_path),
        "variants": normalized_variants or list(DEFAULT_VARIANTS),
        "output_path": str(Path(output_path).expanduser().resolve()) if output_path else "",
        "progress_message": "等待开始",
        "current_variant": "",
        "completed_variants": 0,
        "total_variants": len(normalized_variants or list(DEFAULT_VARIANTS)),
        "case_count": 0,
        "report_id": "",
        "report_path": "",
        "error": "",
        "created_at": _now_iso(),
        "started_at": "",
        "finished_at": "",
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
        job.update(fields)


def get_evaluation_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return _clone_job(_jobs.get(job_id))


def list_evaluation_jobs(limit: int = 10) -> Dict[str, Any]:
    with _jobs_lock:
        jobs = [_clone_job(_jobs.get(job_id)) for job_id in _job_order[:limit]]
    return {"jobs": [job for job in jobs if job is not None]}


def run_evaluation_job(job_id: str) -> None:
    job = get_evaluation_job(job_id)
    if job is None:
        return

    dataset_path = str(job.get("dataset_path") or "").strip()
    variants = list(job.get("variants") or list(DEFAULT_VARIANTS))
    output_path = str(job.get("output_path") or "").strip() or None
    _update_job(
        job_id,
        status=JOB_STATUS_RUNNING,
        started_at=_now_iso(),
        progress_message="正在加载数据集",
        error="",
    )

    def on_progress(event: Dict[str, Any]) -> None:
        stage = str(event.get("stage") or "")
        if stage == "dataset_loaded":
            _update_job(
                job_id,
                progress_message=f"已加载数据集，共 {int(event.get('case_count') or 0)} 条样本",
                case_count=int(event.get("case_count") or 0),
                total_variants=int(event.get("total_variants") or len(variants)),
            )
        elif stage == "variant_started":
            current_variant = str(event.get("variant_name") or "")
            index = int(event.get("variant_index") or 0)
            total = int(event.get("total_variants") or len(variants))
            _update_job(
                job_id,
                current_variant=current_variant,
                completed_variants=max(index - 1, 0),
                total_variants=total,
                progress_message=f"正在评测 {current_variant} ({index}/{total})",
            )
        elif stage == "variant_completed":
            current_variant = str(event.get("variant_name") or "")
            index = int(event.get("variant_index") or 0)
            total = int(event.get("total_variants") or len(variants))
            _update_job(
                job_id,
                current_variant=current_variant,
                completed_variants=index,
                total_variants=total,
                progress_message=f"已完成 {current_variant} ({index}/{total})",
            )
        elif stage == "report_ready":
            _update_job(job_id, progress_message="评测完成，正在写入报告")
        elif stage == "report_saved":
            report_path = str(event.get("output_path") or "")
            _update_job(
                job_id,
                progress_message="报告已生成",
                report_path=report_path,
                report_id=Path(report_path).name if report_path else "",
            )

    try:
        result = evaluate_variants_to_file(
            dataset_path=dataset_path,
            variant_names=variants,
            output_path=output_path,
            progress_callback=on_progress,
        )
        report_path = str(result.get("output_path") or "")
        _update_job(
            job_id,
            status=JOB_STATUS_COMPLETED,
            finished_at=_now_iso(),
            completed_variants=len(variants),
            total_variants=len(variants),
            current_variant="",
            progress_message="评测完成",
            report_path=report_path,
            report_id=Path(report_path).name if report_path else "",
        )
    except Exception as exc:
        _update_job(
            job_id,
            status=JOB_STATUS_FAILED,
            finished_at=_now_iso(),
            progress_message="评测失败",
            error=str(exc),
        )
