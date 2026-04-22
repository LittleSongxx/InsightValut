import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.path_util import PROJECT_ROOT

EVAL_REPORT_PATTERN = "unified_rag_eval_*.json"


def _primary_eval_output_dir() -> Path:
    return Path(PROJECT_ROOT) / "reports" / "eval"


def _legacy_eval_output_dir() -> Path:
    return Path(PROJECT_ROOT) / "output" / "eval"


def _eval_output_dirs() -> List[Path]:
    return [_primary_eval_output_dir(), _legacy_eval_output_dir()]


def _report_mtime(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        .astimezone()
        .isoformat()
    )


def _list_report_paths() -> List[Path]:
    deduped: Dict[str, Path] = {}
    for output_dir in _eval_output_dirs():
        if not output_dir.exists():
            continue
        for path in sorted(
            output_dir.glob(EVAL_REPORT_PATTERN),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            deduped.setdefault(path.name, path)
    return sorted(
        deduped.values(),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    return value


def _dataset_name(report: Dict[str, Any]) -> str:
    dataset_path = str(report.get("dataset_path") or "").strip()
    if not dataset_path:
        return ""
    return Path(dataset_path).name


def _build_report_meta(path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    final_summary = report.get("final_system_metrics") or {}
    return _sanitize_for_json({
        "report_id": path.name,
        "file_name": path.name,
        "generated_at": str(report.get("generated_at") or _report_mtime(path)),
        "updated_at": _report_mtime(path),
        "dataset_path": str(report.get("dataset_path") or ""),
        "dataset_name": _dataset_name(report),
        "case_count": int(report.get("case_count") or 0),
        "final_variant": str(report.get("final_variant") or ""),
        "variants": list((report.get("variants") or {}).keys()),
        "headline_metrics": final_summary.get("headline_metrics") or {},
        "size_bytes": path.stat().st_size,
    })


def _strip_case_results(report: Dict[str, Any]) -> Dict[str, Any]:
    variants_payload = {}
    for variant_name, payload in (report.get("variants") or {}).items():
        variant_data = dict(payload or {})
        variant_data.pop("case_results", None)
        variants_payload[variant_name] = variant_data
    return _sanitize_for_json({
        "generated_at": report.get("generated_at"),
        "dataset_path": report.get("dataset_path"),
        "case_count": int(report.get("case_count") or 0),
        "final_variant": report.get("final_variant"),
        "final_system_metrics": report.get("final_system_metrics") or {},
        "variants": variants_payload,
        "comparisons": report.get("comparisons") or {},
    })


def list_evaluation_reports() -> Dict[str, Any]:
    reports = []
    for path in _list_report_paths():
        try:
            report = _load_json(path)
        except Exception:
            continue
        reports.append(_build_report_meta(path, report))
    latest_report_id = reports[0]["report_id"] if reports else None
    return {"reports": reports, "latest_report_id": latest_report_id}


def _resolve_report_path(report_id: str) -> Optional[Path]:
    safe_name = Path(str(report_id or "")).name
    if not safe_name or safe_name != report_id:
        return None
    for output_dir in _eval_output_dirs():
        path = output_dir / safe_name
        if path.exists() and path.is_file():
            return path
    return None


def get_evaluation_report(report_id: str) -> Optional[Dict[str, Any]]:
    path = _resolve_report_path(report_id)
    if path is None:
        return None
    report = _load_json(path)
    return {
        "meta": _build_report_meta(path, report),
        "report": _strip_case_results(report),
    }


def get_latest_evaluation_report() -> Optional[Dict[str, Any]]:
    report_paths = _list_report_paths()
    if not report_paths:
        return None
    latest_path = report_paths[0]
    report = _load_json(latest_path)
    return {
        "meta": _build_report_meta(latest_path, report),
        "report": _strip_case_results(report),
    }


def delete_evaluation_report(report_id: str) -> Dict[str, Any]:
    safe_name = Path(str(report_id or "")).name
    if not safe_name or safe_name != report_id:
        raise ValueError("非法的评测报告 ID")

    candidate_paths = [
        output_dir / safe_name
        for output_dir in _eval_output_dirs()
        if (output_dir / safe_name).exists() and (output_dir / safe_name).is_file()
    ]
    if not candidate_paths:
        raise FileNotFoundError("evaluation report not found")

    primary_path = candidate_paths[0]
    meta: Dict[str, Any]
    try:
        meta = _build_report_meta(primary_path, _load_json(primary_path))
    except Exception:
        meta = {
            "report_id": safe_name,
            "file_name": safe_name,
            "deleted_paths": [str(path.resolve()) for path in candidate_paths],
        }

    deleted_paths: List[str] = []
    for path in candidate_paths:
        path.unlink(missing_ok=True)
        deleted_paths.append(str(path.resolve()))

    return _sanitize_for_json(
        {
            "ok": True,
            "report_id": safe_name,
            "deleted_paths": deleted_paths,
            "deleted_count": len(deleted_paths),
            "meta": meta,
        }
    )
