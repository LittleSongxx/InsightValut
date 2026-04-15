import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping


STABLE_CHUNK_ID_VERSION = "v1"
STABLE_CHUNK_ID_PREFIX = f"stc_{STABLE_CHUNK_ID_VERSION}_"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_part(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _normalize_image_urls(value: Any) -> List[str]:
    urls: List[str] = []
    for item in value or []:
        text = _normalize_text(item)
        if text and text not in urls:
            urls.append(text)
    return urls


def build_chunk_identity_payload(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "item_name": _normalize_text(chunk.get("item_name")),
        "file_title": _normalize_text(chunk.get("file_title")),
        "parent_title": _normalize_text(chunk.get("parent_title")),
        "title": _normalize_text(chunk.get("title")),
        "part": _normalize_part(chunk.get("part")),
        "content": _normalize_text(chunk.get("content")),
        "image_urls": _normalize_image_urls(chunk.get("image_urls")),
    }


def generate_stable_chunk_id(chunk: Mapping[str, Any]) -> str:
    payload = build_chunk_identity_payload(chunk)
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload_text.encode("utf-8")).hexdigest()[:24]
    return f"{STABLE_CHUNK_ID_PREFIX}{digest}"


def is_stable_chunk_id(value: Any) -> bool:
    return str(value or "").strip().startswith(STABLE_CHUNK_ID_PREFIX)


def ensure_chunk_id(chunk: Dict[str, Any]) -> str:
    existing_stable = str(chunk.get("stable_chunk_id") or "").strip()
    if not existing_stable and is_stable_chunk_id(chunk.get("chunk_id")):
        existing_stable = str(chunk.get("chunk_id")).strip()

    stable_chunk_id = existing_stable or generate_stable_chunk_id(chunk)
    chunk["chunk_id"] = stable_chunk_id
    chunk["stable_chunk_id"] = stable_chunk_id
    return stable_chunk_id


def ensure_chunk_ids(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        if isinstance(chunk, dict):
            ensure_chunk_id(chunk)
            normalized.append(chunk)
    return normalized


def business_chunk_id(entity: Mapping[str, Any]) -> str:
    stable_chunk_id = str(entity.get("stable_chunk_id") or "").strip()
    if stable_chunk_id:
        return stable_chunk_id

    chunk_id = str(entity.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id

    return str(entity.get("id") or "").strip()


def storage_chunk_id(entity: Mapping[str, Any]) -> str:
    chunk_id = str(entity.get("chunk_id") or "").strip()
    if chunk_id and not is_stable_chunk_id(chunk_id):
        return chunk_id

    doc_id = str(entity.get("id") or "").strip()
    if doc_id and not is_stable_chunk_id(doc_id):
        return doc_id
    return ""


def normalize_entity_chunk_fields(
    entity: Dict[str, Any], *, legacy_field: str = "storage_chunk_id"
) -> Dict[str, Any]:
    if not isinstance(entity, dict):
        return entity

    normalized = dict(entity)
    stable_chunk_id = str(normalized.get("stable_chunk_id") or "").strip()
    public_chunk_id = business_chunk_id(normalized)
    raw_storage_chunk_id = storage_chunk_id(normalized)

    if public_chunk_id:
        normalized["chunk_id"] = public_chunk_id
        normalized["id"] = public_chunk_id

    if stable_chunk_id:
        normalized["stable_chunk_id"] = stable_chunk_id

    if raw_storage_chunk_id and raw_storage_chunk_id != public_chunk_id:
        normalized.setdefault(legacy_field, raw_storage_chunk_id)

    return normalized
