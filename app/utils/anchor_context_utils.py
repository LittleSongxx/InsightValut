import re
from typing import Any, Dict, List, Sequence

from app.clients.milvus_schema import extract_chunk_id

FALLBACK_MESSAGE = "抱歉，资料中未提供"

_PUNCT_TRIM = " \t\r\n，。；：、,.!?！？（）()[]【】「」『』“”\"'`*#"
_STOP_ANCHOR_TERMS = {
    "说明",
    "使用",
    "功能",
    "步骤",
    "相关功能",
    "相关步骤",
    "注意事项",
    "限制",
    "区别",
    "使用场景",
}
_GENERIC_SECTION_TERMS = {
    "目录",
    "概述",
    "总览",
    "简介",
    "说明",
    "用户指南",
    "注意",
    "责任限制",
    "环境保护",
}


def clean_anchor_text(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(_PUNCT_TRIM)
    if limit and len(text) > limit:
        return text[:limit].rstrip(_PUNCT_TRIM)
    return text


def normalize_anchor_term(value: Any) -> str:
    text = clean_anchor_text(value).lower()
    text = re.sub(r"[#*_`]+", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip(_PUNCT_TRIM)


_GENERIC_SECTION_KEYS = {
    normalize_anchor_term(term) for term in _GENERIC_SECTION_TERMS
}


def _dedupe(values: Sequence[Any], limit: int = 12) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = clean_anchor_text(value, 80)
        key = normalize_anchor_term(text)
        if not key or key in seen or key in _STOP_ANCHOR_TERMS:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def build_section_path(chunk: Dict[str, Any]) -> str:
    parts = [
        chunk.get("file_title"),
        chunk.get("parent_title"),
        chunk.get("title"),
    ]
    cleaned = _dedupe(parts, limit=4)
    part = chunk.get("part")
    if part not in (None, "", 0, "0"):
        cleaned.append(f"part {part}")
    return " > ".join(cleaned)


def extract_anchor_terms_from_doc(chunk: Dict[str, Any]) -> List[str]:
    candidates: List[str] = [
        chunk.get("item_name"),
        chunk.get("file_title"),
        chunk.get("parent_title"),
        chunk.get("title"),
        chunk.get("section_path"),
    ]
    content = str(chunk.get("content") or "")
    for heading in re.findall(r"^\s{0,3}#{1,6}\s+(.+)$", content, flags=re.M):
        candidates.append(heading)
    for quoted in re.findall(r"[“\"'‘]([^”\"'’]{2,60})[”\"'’]", content):
        candidates.append(quoted)
    return _dedupe(candidates, limit=16)


def build_chunk_context(
    chunk: Dict[str, Any],
    previous_chunk: Dict[str, Any] | None = None,
    next_chunk: Dict[str, Any] | None = None,
) -> str:
    parts = [
        f"商品：{clean_anchor_text(chunk.get('item_name'))}",
        f"文件：{clean_anchor_text(chunk.get('file_title'))}",
        f"章节路径：{build_section_path(chunk)}",
    ]
    previous_title = clean_anchor_text((previous_chunk or {}).get("title"))
    next_title = clean_anchor_text((next_chunk or {}).get("title"))
    if previous_title:
        parts.append(f"上一节：{previous_title}")
    if next_title:
        parts.append(f"下一节：{next_title}")
    return "\n".join(part for part in parts if part and not part.endswith("："))


def build_contextual_chunk_fields(
    chunk: Dict[str, Any],
    previous_chunk: Dict[str, Any] | None = None,
    next_chunk: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    section_path = build_section_path(chunk)
    chunk_context = build_chunk_context(chunk, previous_chunk, next_chunk)
    content = str(chunk.get("content") or "").strip()
    title = clean_anchor_text(chunk.get("title"))
    parent_title = clean_anchor_text(chunk.get("parent_title"))
    item_name = clean_anchor_text(chunk.get("item_name"))
    file_title = clean_anchor_text(chunk.get("file_title"))
    embedding_text = "\n".join(
        part
        for part in [
            f"商品：{item_name}" if item_name else "",
            f"章节：{section_path}" if section_path else "",
            chunk_context,
            content,
        ]
        if part
    )
    bm25_text = "\n".join(
        part
        for part in [
            item_name,
            file_title,
            parent_title,
            title,
            section_path,
            chunk_context,
            content,
        ]
        if part
    )
    return {
        "section_path": section_path,
        "chunk_context": chunk_context,
        "embedding_text": embedding_text,
        "bm25_text": bm25_text,
        "anchor_terms": extract_anchor_terms_from_doc(
            {
                **chunk,
                "section_path": section_path,
            }
        ),
    }


def apply_contextual_fields_to_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            continue
        previous_chunk = chunks[index - 1] if index > 0 else None
        next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None
        item = dict(chunk)
        item.update(build_contextual_chunk_fields(item, previous_chunk, next_chunk))
        enriched.append(item)
    return enriched


def extract_query_anchor_targets(query: str, item_names: Sequence[str] | None = None) -> List[str]:
    text = clean_anchor_text(query)
    for item_name in item_names or []:
        item = clean_anchor_text(item_name)
        if item:
            text = text.replace(item, " ")

    candidates: List[str] = []
    quoted_candidates = re.findall(r"[“\"'‘]([^”\"'’]{2,80})[”\"'’]", text)
    for match in quoted_candidates:
        candidates.append(match)

    if not quoted_candidates:
        patterns = [
            r"中(.{2,40}?)涉及哪些相关功能或步骤",
            r"使用(.{2,40}?)时有哪些注意事项或限制",
            r"的(.{2,40}?)主要解决什么问题",
            r"关于(.{2,40}?)的说明",
            r"中，?(.{2,40}?)应该怎么操作",
            r"中(.{2,40}?)分别讲什么",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                candidates.append(match)

    cleaned: List[str] = []
    for candidate in candidates:
        value = re.split(r"[，。；：、,.!?！？]", clean_anchor_text(candidate))[0]
        value = re.sub(r"^(和|与|及|在|中|关于|使用)", "", value).strip(_PUNCT_TRIM)
        if value:
            cleaned.append(value)
    return _dedupe(cleaned, limit=6)


def classify_router_query_family(
    query: str,
    query_type: str = "general",
    targets: Sequence[str] | None = None,
) -> str:
    text = clean_anchor_text(query)
    target_count = len([target for target in targets or [] if clean_anchor_text(target)])
    if target_count >= 2 and (
        query_type == "comparison"
        or any(marker in text for marker in ("分别", "区别", "差异", "对比", "比较"))
    ):
        return "comparison"
    if re.search(r"涉及哪些相关功能或步骤", text):
        return "section_summary"
    if re.search(r"主要解决什么问题|关于.+的说明|使用.+时有哪些注意事项或限制", text):
        return "section_lookup"
    if query_type == "navigation":
        return "procedure_lookup"
    if query_type == "relation":
        return "multi_hop_relation"
    return "general"


def anchor_text_for_doc(doc: Dict[str, Any]) -> str:
    parts = [
        doc.get("item_name"),
        doc.get("file_title"),
        doc.get("parent_title"),
        doc.get("title"),
        doc.get("section_path"),
        " ".join(doc.get("anchor_terms") or []),
        doc.get("bm25_text"),
        doc.get("content"),
    ]
    return "\n".join(clean_anchor_text(part) for part in parts if clean_anchor_text(part))


def score_anchor_doc(query: str, targets: Sequence[str], doc: Dict[str, Any]) -> float:
    doc_blob = normalize_anchor_term(anchor_text_for_doc(doc))
    if not doc_blob:
        return 0.0
    score = 0.0
    normalized_targets = [normalize_anchor_term(target) for target in targets or [] if normalize_anchor_term(target)]
    title_blob = normalize_anchor_term(
        " ".join(
            str(doc.get(field) or "")
            for field in ("title", "parent_title", "section_path", "file_title")
        )
    )
    for target in normalized_targets:
        if target and target in title_blob:
            score += 8.0
        elif target and target in doc_blob:
            score += 3.0
    query_terms = extract_query_anchor_targets(query)
    for term in query_terms:
        normalized = normalize_anchor_term(term)
        if normalized and normalized in title_blob:
            score += 2.0
    if normalized_targets and all(target in doc_blob for target in normalized_targets):
        score += 2.0
    return score


def _is_generic_heading(value: Any) -> bool:
    heading = normalize_anchor_term(value)
    if not heading:
        return True
    if heading in _GENERIC_SECTION_KEYS:
        return True
    return any(term in heading for term in ("目录", "总览", "概述", "注意事项"))


def _target_match_score(doc: Dict[str, Any], target: str) -> float:
    target_key = normalize_anchor_term(target)
    if not target_key:
        return -1.0

    title = clean_anchor_text(doc.get("title"))
    parent_title = clean_anchor_text(doc.get("parent_title"))
    section_path = clean_anchor_text(doc.get("section_path"))
    file_title = clean_anchor_text(doc.get("file_title"))
    content = clean_anchor_text(doc.get("text") or doc.get("content"), 400)
    anchor_terms = [clean_anchor_text(term) for term in (doc.get("anchor_terms") or [])]
    source = clean_anchor_text(doc.get("source"))

    title_key = normalize_anchor_term(title)
    parent_key = normalize_anchor_term(parent_title)
    section_key = normalize_anchor_term(section_path)
    file_key = normalize_anchor_term(file_title)
    content_key = normalize_anchor_term(content)
    anchor_keys = [normalize_anchor_term(term) for term in anchor_terms if normalize_anchor_term(term)]

    score = 0.0
    if title_key == target_key:
        score += 18.0
    elif title_key and target_key in title_key:
        score += 14.0

    if target_key and target_key in anchor_keys:
        score += 10.0
    if section_key and target_key in section_key:
        score += 8.0
    if parent_key and target_key in parent_key:
        score += 6.0
    if content_key and target_key in content_key:
        score += 4.0

    if source == "anchor":
        score += 2.0
    elif source == "bm25":
        score += 1.0

    if _is_generic_heading(title):
        score -= 4.0
    if _is_generic_heading(parent_title):
        score -= 2.0
    if file_key and target_key == file_key:
        score -= 3.0

    return score


def build_target_coverage(
    docs: Sequence[Dict[str, Any]],
    targets: Sequence[str],
) -> Dict[str, Any]:
    target_rows = []
    covered = 0
    for target in targets or []:
        target_key = normalize_anchor_term(target)
        matched_docs = []
        if target_key:
            for doc in docs or []:
                if target_key in normalize_anchor_term(anchor_text_for_doc(doc)):
                    matched_docs.append(
                        {
                            "chunk_id": str(doc.get("chunk_id") or doc.get("doc_id") or ""),
                            "title": clean_anchor_text(doc.get("title") or doc.get("section_path")),
                        }
                    )
        if matched_docs:
            covered += 1
        target_rows.append(
            {
                "target": clean_anchor_text(target),
                "covered": bool(matched_docs),
                "chunks": matched_docs[:3],
            }
        )
    total = len(target_rows)
    return {
        "targets": target_rows,
        "covered_targets": covered,
        "target_count": total,
        "coverage_rate": round(covered / total, 4) if total else 1.0,
        "missing_targets": [row["target"] for row in target_rows if not row["covered"]],
    }


def build_evidence_pack(
    question: str,
    docs: Sequence[Dict[str, Any]],
    *,
    query_type: str = "general",
    targets: Sequence[str] | None = None,
    max_chars: int = 7000,
) -> Dict[str, Any]:
    resolved_targets = list(targets or extract_query_anchor_targets(question))
    coverage = build_target_coverage(docs, resolved_targets)
    lines: List[str] = []
    used = 0
    for index, doc in enumerate(docs or [], start=1):
        text = clean_anchor_text(doc.get("text") or doc.get("content"), 900)
        if not text:
            continue
        chunk_id = str(doc.get("chunk_id") or doc.get("doc_id") or "")
        title = clean_anchor_text(
            doc.get("section_path")
            or doc.get("title")
            or doc.get("document_title")
            or doc.get("file_title")
        )
        row = f"[{index}] chunk_id={chunk_id or '-'} title={title or '-'}\n{text}"
        if used + len(row) > max_chars:
            break
        lines.append(row)
        used += len(row) + 2
    return {
        "query_type": query_type,
        "targets": resolved_targets,
        "target_coverage": coverage,
        "context_budget_chars": max_chars,
        "used_chars": used,
        "text": "\n\n".join(lines) if lines else "无参考内容",
        "doc_count": len(lines),
        "fallback_message": FALLBACK_MESSAGE,
    }


def summarize_evidence_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    coverage = pack.get("target_coverage") or {}
    return {
        "doc_count": int(pack.get("doc_count") or 0),
        "target_count": int(coverage.get("target_count") or 0),
        "covered_targets": int(coverage.get("covered_targets") or 0),
        "coverage_rate": coverage.get("coverage_rate", 1.0),
        "missing_targets": coverage.get("missing_targets") or [],
        "used_chars": int(pack.get("used_chars") or 0),
        "context_budget_chars": int(pack.get("context_budget_chars") or 0),
    }


def reorder_docs_for_target_coverage(
    docs: Sequence[Dict[str, Any]],
    targets: Sequence[str],
) -> List[Dict[str, Any]]:
    if not targets:
        return list(docs or [])
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for target in targets:
        target_key = normalize_anchor_term(target)
        if not target_key:
            continue
        best_doc = None
        best_score = float("-inf")
        for index, doc in enumerate(docs or []):
            doc_id = str(extract_chunk_id(doc) or doc.get("doc_id") or id(doc))
            if doc_id in selected_ids:
                continue
            if target_key not in normalize_anchor_term(anchor_text_for_doc(doc)):
                continue
            score = _target_match_score(doc, target) - (index * 0.01)
            if score > best_score:
                best_doc = doc
                best_score = score
        if best_doc is not None:
            doc_id = str(extract_chunk_id(best_doc) or best_doc.get("doc_id") or id(best_doc))
            selected.append(best_doc)
            selected_ids.add(doc_id)
    for doc in docs or []:
        doc_id = str(extract_chunk_id(doc) or doc.get("doc_id") or id(doc))
        if doc_id not in selected_ids:
            selected.append(doc)
            selected_ids.add(doc_id)
    return selected
