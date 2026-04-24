import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.clients.milvus_schema import (
    CHUNKS_OUTPUT_FIELDS,
    extract_chunk_id,
    normalize_entity_for_business_id,
)
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
    query_chunks_by_filter,
)
from app.conf.query_threshold_config import query_threshold_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.utils.bm25_utils import rank_documents_bm25
from app.utils.anchor_context_utils import (
    anchor_text_for_doc,
    extract_query_anchor_targets,
    score_anchor_doc,
)
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.query_cache_utils import normalize_cache_sequence, query_cache_get, query_cache_set


def _sorted_cache_strings(values: Sequence[Any] | None) -> list[str]:
    return sorted(
        {
            str(value).strip()
            for value in normalize_cache_sequence(values)
            if str(value).strip()
        }
    )


def build_item_name_filter_expr(item_names: Optional[Sequence[str]]) -> str:
    values = [escape_milvus_string(str(name).strip()) for name in (item_names or []) if str(name).strip()]
    if not values:
        return ""
    quoted = ", ".join(f'"{value}"' for value in values)
    return f"item_name in [{quoted}]"


def _get_chunks_collection_name(collection_name: Optional[str] = None) -> str:
    return collection_name or os.environ.get("CHUNKS_COLLECTION") or ""


def run_embedding_hybrid_search(
    query_text: str,
    item_names: Optional[Sequence[str]] = None,
    *,
    collection_name: Optional[str] = None,
    req_limit: Optional[int] = None,
    top_k: Optional[int] = None,
    output_fields: Optional[List[str]] = None,
    ranker_weights: Optional[Tuple[float, float]] = None,
    norm_score: bool = True,
) -> List[Dict[str, Any]]:
    if not query_text:
        return []

    target_collection = _get_chunks_collection_name(collection_name)
    if not target_collection:
        logger.error("检索失败：未配置集合名称")
        return []

    cache_descriptor = {
        "collection": target_collection,
        "query_text": query_text,
        "item_names": _sorted_cache_strings(item_names),
        "req_limit": req_limit if req_limit is not None else query_threshold_config.embedding_req_limit,
        "top_k": top_k if top_k is not None else query_threshold_config.embedding_top_k,
        "output_fields": normalize_cache_sequence(output_fields or list(CHUNKS_OUTPUT_FIELDS)),
        "ranker_weights": list(ranker_weights or (query_threshold_config.hybrid_dense_weight, query_threshold_config.hybrid_sparse_weight)),
        "norm_score": bool(norm_score),
    }
    cached = query_cache_get("retrieval_embedding", cache_descriptor)
    if isinstance(cached, list):
        return cached

    client = get_milvus_client()
    if not client:
        logger.error("检索失败：Milvus 客户端不可用")
        return []

    cfg = query_threshold_config
    embeddings = generate_embeddings([query_text])
    dense_vector = embeddings.get("dense", [None])[0]
    sparse_vector = embeddings.get("sparse", [None])[0]
    if dense_vector is None or sparse_vector is None:
        return []

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=build_item_name_filter_expr(item_names),
        limit=req_limit if req_limit is not None else cfg.embedding_req_limit,
    )
    weights = ranker_weights or (cfg.hybrid_dense_weight, cfg.hybrid_sparse_weight)
    res = hybrid_search(
        client=client,
        collection_name=target_collection,
        reqs=reqs,
        ranker_weights=weights,
        norm_score=norm_score,
        limit=top_k if top_k is not None else cfg.embedding_top_k,
        output_fields=output_fields or list(CHUNKS_OUTPUT_FIELDS),
    )
    hits = res[0] if res else []
    query_cache_set("retrieval_embedding", cache_descriptor, hits)
    return hits


def _bm25_document_text(doc: Dict[str, Any]) -> str:
    parts = [
        str(doc.get("item_name") or "").strip(),
        str(doc.get("title") or "").strip(),
        str(doc.get("parent_title") or "").strip(),
        str(doc.get("part") or "").strip(),
        str(doc.get("file_title") or "").strip(),
        str(doc.get("section_path") or "").strip(),
        " ".join(str(term or "").strip() for term in (doc.get("anchor_terms") or [])),
        str(doc.get("bm25_text") or "").strip(),
        str(doc.get("content") or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def run_bm25_search(
    query_text: str,
    item_names: Optional[Sequence[str]] = None,
    *,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
    candidate_limit: Optional[int] = None,
    output_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not query_text:
        return []

    target_collection = _get_chunks_collection_name(collection_name)
    if not target_collection:
        logger.error("BM25 检索失败：未配置集合名称")
        return []

    cache_descriptor = {
        "collection": target_collection,
        "query_text": query_text,
        "item_names": _sorted_cache_strings(item_names),
        "top_k": top_k if top_k is not None else query_threshold_config.bm25_top_k,
        "candidate_limit": candidate_limit
        if candidate_limit is not None
        else query_threshold_config.bm25_candidate_limit,
        "output_fields": normalize_cache_sequence(output_fields or list(CHUNKS_OUTPUT_FIELDS)),
    }
    cached = query_cache_get("retrieval_bm25", cache_descriptor)
    if isinstance(cached, list):
        return cached

    client = get_milvus_client()
    if not client:
        logger.error("BM25 检索失败：Milvus 客户端不可用")
        return []

    cfg = query_threshold_config
    docs = query_chunks_by_filter(
        client=client,
        collection_name=target_collection,
        filter_expr=build_item_name_filter_expr(item_names),
        output_fields=output_fields or list(CHUNKS_OUTPUT_FIELDS),
        limit=candidate_limit if candidate_limit is not None else cfg.bm25_candidate_limit,
    )
    if not docs:
        return []

    ranked_docs = rank_documents_bm25(
        query_text=query_text,
        documents=docs,
        text_getter=_bm25_document_text,
        top_k=top_k if top_k is not None else cfg.bm25_top_k,
        k1=cfg.bm25_k1,
        b=cfg.bm25_b,
    )
    results: List[Dict[str, Any]] = []
    for doc, score in ranked_docs:
        entity = normalize_entity_for_business_id(dict(doc))
        chunk_id = extract_chunk_id(entity)
        results.append(
            {
                "entity": entity,
                "distance": score,
                "id": chunk_id,
            }
        )
    query_cache_set("retrieval_bm25", cache_descriptor, results)
    return results


def run_anchor_search(
    query_text: str,
    item_names: Optional[Sequence[str]] = None,
    *,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
    candidate_limit: Optional[int] = None,
    output_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not query_text:
        return []

    targets = extract_query_anchor_targets(query_text, item_names=item_names)
    if not targets:
        return []

    target_collection = _get_chunks_collection_name(collection_name)
    if not target_collection:
        logger.error("Anchor 检索失败：未配置集合名称")
        return []

    cache_descriptor = {
        "collection": target_collection,
        "query_text": query_text,
        "item_names": _sorted_cache_strings(item_names),
        "targets": _sorted_cache_strings(targets),
        "top_k": top_k if top_k is not None else query_threshold_config.anchor_top_k,
        "candidate_limit": candidate_limit
        if candidate_limit is not None
        else query_threshold_config.anchor_candidate_limit,
        "output_fields": normalize_cache_sequence(output_fields or list(CHUNKS_OUTPUT_FIELDS)),
    }
    cached = query_cache_get("retrieval_anchor", cache_descriptor)
    if isinstance(cached, list):
        return cached

    client = get_milvus_client()
    if not client:
        logger.error("Anchor 检索失败：Milvus 客户端不可用")
        return []

    cfg = query_threshold_config
    docs = query_chunks_by_filter(
        client=client,
        collection_name=target_collection,
        filter_expr=build_item_name_filter_expr(item_names),
        output_fields=output_fields or list(CHUNKS_OUTPUT_FIELDS),
        limit=candidate_limit if candidate_limit is not None else cfg.anchor_candidate_limit,
    )
    if not docs:
        return []

    scored_docs: List[tuple[Dict[str, Any], float]] = []
    for doc in docs:
        entity = normalize_entity_for_business_id(dict(doc))
        score = score_anchor_doc(query_text, targets, entity)
        if score <= 0:
            continue
        # Secondary signal: longer exact anchor text match wins over broad content-only hits.
        if any(target and target in anchor_text_for_doc(entity) for target in targets):
            score += 0.25
        scored_docs.append((entity, score))

    scored_docs.sort(key=lambda item: item[1], reverse=True)
    results: List[Dict[str, Any]] = []
    for entity, score in scored_docs[: top_k if top_k is not None else cfg.anchor_top_k]:
        chunk_id = extract_chunk_id(entity)
        results.append(
            {
                "entity": entity,
                "distance": score,
                "id": chunk_id,
            }
        )
    query_cache_set("retrieval_anchor", cache_descriptor, results)
    return results
