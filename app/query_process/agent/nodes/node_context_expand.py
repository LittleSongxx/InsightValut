import sys
from typing import Any, Dict, List

from app.clients.neo4j_graph_utils import expand_chunk_context
from app.conf.query_threshold_config import query_threshold_config
from app.core.logger import logger
from app.query_process.agent.agentic_utils import is_agentic_feature_enabled
from app.utils.task_utils import add_done_task, add_running_task


def _primary_chunk_id(doc: Dict[str, Any]) -> str:
    chunk_id = doc.get("chunk_id") or doc.get("id")
    if chunk_id:
        return str(chunk_id)
    evidence_chunk_ids = doc.get("evidence_chunk_ids") or []
    if evidence_chunk_ids:
        return str(evidence_chunk_ids[0])
    return ""


def node_context_expand(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("---node_context_expand (命中上下文扩展) 开始处理---")
    add_running_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )

    rrf_chunks = list(state.get("rrf_chunks") or [])
    if not rrf_chunks:
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {"rrf_chunks": rrf_chunks, "context_expansion_summary": {"expanded_docs": 0}}

    if not is_agentic_feature_enabled(state, "context_expansion"):
        logger.info("node_context_expand: 上下文扩展已被功能开关关闭")
        add_done_task(
            state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
        )
        return {
            "rrf_chunks": rrf_chunks,
            "context_expansion_summary": {
                "expanded_docs": 0,
                "candidate_docs": len(rrf_chunks),
                "enabled": False,
            },
        }

    top_k = min(query_threshold_config.context_expand_top_k, len(rrf_chunks))
    candidate_ids: List[str] = []
    for doc in rrf_chunks[:top_k]:
        if not isinstance(doc, dict):
            continue
        source = str(doc.get("source") or "local")
        if source == "web":
            continue
        chunk_id = _primary_chunk_id(doc)
        if chunk_id and chunk_id not in candidate_ids:
            candidate_ids.append(chunk_id)

    expansions = expand_chunk_context(
        candidate_ids,
        neighbor_limit=query_threshold_config.context_expand_neighbor_limit,
        evidence_limit=query_threshold_config.context_expand_evidence_limit,
        max_chars=query_threshold_config.context_expand_max_chars,
    )

    expanded_docs = 0
    affected_chunk_ids: List[str] = []
    for index, doc in enumerate(rrf_chunks):
        if not isinstance(doc, dict):
            continue
        chunk_id = _primary_chunk_id(doc)
        expansion = expansions.get(chunk_id)
        if not expansion:
            continue

        base_text = str(doc.get("content") or doc.get("text") or "").strip()
        expanded_text = str(expansion.get("expanded_text") or "").strip()
        if expanded_text:
            merged_text = base_text
            if expanded_text not in merged_text:
                merged_text = (
                    f"{base_text}\n\n【命中上下文扩展】\n{expanded_text}"
                    if base_text
                    else expanded_text
                )
            doc["expanded_text"] = merged_text
        else:
            doc["expanded_text"] = base_text

        doc["expanded_chunk_ids"] = expansion.get("expanded_chunk_ids") or [chunk_id]
        doc["context_expanded"] = True
        doc["section_title"] = doc.get("section_title") or expansion.get("section_title") or ""
        doc["document_title"] = doc.get("document_title") or expansion.get("document_title") or ""
        entity_names = list(doc.get("graph_entities") or [])
        for name in expansion.get("entity_names") or []:
            if name and name not in entity_names:
                entity_names.append(name)
        doc["graph_entities"] = entity_names
        image_urls = list(doc.get("image_urls") or [])
        for url in expansion.get("image_urls") or []:
            if url and url not in image_urls:
                image_urls.append(url)
        doc["image_urls"] = image_urls

        rrf_chunks[index] = doc
        expanded_docs += 1
        if chunk_id and chunk_id not in affected_chunk_ids:
            affected_chunk_ids.append(chunk_id)

    summary = {
        "enabled": True,
        "candidate_docs": len(candidate_ids),
        "expanded_docs": expanded_docs,
        "affected_chunk_ids": affected_chunk_ids,
    }
    add_done_task(
        state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")
    )
    logger.info(
        f"node_context_expand: 完成，候选 {len(candidate_ids)} 条，实际扩展 {expanded_docs} 条"
    )
    return {"rrf_chunks": rrf_chunks, "context_expansion_summary": summary}
