import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence

from app.clients.neo4j_utils import ensure_indexes, get_neo4j_driver, query_chunks_by_product
from app.core.logger import logger
from app.import_process.agent.graph_extract_utils import build_graph_payload
from app.query_process.agent.graph_query_utils import extract_focus_terms

_ALLOWED_ENTITY_LABELS = {
    "Step",
    "Parameter",
    "Component",
    "Fault",
    "Cause",
    "Solution",
    "Warning",
}

_ALLOWED_RELATION_TYPES = {
    "HAS_STEP",
    "HAS_PARAMETER",
    "HAS_COMPONENT",
    "HAS_FAULT",
    "HAS_CAUSE",
    "HAS_SOLUTION",
    "HAS_WARNING",
    "CAUSED_BY",
    "RESOLVED_BY",
    "RELATED_TO",
    "NEXT_STEP",
}


def _get_database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _slug(text: Any) -> str:
    cleaned = _clean_text(text).lower()
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:120] or "unknown"


def _product_key(item_name: str) -> str:
    return f"product::{_slug(item_name)}"


def ensure_graph_indexes() -> None:
    ensure_indexes()
    driver = get_neo4j_driver()
    with driver.session(database=_get_database()) as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS "
            "FOR (n:KGNode) REQUIRE n.node_key IS UNIQUE"
        )
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (n:KGNode) ON (n.product_name)"
        )
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)"
        )
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.title)"
        )
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (e:KGEntity) ON (e.name)"
        )
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (e:KGEntity) ON (e.value)"
        )


def _dedupe_rows(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows or []:
        marker = tuple(row.get(field) for field in key_fields)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)
    return deduped


def import_chunks_to_graph(item_name: str, chunks: List[Dict[str, Any]]) -> int:
    if not item_name or not chunks:
        logger.warning("import_chunks_to_graph: item_name 或 chunks 为空，跳过")
        return 0

    ensure_graph_indexes()
    driver = get_neo4j_driver()
    product_key = _product_key(item_name)
    sorted_chunks = sorted(chunks, key=lambda row: row.get("part", 0))
    payload = build_graph_payload(item_name, sorted_chunks)

    document_key_by_title = {
        _clean_text(row.get("title")): row.get("node_key")
        for row in payload.get("documents") or []
        if row.get("node_key")
    }
    section_key_by_pair = {
        (
            _clean_text(row.get("file_title")),
            _clean_text(row.get("title")),
        ): row.get("node_key")
        for row in payload.get("sections") or []
        if row.get("node_key")
    }

    document_rows = _dedupe_rows(payload.get("documents") or [], ["node_key"])
    section_rows: List[Dict[str, Any]] = []
    for row in payload.get("sections") or []:
        row_copy = dict(row)
        row_copy["document_key"] = document_key_by_title.get(_clean_text(row.get("file_title")), "")
        section_rows.append(row_copy)
    section_rows = _dedupe_rows(section_rows, ["node_key"])

    chunk_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    for sequence, chunk in enumerate(sorted_chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        file_title = _clean_text(chunk.get("file_title") or item_name)
        section_title = _clean_text(chunk.get("title") or chunk.get("parent_title") or file_title)
        section_key = section_key_by_pair.get((file_title, section_title), "")
        document_key = document_key_by_title.get(file_title, "")
        chunk_row = {
            "node_key": f"{product_key}::chunk::{chunk_id}",
            "chunk_id": chunk_id,
            "content": _clean_text(chunk.get("content") or "")[:4000],
            "title": _clean_text(chunk.get("title") or ""),
            "parent_title": _clean_text(chunk.get("parent_title") or ""),
            "part": int(chunk.get("part", 0) or 0),
            "sequence": sequence,
            "item_name": item_name,
            "file_title": file_title,
            "image_urls": chunk.get("image_urls") or [],
            "section_key": section_key,
            "document_key": document_key,
        }
        chunk_rows.append(chunk_row)
        for image_url in chunk_row["image_urls"]:
            normalized_url = _clean_text(image_url)
            if not normalized_url:
                continue
            image_rows.append(
                {
                    "node_key": f"{product_key}::image::{_slug(normalized_url)}",
                    "url": normalized_url,
                    "chunk_id": chunk_id,
                    "product_name": item_name,
                }
            )

    semantic_nodes: List[Dict[str, Any]] = []
    for node in payload.get("semantic_nodes") or []:
        label = node.get("label")
        if label not in _ALLOWED_ENTITY_LABELS:
            continue
        semantic_nodes.append(dict(node))

    semantic_relations: List[Dict[str, Any]] = []
    for relation in payload.get("semantic_relations") or []:
        relation_type = relation.get("type")
        if relation_type not in _ALLOWED_RELATION_TYPES:
            continue
        semantic_relations.append(dict(relation))

    evidence_rows: List[Dict[str, Any]] = []
    for node in semantic_nodes:
        for chunk_id in node.get("source_chunk_ids") or []:
            if chunk_id:
                evidence_rows.append(
                    {
                        "node_key": node.get("node_key"),
                        "chunk_id": str(chunk_id),
                    }
                )

    with driver.session(database=_get_database()) as session:
        session.run(
            "MATCH (n) WHERE n.product_name = $name DETACH DELETE n",
            name=item_name,
        )
        session.run(
            "MERGE (p:KGNode:Product {node_key: $node_key}) "
            "SET p.name = $name, p.product_name = $name",
            node_key=product_key,
            name=item_name,
        )

        if document_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (p:Product {node_key: $product_key})
                MERGE (d:KGNode:Document {node_key: row.node_key})
                SET d.title = row.title,
                    d.product_name = $product_name
                MERGE (p)-[:HAS_DOCUMENT]->(d)
                """,
                rows=document_rows,
                product_key=product_key,
                product_name=item_name,
            )

        if section_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (d:Document {node_key: row.document_key})
                MERGE (s:KGNode:Section {node_key: row.node_key})
                SET s.title = row.title,
                    s.parent_title = row.parent_title,
                    s.file_title = row.file_title,
                    s.product_name = $product_name
                MERGE (d)-[:HAS_SECTION]->(s)
                """,
                rows=section_rows,
                product_name=item_name,
            )

        if chunk_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (p:Product {node_key: $product_key})
                MATCH (s:Section {node_key: row.section_key})
                MERGE (c:KGNode:Chunk {chunk_id: row.chunk_id})
                SET c.node_key = row.node_key,
                    c.content = row.content,
                    c.title = row.title,
                    c.parent_title = row.parent_title,
                    c.part = row.part,
                    c.sequence = row.sequence,
                    c.item_name = row.item_name,
                    c.file_title = row.file_title,
                    c.image_urls = row.image_urls,
                    c.product_name = $product_name
                MERGE (p)-[:HAS_CHUNK]->(c)
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                rows=chunk_rows,
                product_key=product_key,
                product_name=item_name,
            )
            session.run(
                """
                MATCH (p:Product {node_key: $product_key})-[:HAS_CHUNK]->(c:Chunk)
                WITH c ORDER BY c.sequence
                WITH collect(c) AS nodes
                UNWIND range(0, size(nodes) - 2) AS i
                WITH nodes[i] AS a, nodes[i + 1] AS b
                MERGE (a)-[:NEXT]->(b)
                """,
                product_key=product_key,
            )

        if image_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (i:KGNode:Image {node_key: row.node_key})
                SET i.url = row.url,
                    i.product_name = $product_name
                MERGE (c)-[:HAS_IMAGE]->(i)
                """,
                rows=_dedupe_rows(image_rows, ["node_key", "chunk_id"]),
                product_name=item_name,
            )

        grouped_nodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in semantic_nodes:
            grouped_nodes[node["label"]].append(node)
        for label, rows in grouped_nodes.items():
            session.run(
                f"""
                UNWIND $rows AS row
                MERGE (e:KGNode:KGEntity:{label} {{node_key: row.node_key}})
                SET e.name = row.name,
                    e.description = row.description,
                    e.value = row.value,
                    e.order = row.order,
                    e.product_name = $product_name,
                    e.source_chunk_ids = row.source_chunk_ids,
                    e.source_titles = row.source_titles,
                    e.source_section_keys = row.source_section_keys,
                    e.entity_type = $entity_type
                """,
                rows=_dedupe_rows(rows, ["node_key"]),
                product_name=item_name,
                entity_type=label,
            )

        if evidence_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (e:KGNode {node_key: row.node_key})
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (e)-[:EVIDENCED_BY]->(c)
                """,
                rows=_dedupe_rows(evidence_rows, ["node_key", "chunk_id"]),
            )

        grouped_relations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for relation in semantic_relations:
            grouped_relations[relation["type"]].append(relation)
        for relation_type, rows in grouped_relations.items():
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:KGNode {{node_key: row.source_key}})
                MATCH (b:KGNode {{node_key: row.target_key}})
                MERGE (a)-[r:{relation_type} {{evidence_chunk_id: row.evidence_chunk_id}}]->(b)
                SET r.product_name = $product_name
                """,
                rows=_dedupe_rows(rows, ["source_key", "target_key", "type", "evidence_chunk_id"]),
                product_name=item_name,
            )

    logger.info(
        f"Neo4j 增强图导入完成: item_name={item_name}, chunks={len(chunk_rows)}, semantic_nodes={len(semantic_nodes)}, semantic_relations={len(semantic_relations)}"
    )
    return len(chunk_rows)


def delete_product_graph(item_name: str) -> None:
    ensure_graph_indexes()
    driver = get_neo4j_driver()
    with driver.session(database=_get_database()) as session:
        session.run(
            "MATCH (n) WHERE n.product_name = $name DETACH DELETE n",
            name=item_name,
        )
        session.run(
            "MATCH (p:Product {name: $name}) DETACH DELETE p",
            name=item_name,
        )
    logger.info(f"Neo4j 增强图删除完成: {item_name}")


def _normalize_terms(question: str, focus_terms: Sequence[str] | None = None) -> List[str]:
    raw_terms = list(focus_terms or extract_focus_terms(question, []))
    normalized: List[str] = []
    for term in raw_terms:
        cleaned = _clean_text(term).lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized[:8]


def _coerce_chunk_maps(chunk_maps: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in chunk_maps or []:
        if not isinstance(row, dict):
            continue
        results.append(
            {
                "chunk_id": str(row.get("chunk_id") or ""),
                "content": _clean_text(row.get("content") or ""),
                "title": _clean_text(row.get("title") or ""),
                "image_urls": row.get("image_urls") or [],
            }
        )
    return results


def _build_graph_doc(
    *,
    item_name: str,
    title: str,
    graph_fact: str,
    evidence_chunks: Sequence[Dict[str, Any]] | None = None,
    graph_entities: Sequence[str] | None = None,
    graph_relations: Sequence[str] | None = None,
    graph_query_type: str,
) -> Dict[str, Any]:
    chunks = _coerce_chunk_maps(evidence_chunks)
    chunk_id = next((row.get("chunk_id") for row in chunks if row.get("chunk_id")), "")
    evidence_text = "\n\n".join(
        f"[chunk_id={row.get('chunk_id')}] {row.get('content')}" for row in chunks if row.get("content")
    )
    merged_image_urls: List[str] = []
    for row in chunks:
        for url in row.get("image_urls") or []:
            cleaned = _clean_text(url)
            if cleaned and cleaned not in merged_image_urls:
                merged_image_urls.append(cleaned)
    content_parts = [part for part in [graph_fact, evidence_text] if part]
    return {
        "chunk_id": chunk_id,
        "content": "\n\n".join(content_parts),
        "title": title,
        "item_name": item_name,
        "part": 0,
        "image_urls": merged_image_urls,
        "source": "kg",
        "graph_fact": graph_fact,
        "graph_entities": list(graph_entities or []),
        "graph_relations": list(graph_relations or []),
        "evidence_chunk_ids": [row.get("chunk_id") for row in chunks if row.get("chunk_id")],
        "evidence_text": evidence_text,
        "graph_query_type": graph_query_type,
    }


def _query_navigation(session, item_names: Sequence[str], terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    records = session.run(
        """
        MATCH (p:Product)
        WHERE size($names) = 0 OR p.name IN $names
        MATCH (p)-[:HAS_DOCUMENT]->(:Document)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_STEP]->(step:Step)
        OPTIONAL MATCH (prev:Step)-[:NEXT_STEP]->(step)
        OPTIONAL MATCH (step)-[:NEXT_STEP]->(next:Step)
        OPTIONAL MATCH (step)-[:EVIDENCED_BY]->(c:Chunk)
        WITH p, s, step, prev, next, collect(distinct {
            chunk_id: c.chunk_id,
            content: c.content,
            title: c.title,
            image_urls: c.image_urls
        })[0..2] AS chunks
        WHERE step IS NOT NULL AND (
            size($terms) = 0 OR any(term IN $terms WHERE
                toLower(coalesce(step.name, '')) CONTAINS term OR
                toLower(coalesce(step.description, '')) CONTAINS term OR
                toLower(coalesce(s.title, '')) CONTAINS term
            )
        )
        RETURN p.name AS item_name,
               s.title AS section_title,
               step.name AS step_name,
               step.description AS step_description,
               step.order AS step_order,
               prev.name AS prev_step,
               next.name AS next_step,
               chunks AS chunks
        ORDER BY item_name, step_order
        LIMIT $limit
        """,
        names=list(item_names),
        terms=list(terms),
        limit=limit,
    )
    results: List[Dict[str, Any]] = []
    for row in records:
        graph_fact = f"章节【{row['section_title']}】中的步骤【{row['step_name']}】"
        if row.get("prev_step"):
            graph_fact += f"，上一步是【{row['prev_step']}】"
        if row.get("next_step"):
            graph_fact += f"，下一步是【{row['next_step']}】"
        if row.get("step_description"):
            graph_fact += f"。步骤说明：{row['step_description']}"
        results.append(
            _build_graph_doc(
                item_name=row["item_name"],
                title=row.get("section_title") or "导航结果",
                graph_fact=graph_fact,
                evidence_chunks=row.get("chunks"),
                graph_entities=[row.get("step_name") or "", row.get("section_title") or ""],
                graph_relations=["HAS_STEP", "NEXT_STEP"],
                graph_query_type="navigation",
            )
        )
    return results


def _query_comparison(session, item_names: Sequence[str], terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    records = session.run(
        """
        MATCH (p:Product)
        WHERE size($names) = 0 OR p.name IN $names
        MATCH (p)-[:HAS_DOCUMENT]->(:Document)-[:HAS_SECTION]->(:Section)-[:HAS_PARAMETER]->(param:Parameter)
        OPTIONAL MATCH (param)-[:EVIDENCED_BY]->(c:Chunk)
        WITH p, param, collect(distinct {
            chunk_id: c.chunk_id,
            content: c.content,
            title: c.title,
            image_urls: c.image_urls
        })[0..2] AS chunks
        WHERE size($terms) = 0 OR any(term IN $terms WHERE
            toLower(coalesce(param.name, '')) CONTAINS term OR
            toLower(coalesce(param.value, '')) CONTAINS term OR
            toLower(coalesce(param.description, '')) CONTAINS term
        )
        RETURN p.name AS item_name,
               param.name AS parameter_name,
               param.value AS parameter_value,
               param.description AS parameter_description,
               chunks AS chunks
        ORDER BY parameter_name, item_name
        LIMIT $limit
        """,
        names=list(item_names),
        terms=list(terms),
        limit=limit,
    )
    results: List[Dict[str, Any]] = []
    for row in records:
        graph_fact = f"产品【{row['item_name']}】的参数【{row['parameter_name']}】"
        if row.get("parameter_value"):
            graph_fact += f" = {row['parameter_value']}"
        if row.get("parameter_description") and row.get("parameter_description") != graph_fact:
            graph_fact += f"。说明：{row['parameter_description']}"
        results.append(
            _build_graph_doc(
                item_name=row["item_name"],
                title=row.get("parameter_name") or "参数对比",
                graph_fact=graph_fact,
                evidence_chunks=row.get("chunks"),
                graph_entities=[row.get("parameter_name") or ""],
                graph_relations=["HAS_PARAMETER"],
                graph_query_type="comparison",
            )
        )
    return results


def _query_relation_like(session, item_names: Sequence[str], terms: Sequence[str], limit: int, query_type: str) -> List[Dict[str, Any]]:
    records = session.run(
        """
        MATCH (p:Product)
        WHERE size($names) = 0 OR p.name IN $names
        MATCH (e:KGEntity)
        WHERE e.product_name = p.name AND (
            size($terms) = 0 OR any(term IN $terms WHERE
                toLower(coalesce(e.name, '')) CONTAINS term OR
                toLower(coalesce(e.description, '')) CONTAINS term OR
                toLower(coalesce(e.value, '')) CONTAINS term
            )
        )
        OPTIONAL MATCH (e)-[r]-(neighbor:KGNode)
        WITH p, e, collect(distinct CASE WHEN type(r) IN $allowed_relations THEN {
            relation: type(r),
            neighbor_name: coalesce(neighbor.name, neighbor.title, neighbor.url, neighbor.chunk_id),
            neighbor_labels: labels(neighbor)
        } ELSE NULL END) AS link_candidates
        OPTIONAL MATCH (e)-[:EVIDENCED_BY]->(c:Chunk)
        WITH p, e,
             [link IN link_candidates WHERE link IS NOT NULL][0..6] AS links,
             collect(distinct {
                chunk_id: c.chunk_id,
                content: c.content,
                title: c.title,
                image_urls: c.image_urls
             })[0..2] AS chunks
        RETURN p.name AS item_name,
               labels(e) AS entity_labels,
               e.name AS entity_name,
               e.description AS entity_description,
               e.value AS entity_value,
               links AS links,
               chunks AS chunks
        ORDER BY item_name, entity_name
        LIMIT $limit
        """,
        names=list(item_names),
        terms=list(terms),
        allowed_relations=[
            "CAUSED_BY",
            "RESOLVED_BY",
            "RELATED_TO",
            "HAS_COMPONENT",
            "HAS_PARAMETER",
            "HAS_STEP",
            "HAS_FAULT",
            "HAS_CAUSE",
            "HAS_SOLUTION",
            "HAS_WARNING",
            "NEXT_STEP",
        ],
        limit=limit,
    )
    results: List[Dict[str, Any]] = []
    for row in records:
        entity_name = row.get("entity_name") or "实体"
        link_parts = []
        relation_names = []
        for link in row.get("links") or []:
            relation_name = link.get("relation")
            neighbor_name = link.get("neighbor_name")
            if relation_name and neighbor_name:
                relation_names.append(relation_name)
                link_parts.append(f"{relation_name} → {neighbor_name}")
        graph_fact = f"实体【{entity_name}】"
        if row.get("entity_description"):
            graph_fact += f"：{row['entity_description']}"
        if row.get("entity_value"):
            graph_fact += f"，值为 {row['entity_value']}"
        if link_parts:
            graph_fact += f"。关联路径：{'；'.join(link_parts)}"
        results.append(
            _build_graph_doc(
                item_name=row["item_name"],
                title=entity_name,
                graph_fact=graph_fact,
                evidence_chunks=row.get("chunks"),
                graph_entities=[entity_name],
                graph_relations=relation_names,
                graph_query_type=query_type,
            )
        )
    return results


def _query_constraint(session, item_names: Sequence[str], terms: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    records = session.run(
        """
        MATCH (p:Product)
        WHERE size($names) = 0 OR p.name IN $names
        MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
        WITH p, collect(distinct {
            chunk_id: c.chunk_id,
            content: c.content,
            title: c.title,
            image_urls: c.image_urls
        })[0..30] AS chunks
        WHERE size($terms) = 0 OR all(term IN $terms WHERE any(chunk IN chunks WHERE
            toLower(coalesce(chunk.content, '')) CONTAINS term OR
            toLower(coalesce(chunk.title, '')) CONTAINS term
        ))
        RETURN p.name AS item_name, chunks[0..3] AS chunks
        LIMIT $limit
        """,
        names=list(item_names),
        terms=list(terms),
        limit=limit,
    )
    results: List[Dict[str, Any]] = []
    for row in records:
        term_text = "、".join(terms) if terms else "查询条件"
        graph_fact = f"产品【{row['item_name']}】满足约束条件：{term_text}"
        results.append(
            _build_graph_doc(
                item_name=row["item_name"],
                title="约束匹配",
                graph_fact=graph_fact,
                evidence_chunks=row.get("chunks"),
                graph_entities=[row.get("item_name") or ""],
                graph_relations=["HAS_CHUNK"],
                graph_query_type="constraint",
            )
        )
    return results


def _fallback_graph_chunks(item_names: Sequence[str], limit: int, query_type: str) -> List[Dict[str, Any]]:
    rows = query_chunks_by_product(list(item_names), limit=limit)
    results: List[Dict[str, Any]] = []
    for row in rows:
        graph_fact = f"图谱回退命中 chunk【{row.get('chunk_id')}】"
        results.append(
            _build_graph_doc(
                item_name=row.get("item_name") or "",
                title=row.get("title") or "图谱回退",
                graph_fact=graph_fact,
                evidence_chunks=[
                    {
                        "chunk_id": row.get("chunk_id"),
                        "content": row.get("content"),
                        "title": row.get("title"),
                        "image_urls": row.get("image_urls") or [],
                    }
                ],
                graph_entities=[row.get("title") or ""],
                graph_relations=["HAS_CHUNK"],
                graph_query_type=query_type,
            )
        )
    return results


def query_graph_context(
    question: str,
    item_names: Sequence[str] | None,
    *,
    query_type: str,
    focus_terms: Sequence[str] | None = None,
    limit: int = 8,
) -> Dict[str, Any]:
    if not question and not item_names:
        return {"kg_chunks": [], "summary": {"query_type": query_type, "result_count": 0}}

    ensure_graph_indexes()
    driver = get_neo4j_driver()
    terms = _normalize_terms(question, focus_terms)
    target_names = [name for name in (item_names or []) if _clean_text(name)]
    safe_limit = max(1, int(limit or 8))
    template = query_type if query_type in {"navigation", "comparison", "relation", "constraint", "explain"} else "general"

    with driver.session(database=_get_database()) as session:
        if template == "navigation":
            docs = _query_navigation(session, target_names, terms, safe_limit)
        elif template == "comparison":
            docs = _query_comparison(session, target_names, terms, safe_limit)
        elif template == "constraint":
            docs = _query_constraint(session, target_names, terms, safe_limit)
        elif template in {"relation", "explain"}:
            docs = _query_relation_like(session, target_names, terms, safe_limit, template)
        else:
            docs = []

    fallback_used = False
    if not docs and target_names:
        docs = _fallback_graph_chunks(target_names, safe_limit, template)
        fallback_used = bool(docs)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs:
        marker = (
            doc.get("chunk_id"),
            doc.get("title"),
            doc.get("item_name"),
            doc.get("graph_fact"),
        )
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(doc)

    summary = {
        "query_type": template,
        "focus_terms": terms,
        "result_count": len(deduped),
        "used_fallback": fallback_used,
    }
    return {"kg_chunks": deduped[:safe_limit], "summary": summary}
