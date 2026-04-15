from collections import Counter
from typing import Any, Dict, List, Sequence

from app.clients.milvus_schema import FIELD_STABLE_CHUNK_ID
from app.clients.milvus_utils import get_milvus_client, query_chunks_by_filter
from app.clients.neo4j_graph_utils import import_chunks_to_graph
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.nodes.node_import_milvus import create_collection
from app.query_process.agent.retrieval_utils import build_item_name_filter_expr
from app.utils.chunk_id_utils import ensure_chunk_id


DISCOVERY_OUTPUT_FIELDS = ["item_name", "chunk_id", "stable_chunk_id"]
MIGRATION_OUTPUT_FIELDS = [
    "chunk_id",
    "stable_chunk_id",
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    "image_urls",
    "dense_vector",
    "sparse_vector",
]
RESTORE_OUTPUT_FIELDS = [
    "content",
    "title",
    "parent_title",
    "part",
    "file_title",
    "item_name",
    "image_urls",
    "dense_vector",
    "sparse_vector",
    "stable_chunk_id",
]


def _target_collection_name(collection_name: str | None = None) -> str:
    return collection_name or milvus_config.chunks_collection or ""


def _flush_collection(client, collection_name: str) -> None:
    if hasattr(client, "flush"):
        client.flush(collection_name=collection_name)


def _normalize_item_names(item_names: Sequence[str] | None) -> List[str]:
    normalized = []
    for item_name in item_names or []:
        text = str(item_name or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _discover_item_names(client, collection_name: str) -> List[str]:
    rows = query_chunks_by_filter(
        client=client,
        collection_name=collection_name,
        filter_expr="",
        output_fields=list(DISCOVERY_OUTPUT_FIELDS),
        limit=-1,
    )
    return sorted(
        {
            str(row.get("item_name") or "").strip()
            for row in rows
            if str(row.get("item_name") or "").strip()
        }
    )


def _collection_supports_stable_ids(client, collection_name: str) -> bool:
    description = client.describe_collection(collection_name=collection_name)
    field_names = {
        str(field.get("name") or "")
        for field in (description.get("fields") or [])
        if isinstance(field, dict)
    }
    return bool(
        description.get("enable_dynamic_field") or FIELD_STABLE_CHUNK_ID in field_names
    )


def _batch_insert(
    client, collection_name: str, rows: Sequence[Dict[str, Any]], batch_size: int = 200
) -> int:
    inserted_total = 0
    for index in range(0, len(rows), batch_size):
        batch = list(rows[index : index + batch_size])
        if not batch:
            continue
        result = client.insert(collection_name=collection_name, data=batch)
        inserted_total += int(result.get("insert_count", 0) or 0)
    _flush_collection(client, collection_name)
    return inserted_total


def _storage_payload(row: Dict[str, Any], stable_chunk_id: str) -> Dict[str, Any]:
    payload = {
        "content": row.get("content") or "",
        "title": row.get("title") or "",
        "parent_title": row.get("parent_title") or "",
        "part": int(row.get("part", 0) or 0),
        "file_title": row.get("file_title") or "",
        "item_name": row.get("item_name") or "",
        "image_urls": row.get("image_urls") or [],
        "dense_vector": row.get("dense_vector"),
        "sparse_vector": row.get("sparse_vector"),
        "stable_chunk_id": stable_chunk_id,
    }
    return payload


def _original_storage_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": row.get("content") or "",
        "title": row.get("title") or "",
        "parent_title": row.get("parent_title") or "",
        "part": int(row.get("part", 0) or 0),
        "file_title": row.get("file_title") or "",
        "item_name": row.get("item_name") or "",
        "image_urls": row.get("image_urls") or [],
        "dense_vector": row.get("dense_vector"),
        "sparse_vector": row.get("sparse_vector"),
    }


def _graph_payload(row: Dict[str, Any], stable_chunk_id: str) -> Dict[str, Any]:
    payload = _storage_payload(row, stable_chunk_id)
    payload["chunk_id"] = stable_chunk_id
    storage_chunk_id = str(
        row.get("storage_chunk_id") or row.get("chunk_id") or row.get("id") or ""
    ).strip()
    if storage_chunk_id and storage_chunk_id != stable_chunk_id:
        payload["storage_chunk_id"] = storage_chunk_id
    return payload


def _load_all_collection_rows(client, collection_name: str) -> List[Dict[str, Any]]:
    return query_chunks_by_filter(
        client=client,
        collection_name=collection_name,
        filter_expr="",
        output_fields=list(MIGRATION_OUTPUT_FIELDS),
        limit=-1,
    )


def _rebuild_collection_with_stable_ids(
    client,
    *,
    collection_name: str,
    dry_run: bool,
    sync_graph: bool,
) -> Dict[str, Any]:
    all_rows = _load_all_collection_rows(client, collection_name)
    item_names = sorted(
        {
            str(row.get("item_name") or "").strip()
            for row in all_rows
            if str(row.get("item_name") or "").strip()
        }
    )
    if not all_rows:
        return {
            "collection_name": collection_name,
            "item_names": item_names,
            "dry_run": dry_run,
            "sync_graph": sync_graph,
            "items_scanned": 0,
            "chunks_scanned": 0,
            "chunks_migrated": 0,
            "graph_synced_items": 0,
            "status_breakdown": {},
            "details": [],
            "message": "当前知识库为空，无需迁移。",
        }

    migrated_storage_rows: List[Dict[str, Any]] = []
    original_storage_rows: List[Dict[str, Any]] = []
    graph_rows_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        row_copy = dict(row)
        stable_chunk_id = ensure_chunk_id(row_copy)
        migrated_storage_rows.append(_storage_payload(row_copy, stable_chunk_id))
        original_storage_rows.append(_original_storage_payload(row))
        item_name = str(row_copy.get("item_name") or "").strip()
        if item_name:
            graph_rows_by_item.setdefault(item_name, []).append(
                _graph_payload(row_copy, stable_chunk_id)
            )

    if dry_run:
        return {
            "collection_name": collection_name,
            "item_names": item_names,
            "dry_run": True,
            "sync_graph": sync_graph,
            "items_scanned": len(item_names),
            "chunks_scanned": len(all_rows),
            "chunks_migrated": len(all_rows),
            "graph_synced_items": 0,
            "status_breakdown": {"dry_run_rebuild": len(item_names)},
            "details": [
                {
                    "item_name": item_name,
                    "status": "dry_run_rebuild",
                    "chunks": len(graph_rows_by_item.get(item_name) or []),
                    "legacy_rows": len(graph_rows_by_item.get(item_name) or []),
                    "already_stable_rows": 0,
                    "graph_synced": False,
                }
                for item_name in item_names
            ],
            "message": f"检测到旧版 Milvus schema，预计重建集合并迁移 {len(all_rows)} 条 chunk 到稳定 ID。",
        }

    first_dense_vector = next(
        (row.get("dense_vector") for row in migrated_storage_rows if row.get("dense_vector")),
        None,
    )
    vector_dimension = len(first_dense_vector or [])
    if vector_dimension <= 0:
        raise ValueError("无法从现有集合中解析 dense_vector 维度，迁移终止")

    client.drop_collection(collection_name=collection_name)
    create_collection(client, collection_name, vector_dimension)
    try:
        inserted = _batch_insert(client, collection_name, migrated_storage_rows)
        if inserted != len(migrated_storage_rows):
            raise ValueError(
                f"集合重建后写回数量异常: expected={len(migrated_storage_rows)}, actual={inserted}"
            )
    except Exception:
        logger.exception("集合重建迁移失败，开始回滚为无稳定ID版本数据")
        client.drop_collection(collection_name=collection_name)
        create_collection(client, collection_name, vector_dimension)
        _batch_insert(client, collection_name, original_storage_rows)
        raise

    graph_synced_items = 0
    if sync_graph:
        for item_name, rows in graph_rows_by_item.items():
            import_chunks_to_graph(item_name, rows)
            graph_synced_items += 1

    return {
        "collection_name": collection_name,
        "item_names": item_names,
        "dry_run": False,
        "sync_graph": sync_graph,
        "items_scanned": len(item_names),
        "chunks_scanned": len(all_rows),
        "chunks_migrated": len(all_rows),
        "graph_synced_items": graph_synced_items,
        "status_breakdown": {"rebuild_migrated": len(item_names)},
        "details": [
            {
                "item_name": item_name,
                "status": "rebuild_migrated",
                "chunks": len(graph_rows_by_item.get(item_name) or []),
                "legacy_rows": len(graph_rows_by_item.get(item_name) or []),
                "already_stable_rows": 0,
                "graph_synced": sync_graph,
            }
            for item_name in item_names
        ],
        "message": f"已完成旧版 Milvus collection 重建，并将 {len(all_rows)} 条 chunk 迁移到稳定 ID。",
    }


def _migrate_single_item(
    client,
    *,
    collection_name: str,
    item_name: str,
    dry_run: bool,
    sync_graph: bool,
) -> Dict[str, Any]:
    filter_expr = build_item_name_filter_expr([item_name])
    rows = query_chunks_by_filter(
        client=client,
        collection_name=collection_name,
        filter_expr=filter_expr,
        output_fields=list(MIGRATION_OUTPUT_FIELDS),
        limit=-1,
    )
    if not rows:
        return {
            "item_name": item_name,
            "status": "missing",
            "chunks": 0,
            "legacy_rows": 0,
            "already_stable_rows": 0,
            "graph_synced": False,
        }

    storage_payload: List[Dict[str, Any]] = []
    restore_payload: List[Dict[str, Any]] = []
    graph_rows: List[Dict[str, Any]] = []
    legacy_rows = 0
    already_stable_rows = 0

    for row in rows:
        row_copy = dict(row)
        previous_stable_chunk_id = str(row_copy.get("stable_chunk_id") or "").strip()
        stable_chunk_id = ensure_chunk_id(row_copy)
        if previous_stable_chunk_id == stable_chunk_id and previous_stable_chunk_id:
            already_stable_rows += 1
        else:
            legacy_rows += 1

        storage_payload.append(_storage_payload(row_copy, stable_chunk_id))
        graph_rows.append(_graph_payload(row_copy, stable_chunk_id))

        original_payload = {
            key: row.get(key)
            for key in RESTORE_OUTPUT_FIELDS
            if row.get(key) is not None
        }
        restore_payload.append(original_payload)

    will_migrate = legacy_rows > 0
    status = "already_stable"
    graph_synced = False

    if not dry_run and (will_migrate or sync_graph):
        if will_migrate:
            client.delete(collection_name=collection_name, filter=filter_expr)
            _flush_collection(client, collection_name)
            try:
                insert_result = client.insert(
                    collection_name=collection_name,
                    data=storage_payload,
                )
                inserted = int(insert_result.get("insert_count", 0) or 0)
                if inserted != len(storage_payload):
                    raise ValueError(
                        f"{item_name} 迁移写回数量异常: expected={len(storage_payload)}, actual={inserted}"
                    )
                _flush_collection(client, collection_name)
                status = "migrated"
            except Exception:
                logger.exception(f"chunk_id 迁移失败，开始回滚原始数据：{item_name}")
                client.insert(collection_name=collection_name, data=restore_payload)
                _flush_collection(client, collection_name)
                raise

        if sync_graph:
            import_chunks_to_graph(item_name, graph_rows)
            graph_synced = True
            if status == "already_stable":
                status = "graph_refreshed"

    elif dry_run and will_migrate:
        status = "dry_run"

    return {
        "item_name": item_name,
        "status": status,
        "chunks": len(rows),
        "legacy_rows": legacy_rows,
        "already_stable_rows": already_stable_rows,
        "graph_synced": graph_synced,
    }


def migrate_collection_chunk_ids(
    *,
    item_names: Sequence[str] | None = None,
    collection_name: str | None = None,
    dry_run: bool = False,
    sync_graph: bool = True,
) -> Dict[str, Any]:
    client = get_milvus_client()
    target_collection = _target_collection_name(collection_name)
    if client is None:
        raise ValueError("Milvus 客户端不可用，无法执行 chunk_id 迁移")
    if not target_collection:
        raise ValueError("未配置 CHUNKS_COLLECTION，无法执行 chunk_id 迁移")
    if not client.has_collection(collection_name=target_collection):
        raise ValueError(f"Milvus 集合不存在: {target_collection}")
    if not _collection_supports_stable_ids(client, target_collection):
        return _rebuild_collection_with_stable_ids(
            client,
            collection_name=target_collection,
            dry_run=dry_run,
            sync_graph=sync_graph,
        )

    targets = _normalize_item_names(item_names)
    if not targets:
        targets = _discover_item_names(client, target_collection)

    details: List[Dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    chunks_scanned = 0
    chunks_migrated = 0
    graph_synced_items = 0

    for item_name in targets:
        detail = _migrate_single_item(
            client,
            collection_name=target_collection,
            item_name=item_name,
            dry_run=dry_run,
            sync_graph=sync_graph,
        )
        details.append(detail)
        status_counter[str(detail.get("status") or "unknown")] += 1
        chunks_scanned += int(detail.get("chunks") or 0)
        chunks_migrated += int(detail.get("legacy_rows") or 0)
        if detail.get("graph_synced"):
            graph_synced_items += 1

    message = (
        f"已检查 {len(targets)} 个知识库 item_name，"
        f"{'预计迁移' if dry_run else '实际迁移'} {chunks_migrated} 条 chunk 到稳定 ID。"
    )

    return {
        "collection_name": target_collection,
        "item_names": targets,
        "dry_run": dry_run,
        "sync_graph": sync_graph,
        "items_scanned": len(targets),
        "chunks_scanned": chunks_scanned,
        "chunks_migrated": chunks_migrated,
        "graph_synced_items": graph_synced_items,
        "status_breakdown": dict(sorted(status_counter.items())),
        "details": details,
        "message": message,
    }
