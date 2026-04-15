import os
import threading
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.clients.milvus_schema import (
    CHUNKS_OUTPUT_FIELDS,
    FIELD_STABLE_CHUNK_ID,
    extract_chunk_id,
    extract_storage_chunk_id,
    normalize_entity_for_business_id,
)
from app.utils.escape_milvus_string_utils import escape_milvus_string

# 全局Milvus客户端实例，实现单例复用
_milvus_client = None
_milvus_lock = threading.Lock()


def get_milvus_client():
    """
    Milvus客户端单例获取方法
    实现客户端连接复用，避免重复创建连接消耗资源
    :return: MilvusClient实例，连接失败返回None
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    with _milvus_lock:
        if _milvus_client is not None:
            return _milvus_client
        try:
            milvus_uri = milvus_config.milvus_url
            # 校验Milvus连接地址配置
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 初始化Milvus客户端
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
            return _milvus_client
        except Exception as e:
            logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
            return None


def _coerce_int64_ids(ids):
    """
    转换chunk_id为Milvus要求的INT64类型（主键字段schema为INT64）
    过滤无效ID，分离可转换/不可转换的ID
    :param ids: 待转换的chunk_id列表
    :return: 元组(ok_ids, bad_ids)，ok_ids为可转换的int64类型ID列表，bad_ids为无效ID列表
    """
    ok, bad = [], []
    for x in ids or []:
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def _ensure_output_fields(output_fields):
    fields = list(output_fields or [])
    if FIELD_STABLE_CHUNK_ID not in fields:
        fields.insert(1 if fields else 0, FIELD_STABLE_CHUNK_ID)
    return fields


def _normalize_rows(rows):
    return [
        normalize_entity_for_business_id(dict(row))
        for row in (rows or [])
        if isinstance(row, dict)
    ]


def _supports_stable_field_error(exc: Exception) -> bool:
    return FIELD_STABLE_CHUNK_ID in str(exc)


def _without_stable_field(output_fields):
    return [field for field in (output_fields or []) if field != FIELD_STABLE_CHUNK_ID]


def _coerce_string_ids(ids):
    normalized = []
    for value in ids or []:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _ensure_output_fields(output_fields):
    fields = list(output_fields or [])
    if FIELD_STABLE_CHUNK_ID not in fields:
        fields.append(FIELD_STABLE_CHUNK_ID)
    return fields


def _normalize_entities(entities):
    return [
        normalize_entity_for_business_id(dict(entity))
        for entity in (entities or [])
        if isinstance(entity, dict)
    ]


def fetch_chunks_by_chunk_ids(
    client,
    collection_name: str,
    chunk_ids,
    *,
    output_fields=None,
    batch_size: int = 100,
):
    """
    通过chunk_id主键批量查询Milvus中的切片数据
    用于补全「仅拥有chunk_id无文本内容」场景的切片信息
    优先使用get方法（主键直查，性能最优），失败则回退query过滤查询
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param chunk_ids: 待查询的chunk_id列表
    :param output_fields: 需要返回的字段列表，默认返回核心切片字段
    :param batch_size: 分批查询大小，避免单次查询数据量过大，默认100
    :return: List[dict]，Milvus实体字典列表，查询失败返回空列表
    """
    # 前置校验：客户端/集合名无效直接返回空
    if client is None:
        return []
    if not collection_name:
        return []
    # 默认返回字段：核心切片标识与内容字段
    if output_fields is None:
        output_fields = [
            "chunk_id",
            "stable_chunk_id",
            "content",
            "title",
            "parent_title",
            "item_name",
        ]
    output_fields = _ensure_output_fields(output_fields)

    normalized_input_ids = [str(value).strip() for value in (chunk_ids or []) if str(value).strip()]
    numeric_ids, invalid_numeric_ids = _coerce_int64_ids(normalized_input_ids)
    stable_ids = [value for value in normalized_input_ids if value not in {str(x) for x in numeric_ids}]
    if invalid_numeric_ids:
        logger.warning(f"存在无法转换为INT64的chunk_id，将改用 stable_chunk_id 查询：{invalid_numeric_ids}")

    if not numeric_ids and not stable_ids:
        return []

    results = []
    # 分批查询内部主键
    for i in range(0, len(numeric_ids), batch_size):
        batch = numeric_ids[i : i + batch_size]

        # 方式1：优先使用主键get方法查询（性能最优）
        if hasattr(client, "get"):
            try:
                got = client.get(
                    collection_name=collection_name,
                    ids=batch,
                    output_fields=output_fields,
                )
                if got:
                    results.extend(_normalize_rows(got))
                continue
            except Exception as e:
                if _supports_stable_field_error(e) and FIELD_STABLE_CHUNK_ID in output_fields:
                    fallback_fields = _without_stable_field(output_fields)
                    try:
                        got = client.get(
                            collection_name=collection_name,
                            ids=batch,
                            output_fields=fallback_fields,
                        )
                        if got:
                            results.extend(_normalize_rows(got))
                        continue
                    except Exception as retry_exc:
                        logger.warning(
                            f"Milvus get方法移除stable_chunk_id后仍失败，将回退至query方法：{str(retry_exc)}"
                        )
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        # 方式2：get方法失败，回退使用filter过滤查询
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=output_fields,
            )
            if q:
                results.extend(_normalize_rows(q))
        except Exception as e:
            if _supports_stable_field_error(e) and FIELD_STABLE_CHUNK_ID in output_fields:
                try:
                    q = client.query(
                        collection_name=collection_name,
                        filter=expr,
                        output_fields=_without_stable_field(output_fields),
                    )
                    if q:
                        results.extend(_normalize_rows(q))
                    continue
                except Exception as retry_exc:
                    logger.error(
                        f"Milvus query方法移除stable_chunk_id后仍失败：{str(retry_exc)}",
                        exc_info=True,
                    )
            logger.error(
                f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True
            )

    # 分批查询稳定业务ID
    for i in range(0, len(stable_ids), batch_size):
        batch = stable_ids[i : i + batch_size]
        quoted = ", ".join(f'"{escape_milvus_string(value)}"' for value in batch)
        expr = f"stable_chunk_id in [{quoted}]"
        try:
            q = client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=output_fields,
            )
            if q:
                results.extend(_normalize_rows(q))
        except Exception as e:
            if _supports_stable_field_error(e) and FIELD_STABLE_CHUNK_ID in output_fields:
                try:
                    q = client.query(
                        collection_name=collection_name,
                        filter=expr,
                        output_fields=_without_stable_field(output_fields),
                    )
                    if q:
                        results.extend(_normalize_rows(q))
                    continue
                except Exception as retry_exc:
                    logger.error(
                        f"Milvus query方法移除stable_chunk_id后仍失败：{str(retry_exc)}",
                        exc_info=True,
                    )
            logger.error(
                f"Milvus query方法批量查询stable_chunk_id失败：{str(e)}",
                exc_info=True,
            )

    if not results:
        return []

    order = {value: index for index, value in enumerate(normalized_input_ids)}
    deduped = {}
    for row in results:
        business_id = str(extract_chunk_id(row) or "").strip()
        storage_id = str(extract_storage_chunk_id(row) or "").strip()
        key = business_id or storage_id
        if key and key not in deduped:
            deduped[key] = row

    sorted_rows = sorted(
        deduped.values(),
        key=lambda row: min(
            order.get(str(extract_chunk_id(row) or "").strip(), len(order)),
            order.get(str(extract_storage_chunk_id(row) or "").strip(), len(order)),
        ),
    )

    return sorted_rows


def query_chunks_by_filter(
    client,
    collection_name: str,
    *,
    filter_expr: str = "",
    output_fields=None,
    batch_size: int = 1000,
    limit: int = -1,
):
    if client is None:
        return []
    if not collection_name:
        return []
    if output_fields is None:
        output_fields = list(CHUNKS_OUTPUT_FIELDS)
    output_fields = _ensure_output_fields(output_fields)

    iterator = None
    results = []
    try:
        if hasattr(client, "query_iterator"):
            iterator = client.query_iterator(
                collection_name=collection_name,
                batch_size=batch_size,
                limit=limit,
                filter=filter_expr or "",
                output_fields=output_fields,
            )
            while True:
                batch = iterator.next()
                if not batch:
                    break
                results.extend(batch)
            return _normalize_rows(results)

        kwargs = {"filter": filter_expr or "", "output_fields": output_fields}
        if limit is not None and limit >= 0:
            kwargs["limit"] = limit
        return _normalize_rows(client.query(collection_name=collection_name, **kwargs))
    except Exception as e:
        if _supports_stable_field_error(e) and FIELD_STABLE_CHUNK_ID in output_fields:
            fallback_fields = _without_stable_field(output_fields)
            try:
                if hasattr(client, "query_iterator"):
                    iterator = client.query_iterator(
                        collection_name=collection_name,
                        batch_size=batch_size,
                        limit=limit,
                        filter=filter_expr or "",
                        output_fields=fallback_fields,
                    )
                    while True:
                        batch = iterator.next()
                        if not batch:
                            break
                        results.extend(batch)
                    return _normalize_rows(results)

                kwargs = {"filter": filter_expr or "", "output_fields": fallback_fields}
                if limit is not None and limit >= 0:
                    kwargs["limit"] = limit
                return _normalize_rows(client.query(collection_name=collection_name, **kwargs))
            except Exception as retry_exc:
                logger.error(
                    f"Milvus 按过滤条件查询移除stable_chunk_id后仍失败，集合[{collection_name}]：{str(retry_exc)}",
                    exc_info=True,
                )
        logger.error(
            f"Milvus 按过滤条件查询失败，集合[{collection_name}]：{str(e)}",
            exc_info=True,
        )
        return []
    finally:
        if iterator is not None:
            try:
                iterator.close()
            except Exception:
                pass


def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量并与建库参数保持一致
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # pymilvus AnnSearchRequest 的 expr 默认值为 ""，传入 None 可能导致异常
    safe_expr = expr or ""

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=safe_expr,
        limit=limit,
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=safe_expr,
        limit=limit,
    )

    return [dense_req, sparse_req]


def hybrid_search(
    client,
    collection_name,
    reqs,
    ranker_weights=(0.5, 0.5),
    norm_score=False,
    limit=5,
    output_fields=None,
    search_params=None,
):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算
        rerank = WeightedRanker(
            ranker_weights[0], ranker_weights[1], norm_score=norm_score
        )

        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]
        output_fields = _ensure_output_fields(output_fields)

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

        normalized = []
        for result_set in res or []:
            normalized_set = []
            for hit in result_set or []:
                if not isinstance(hit, dict):
                    normalized_set.append(hit)
                    continue
                hit_copy = dict(hit)
                entity = hit_copy.get("entity")
                if isinstance(entity, dict):
                    normalized_entity = normalize_entity_for_business_id(entity)
                    hit_copy["entity"] = normalized_entity
                    hit_copy["id"] = normalized_entity.get("chunk_id") or hit_copy.get("id")
                normalized_set.append(hit_copy)
            normalized.append(normalized_set)

        logger.info(
            f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(normalized[0]) if normalized else 0}条结果"
        )
        return normalized
    except Exception as e:
        if _supports_stable_field_error(e) and FIELD_STABLE_CHUNK_ID in output_fields:
            try:
                res = client.hybrid_search(
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker=rerank,
                    limit=limit,
                    output_fields=_without_stable_field(output_fields),
                    search_params=search_params,
                )
                normalized = []
                for result_set in res or []:
                    normalized_set = []
                    for hit in result_set or []:
                        if not isinstance(hit, dict):
                            normalized_set.append(hit)
                            continue
                        hit_copy = dict(hit)
                        entity = hit_copy.get("entity")
                        if isinstance(entity, dict):
                            normalized_entity = normalize_entity_for_business_id(entity)
                            hit_copy["entity"] = normalized_entity
                            hit_copy["id"] = normalized_entity.get("chunk_id") or hit_copy.get("id")
                        normalized_set.append(hit_copy)
                    normalized.append(normalized_set)
                logger.warning(
                    "Milvus混合搜索检测到旧集合缺少 stable_chunk_id 字段，已自动回退到兼容模式"
                )
                return normalized
            except Exception as retry_exc:
                logger.error(
                    f"Milvus混合搜索移除stable_chunk_id后仍失败，集合[{collection_name}]：{str(retry_exc)}",
                    exc_info=True,
                )
        logger.error(
            f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True
        )
        return None
