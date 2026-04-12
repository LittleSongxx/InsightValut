"""
Milvus Schema 统一字段定义

所有 Milvus collection 的 Schema 应从此模块导入，确保：
1. 字段名称、类型、长度在全项目范围内保持一致
2. 修改字段定义时只需改一处，避免遗漏
3. output_fields 配置统一管理
"""

from pymilvus import DataType


# ==================== CHUNKS_COLLECTION 字段定义 ====================

# CHUNKS_COLLECTION 字段名称常量
FIELD_CHUNK_ID = "chunk_id"
FIELD_CONTENT = "content"
FIELD_TITLE = "title"
FIELD_PARENT_TITLE = "parent_title"
FIELD_PART = "part"
FIELD_FILE_TITLE = "file_title"
FIELD_ITEM_NAME = "item_name"
FIELD_IMAGE_URLS = "image_urls"
FIELD_SPARSE_VECTOR = "sparse_vector"
FIELD_DENSE_VECTOR = "dense_vector"

# CHUNKS_COLLECTION 索引字段
CHUNKS_VECTOR_FIELDS = [FIELD_DENSE_VECTOR, FIELD_SPARSE_VECTOR]

# CHUNKS_COLLECTION 查询时返回的字段（不含向量，节省带宽）
CHUNKS_OUTPUT_FIELDS = [
    FIELD_CHUNK_ID,
    FIELD_CONTENT,
    FIELD_TITLE,
    FIELD_PARENT_TITLE,
    FIELD_PART,
    FIELD_FILE_TITLE,
    FIELD_ITEM_NAME,
    FIELD_IMAGE_URLS,
]

# ==================== ITEM_NAME_COLLECTION 字段定义 ====================

ITEM_NAME_OUTPUT_FIELDS = [
    FIELD_CHUNK_ID,
    FIELD_CONTENT,
    FIELD_TITLE,
    FIELD_PARENT_TITLE,
    FIELD_ITEM_NAME,
    FIELD_IMAGE_URLS,
]

# ==================== 统一 entity 字段访问工具 ====================


def get_entity_field(entity: dict, field: str, default=None):
    """
    从 Milvus entity 中安全获取字段值（兼容新旧数据格式）。

    兼容场景：
    - 直接字段访问：entity["content"]
    - 嵌套在 entity 中：doc["entity"]["content"]
    - 字段不存在时返回默认值

    :param entity: Milvus 返回的 entity 字典
    :param field: 字段名称
    :param default: 字段不存在时的默认值
    :return: 字段值或默认值
    """
    if entity is None:
        return default
    # 尝试直接访问
    if field in entity:
        return entity[field]
    # 尝试从嵌套 entity 中访问（RRF/检索结果格式）
    if "entity" in entity:
        nested = entity["entity"]
        if isinstance(nested, dict) and field in nested:
            return nested[field]
    return default


def entity_to_doc(entity: dict, source: str = "local") -> dict:
    """
    将 Milvus entity 转换为标准化文档格式（用于 rerank/answer 节点）。

    统一各路检索结果（embedding/hyde/kg/web）的字段命名，
    确保下游节点（rerank、answer_output）使用统一字段访问方式。

    输出字段映射：
      chunk_id / id       -> chunk_id
      content             -> content
      title               -> title
      parent_title        -> parent_title
      item_name           -> item_name
      image_urls          -> image_urls
      source              -> source (传入参数)
      score               -> score (若有)

    :param entity: Milvus entity 字典
    :param source: 来源标识 (local/web/kg)
    :return: 标准化文档字典
    """
    doc = {
        "chunk_id": get_entity_field(entity, FIELD_CHUNK_ID) or get_entity_field(entity, "id"),
        "content": get_entity_field(entity, FIELD_CONTENT),
        "title": get_entity_field(entity, FIELD_TITLE),
        "parent_title": get_entity_field(entity, FIELD_PARENT_TITLE),
        "item_name": get_entity_field(entity, FIELD_ITEM_NAME),
        "image_urls": get_entity_field(entity, FIELD_IMAGE_URLS) or [],
        "source": source,
        "score": entity.get("score") if isinstance(entity, dict) else None,
    }
    return doc


def extract_chunk_content(entity: dict) -> str:
    """
    从 entity 中安全提取文本内容。

    :param entity: Milvus entity 字典
    :return: chunk content 文本
    """
    return (get_entity_field(entity, FIELD_CONTENT) or "").strip()


def extract_chunk_id(entity: dict):
    """
    从 entity 中安全提取 chunk_id。

    :param entity: Milvus entity 字典
    :return: chunk_id 或 None
    """
    return get_entity_field(entity, FIELD_CHUNK_ID) or get_entity_field(entity, "id")


def extract_image_urls(entity: dict) -> list:
    """
    从 entity 中安全提取图片 URL 列表。

    兼容格式：
    - 列表：["url1", "url2"]
    - 字符串（JSON）：'["url1", "url2"]'
    - None 或空

    :param entity: Milvus entity 字典
    :return: URL 列表
    """
    urls = get_entity_field(entity, FIELD_IMAGE_URLS, [])
    if urls is None:
        return []
    if isinstance(urls, str):
        import json

        try:
            urls = json.loads(urls)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(urls, list):
        return [u for u in urls if u]
    return []
