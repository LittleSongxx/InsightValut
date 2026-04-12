"""
Neo4j 知识图谱客户端工具
集合：Product / Chunk 节点 + HAS_CHUNK / NEXT 关系
用途：将产品文档切片导入图数据库，查询时通过图谱关系召回相关切片
"""

import os
import logging
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --------------- 单例驱动 ---------------
_driver = None


def get_neo4j_driver():
    """获取 Neo4j 驱动单例"""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "insightvault123")
        _driver = GraphDatabase.driver(uri, auth=(user, pwd))
        logger.info(f"Neo4j 驱动初始化完成: {uri}")
    return _driver


def close_neo4j_driver():
    """关闭 Neo4j 驱动"""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j 驱动已关闭")


def _get_database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


# --------------- 索引初始化 ---------------
_indexes_created = False


def ensure_indexes():
    """确保 Neo4j 中存在必要的索引（仅执行一次）"""
    global _indexes_created
    if _indexes_created:
        return
    driver = get_neo4j_driver()
    with driver.session(database=_get_database()) as session:
        # Product.name 唯一约束
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS "
            "FOR (p:Product) REQUIRE p.name IS UNIQUE"
        )
        # Chunk.chunk_id 唯一约束
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )
        # item_name 索引加速查询
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.item_name)"
        )
    _indexes_created = True
    logger.info("Neo4j 索引/约束初始化完成")


# --------------- 写入操作（导入流程） ---------------

def import_chunks_to_kg(item_name: str, chunks: List[Dict[str, Any]]) -> int:
    """
    将产品切片批量导入 Neo4j 知识图谱（幂等：先删旧数据再写入）

    图谱结构：
      (:Product {name}) -[:HAS_CHUNK]-> (:Chunk {chunk_id, content, title, part, item_name})
      (:Chunk)-[:NEXT]->(:Chunk)  按 part 顺序串联

    :param item_name: 产品/商品名称
    :param chunks: 切片列表，每个含 chunk_id, content, title, part, item_name 等字段
    :return: 实际写入的切片数量
    """
    if not item_name or not chunks:
        logger.warning("import_chunks_to_kg: item_name 或 chunks 为空，跳过")
        return 0

    ensure_indexes()
    driver = get_neo4j_driver()

    with driver.session(database=_get_database()) as session:
        # 1. 幂等清理：删除该产品的旧切片和关系
        session.run(
            "MATCH (p:Product {name: $name})-[r:HAS_CHUNK]->(c:Chunk) "
            "DETACH DELETE c",
            name=item_name,
        )
        logger.info(f"Neo4j 幂等清理完成: 已删除 {item_name} 的旧切片")

        # 2. MERGE Product 节点（保证幂等）
        session.run(
            "MERGE (p:Product {name: $name})",
            name=item_name,
        )

        # 3. 批量创建 Chunk 节点 + HAS_CHUNK 关系
        # 按 part 排序，方便后续建立 NEXT 链
        sorted_chunks = sorted(chunks, key=lambda c: c.get("part", 0))

        chunk_data = []
        for c in sorted_chunks:
            cid = str(c.get("chunk_id", ""))
            if not cid:
                continue
            chunk_data.append({
                "chunk_id": cid,
                "content": (c.get("content") or "")[:2000],
                "title": c.get("title") or "",
                "parent_title": c.get("parent_title") or "",
                "part": c.get("part", 0),
                "item_name": item_name,
                "file_title": c.get("file_title") or "",
                "image_urls": c.get("image_urls") or [],
            })

        if not chunk_data:
            logger.warning(f"Neo4j 导入跳过: {item_name} 无有效 chunk_id")
            return 0

        # 使用 UNWIND 批量写入
        session.run(
            """
            UNWIND $chunks AS cd
            MATCH (p:Product {name: $name})
            CREATE (c:Chunk {
                chunk_id:     cd.chunk_id,
                content:      cd.content,
                title:        cd.title,
                parent_title: cd.parent_title,
                part:         cd.part,
                item_name:    cd.item_name,
                file_title:   cd.file_title,
                image_urls:   cd.image_urls
            })
            CREATE (p)-[:HAS_CHUNK]->(c)
            """,
            name=item_name,
            chunks=chunk_data,
        )

        # 4. 建立 NEXT 链（按 part 顺序）
        if len(chunk_data) > 1:
            session.run(
                """
                MATCH (p:Product {name: $name})-[:HAS_CHUNK]->(c:Chunk)
                WITH c ORDER BY c.part
                WITH collect(c) AS nodes
                UNWIND range(0, size(nodes)-2) AS i
                WITH nodes[i] AS a, nodes[i+1] AS b
                CREATE (a)-[:NEXT]->(b)
                """,
                name=item_name,
            )

        logger.info(f"Neo4j 导入完成: {item_name}, 写入 {len(chunk_data)} 个切片节点")
        return len(chunk_data)


def delete_product(item_name: str) -> None:
    """删除产品及其所有关联切片"""
    driver = get_neo4j_driver()
    with driver.session(database=_get_database()) as session:
        session.run(
            "MATCH (p:Product {name: $name}) "
            "OPTIONAL MATCH (p)-[*]->(n) "
            "DETACH DELETE p, n",
            name=item_name,
        )
    logger.info(f"Neo4j 删除产品: {item_name}")


# --------------- 查询操作（查询流程） ---------------

def query_chunks_by_product(
    item_names: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    根据产品名称列表，从知识图谱中检索关联切片

    :param item_names: 产品名称列表
    :param limit: 每个产品最多返回的切片数
    :return: 切片字典列表，包含 chunk_id, content, title, item_name, source="kg"
    """
    if not item_names:
        return []

    ensure_indexes()
    driver = get_neo4j_driver()
    results = []

    with driver.session(database=_get_database()) as session:
        records = session.run(
            """
            MATCH (p:Product)-[:HAS_CHUNK]->(c:Chunk)
            WHERE p.name IN $names
            RETURN c.chunk_id   AS chunk_id,
                   c.content    AS content,
                   c.title      AS title,
                   c.item_name  AS item_name,
                   c.part       AS part,
                   c.image_urls AS image_urls
            ORDER BY c.part
            LIMIT $limit
            """,
            names=item_names,
            limit=limit * len(item_names),
        )
        for r in records:
            results.append({
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "title": r["title"],
                "item_name": r["item_name"],
                "part": r["part"],
                "image_urls": r["image_urls"] or [],
                "source": "kg",
            })

    logger.info(
        f"Neo4j 查询完成: item_names={item_names}, 返回 {len(results)} 条切片"
    )
    return results


def query_related_products(item_name: str, limit: int = 5) -> List[str]:
    """
    查询与指定产品共享切片关键词的相关产品（未来可扩展实体关系）

    :param item_name: 产品名称
    :param limit: 最多返回数量
    :return: 相关产品名称列表
    """
    driver = get_neo4j_driver()
    with driver.session(database=_get_database()) as session:
        records = session.run(
            """
            MATCH (p:Product)
            WHERE p.name <> $name
            RETURN p.name AS name
            LIMIT $limit
            """,
            name=item_name,
            limit=limit,
        )
        return [r["name"] for r in records]


def verify_connection() -> bool:
    """验证 Neo4j 连接是否可用"""
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        logger.info("Neo4j 连接验证成功")
        return True
    except Exception as e:
        logger.error(f"Neo4j 连接验证失败: {e}")
        return False
