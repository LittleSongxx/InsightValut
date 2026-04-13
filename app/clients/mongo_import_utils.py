"""
导入任务 MongoDB 持久化工具
集合名：import_tasks
用途：持久化保存每个文件导入任务的元信息和处理状态，服务重启后仍可查询历史任务
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv

load_dotenv()

# --------------- 单例连接 ---------------
_collection = None


def _get_collection():
    """获取 import_tasks 集合（懒加载单例）"""
    global _collection
    if _collection is None:
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME") or "insightvault_rag"
        if not mongo_url:
            raise RuntimeError("MONGO_URL is not configured")
        client = MongoClient(mongo_url)
        db = client[db_name]
        _collection = db["import_tasks"]
        # 创建索引
        _collection.create_index("task_id", unique=True)
        _collection.create_index([("created_at", DESCENDING)])
        _collection.create_index("status")
        logging.info("import_tasks MongoDB collection initialized")
    return _collection


# --------------- CRUD ---------------


def create_import_task(
    task_id: str,
    file_name: str,
    status: str = "pending",
) -> None:
    """
    创建一条导入任务记录（上传时调用）
    """
    col = _get_collection()
    now = datetime.now().timestamp()
    doc = {
        "task_id": task_id,
        "file_name": file_name,
        "file_title": os.path.splitext(file_name)[0],
        "status": status,
        "done_list": [],
        "running_list": [],
        "error_message": "",
        "item_name": "",
        "created_at": now,
        "updated_at": now,
    }
    try:
        col.insert_one(doc)
        logging.info(f"[{task_id}] 导入任务记录已创建: {file_name}")
    except Exception as e:
        logging.error(f"[{task_id}] 创建导入任务记录失败: {e}")


def update_import_task(task_id: str, **fields) -> None:
    """
    更新导入任务的任意字段（如 status / done_list / running_list / error_message / item_name）
    """
    col = _get_collection()
    fields["updated_at"] = datetime.now().timestamp()
    try:
        col.update_one({"task_id": task_id}, {"$set": fields})
    except Exception as e:
        logging.error(f"[{task_id}] 更新导入任务记录失败: {e}")


def get_import_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    查询单个导入任务（按 task_id）
    """
    col = _get_collection()
    try:
        doc = col.find_one({"task_id": task_id}, {"_id": 0})
        return doc
    except Exception as e:
        logging.error(f"[{task_id}] 查询导入任务记录失败: {e}")
        return None


def get_import_tasks_by_ids(task_ids: List[str]) -> List[Dict[str, Any]]:
    col = _get_collection()
    cleaned_task_ids = [task_id for task_id in task_ids if task_id]
    if not cleaned_task_ids:
        return []
    try:
        cursor = col.find({"task_id": {"$in": cleaned_task_ids}}, {"_id": 0})
        return list(cursor)
    except Exception as e:
        logging.error(f"批量查询导入任务记录失败: {e}")
        return []


def list_import_tasks(limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
    """
    查询导入任务列表，按创建时间倒序
    :param limit: 最大返回条数
    :param status: 可选状态过滤（pending/processing/completed/failed）
    """
    col = _get_collection()
    try:
        query = {}
        if status:
            query["status"] = status
        cursor = col.find(query, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logging.error(f"查询导入任务列表失败: {e}")
        return []


def delete_import_task(task_id: str) -> int:
    """删除单个导入任务记录"""
    col = _get_collection()
    try:
        result = col.delete_one({"task_id": task_id})
        return result.deleted_count
    except Exception as e:
        logging.error(f"[{task_id}] 删除导入任务记录失败: {e}")
        return 0
