"""
性能埋点工具模块
在 LangGraph 各节点记录耗时数据，写入 MongoDB performance_records 集合。
参考 CookHero 的性能量化方案。
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── MongoDB 连接 ─────────────────────────────────────────────
_perf_collection = None


def _get_perf_collection():
    """获取 performance_records 集合（懒加载单例）"""
    global _perf_collection
    if _perf_collection is None:
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        client = MongoClient(mongo_url)
        db = client[db_name]
        _perf_collection = db["performance_records"]
        # 创建索引
        _perf_collection.create_index([("session_id", ASCENDING)])
        _perf_collection.create_index([("created_at", DESCENDING)])
        logger.info("Performance collection initialized")
    return _perf_collection


# ─── 内存态：单次请求的阶段计时 ────────────────────────────────
# key: session_id, value: PerfSession
_active_sessions: Dict[str, "PerfSession"] = {}


class PerfSession:
    """单个查询请求的性能追踪会话"""

    def __init__(self, session_id: str, query: str = ""):
        self.session_id = session_id
        self.query = query
        self.start_time = time.time()
        self.first_answer_time: Optional[float] = None
        self.stages: List[Dict[str, Any]] = []
        self._stage_starts: Dict[str, float] = {}

    def begin_stage(self, stage_name: str):
        """标记阶段开始"""
        self._stage_starts[stage_name] = time.time()

    def end_stage(self, stage_name: str, status: str = "success", error: str = ""):
        """标记阶段结束并记录耗时"""
        start = self._stage_starts.pop(stage_name, None)
        if start is None:
            return
        duration_ms = (time.time() - start) * 1000
        self.stages.append({
            "stage": stage_name,
            "duration_ms": round(duration_ms, 2),
            "status": status,
            "error": error,
        })

    def mark_first_answer(self):
        """标记首次产生回答的时间"""
        if self.first_answer_time is None:
            self.first_answer_time = time.time()

    def to_document(self) -> Dict[str, Any]:
        """转为 MongoDB 文档格式"""
        total_duration_ms = (time.time() - self.start_time) * 1000
        first_answer_ms = None
        if self.first_answer_time:
            first_answer_ms = (self.first_answer_time - self.start_time) * 1000

        return {
            "session_id": self.session_id,
            "query": self.query,
            "total_duration_ms": round(total_duration_ms, 2),
            "first_answer_ms": round(first_answer_ms, 2) if first_answer_ms else None,
            "stages": self.stages,
            "stage_count": len(self.stages),
            "created_at": datetime.utcnow(),
        }


# ─── 公共 API ─────────────────────────────────────────────────

def perf_start(session_id: str, query: str = ""):
    """开始追踪一次查询请求"""
    session = PerfSession(session_id, query)
    _active_sessions[session_id] = session
    return session


def perf_begin_stage(session_id: str, stage_name: str):
    """标记阶段开始"""
    session = _active_sessions.get(session_id)
    if session:
        session.begin_stage(stage_name)


def perf_end_stage(session_id: str, stage_name: str, status: str = "success", error: str = ""):
    """标记阶段结束"""
    session = _active_sessions.get(session_id)
    if session:
        session.end_stage(stage_name, status, error)


def perf_mark_first_answer(session_id: str):
    """标记首次产生回答"""
    session = _active_sessions.get(session_id)
    if session:
        session.mark_first_answer()


def perf_finish(session_id: str):
    """结束追踪并写入 MongoDB"""
    session = _active_sessions.pop(session_id, None)
    if session is None:
        return

    try:
        doc = session.to_document()
        collection = _get_perf_collection()
        collection.insert_one(doc)
        logger.info(
            f"Performance record saved: session={session_id}, "
            f"total={doc['total_duration_ms']:.0f}ms, stages={doc['stage_count']}"
        )
    except Exception as e:
        logger.error(f"Failed to save performance record for {session_id}: {e}")


# ─── 查询 API（供性能看板调用）────────────────────────────────

def get_performance_summary(start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """获取性能摘要统计"""
    collection = _get_perf_collection()
    query: Dict[str, Any] = {}
    if start_date or end_date:
        query["created_at"] = {}
        if start_date:
            query["created_at"]["$gte"] = datetime.fromisoformat(start_date)
        if end_date:
            query["created_at"]["$lte"] = datetime.fromisoformat(end_date)

    pipeline = [
        {"$match": query} if query else {"$match": {}},
        {
            "$group": {
                "_id": None,
                "run_count": {"$sum": 1},
                "avg_total_duration_ms": {"$avg": "$total_duration_ms"},
                "durations": {"$push": "$total_duration_ms"},
                "avg_first_answer_ms": {"$avg": "$first_answer_ms"},
            }
        },
    ]

    results = list(collection.aggregate(pipeline))
    if not results:
        return {
            "run_count": 0,
            "avg_total_duration_ms": 0,
            "p50_total_duration_ms": 0,
            "p95_total_duration_ms": 0,
            "avg_first_answer_ms": None,
            "stages": [],
        }

    data = results[0]
    durations = sorted(data.get("durations", []))
    n = len(durations)

    p50 = durations[n // 2] if n > 0 else 0
    p95_idx = min(int(n * 0.95), n - 1)
    p95 = durations[p95_idx] if n > 0 else 0

    # 获取阶段统计
    stage_pipeline = [
        {"$match": query} if query else {"$match": {}},
        {"$unwind": "$stages"},
        {
            "$group": {
                "_id": "$stages.stage",
                "count": {"$sum": 1},
                "avg_duration_ms": {"$avg": "$stages.duration_ms"},
                "durations": {"$push": "$stages.duration_ms"},
                "errors": {
                    "$sum": {"$cond": [{"$eq": ["$stages.status", "error"]}, 1, 0]}
                },
            }
        },
        {"$sort": {"avg_duration_ms": -1}},
    ]
    stage_results = list(collection.aggregate(stage_pipeline))
    stages = []
    for sr in stage_results:
        sd = sorted(sr.get("durations", []))
        sn = len(sd)
        sp95_idx = min(int(sn * 0.95), sn - 1)
        stages.append({
            "stage": sr["_id"],
            "count": sr["count"],
            "avg_duration_ms": round(sr["avg_duration_ms"], 2),
            "p95_duration_ms": round(sd[sp95_idx], 2) if sn > 0 else 0,
            "error_rate": round(sr["errors"] / sr["count"], 4) if sr["count"] > 0 else 0,
        })

    return {
        "run_count": data["run_count"],
        "avg_total_duration_ms": round(data["avg_total_duration_ms"], 2),
        "p50_total_duration_ms": round(p50, 2),
        "p95_total_duration_ms": round(p95, 2),
        "avg_first_answer_ms": round(data["avg_first_answer_ms"], 2) if data.get("avg_first_answer_ms") else None,
        "stages": stages,
    }


def get_performance_time_series(
    granularity: str = "day", start_date: str = None, end_date: str = None
) -> List[Dict[str, Any]]:
    """获取性能时间序列数据"""
    collection = _get_perf_collection()
    query: Dict[str, Any] = {}
    if start_date or end_date:
        query["created_at"] = {}
        if start_date:
            query["created_at"]["$gte"] = datetime.fromisoformat(start_date)
        if end_date:
            query["created_at"]["$lte"] = datetime.fromisoformat(end_date)

    if granularity == "hour":
        date_format = "%Y-%m-%dT%H:00:00"
    else:
        date_format = "%Y-%m-%dT00:00:00"

    pipeline = [
        {"$match": query} if query else {"$match": {}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": date_format, "date": "$created_at"}},
                "run_count": {"$sum": 1},
                "avg_total_duration_ms": {"$avg": "$total_duration_ms"},
                "durations": {"$push": "$total_duration_ms"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    results = list(collection.aggregate(pipeline))
    time_series = []
    for r in results:
        durations = sorted(r.get("durations", []))
        n = len(durations)
        p95_idx = min(int(n * 0.95), n - 1)
        time_series.append({
            "period": r["_id"],
            "run_count": r["run_count"],
            "avg_total_duration_ms": round(r["avg_total_duration_ms"], 2),
            "p95_total_duration_ms": round(durations[p95_idx], 2) if n > 0 else 0,
        })

    return time_series


def get_stage_breakdown(start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """获取阶段耗时分布"""
    collection = _get_perf_collection()
    query: Dict[str, Any] = {}
    if start_date or end_date:
        query["created_at"] = {}
        if start_date:
            query["created_at"]["$gte"] = datetime.fromisoformat(start_date)
        if end_date:
            query["created_at"]["$lte"] = datetime.fromisoformat(end_date)

    pipeline = [
        {"$match": query} if query else {"$match": {}},
        {"$unwind": "$stages"},
        {
            "$group": {
                "_id": "$stages.stage",
                "count": {"$sum": 1},
                "avg_duration_ms": {"$avg": "$stages.duration_ms"},
                "durations": {"$push": "$stages.duration_ms"},
                "errors": {
                    "$sum": {"$cond": [{"$eq": ["$stages.status", "error"]}, 1, 0]}
                },
            }
        },
        {"$sort": {"avg_duration_ms": -1}},
    ]

    results = list(collection.aggregate(pipeline))
    stages = []
    for sr in results:
        sd = sorted(sr.get("durations", []))
        sn = len(sd)
        sp95_idx = min(int(sn * 0.95), sn - 1)
        stages.append({
            "stage": sr["_id"],
            "count": sr["count"],
            "avg_duration_ms": round(sr["avg_duration_ms"], 2),
            "p95_duration_ms": round(sd[sp95_idx], 2) if sn > 0 else 0,
            "error_rate": round(sr["errors"] / sr["count"], 4) if sr["count"] > 0 else 0,
        })

    return stages
