import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import (
    update_task_status,
    get_task_result,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from app.utils.sse_utils import (
    push_to_session,
    create_sse_queue,
    SSEEvent,
    sse_generator,
)
from app.utils.perf_tracker import (
    perf_start,
    perf_finish,
    get_performance_summary,
    get_performance_time_series,
    get_stage_breakdown,
)
from app.utils.eval_report_utils import (
    delete_evaluation_report,
    list_evaluation_reports,
    get_evaluation_report,
    get_latest_evaluation_report,
)
from app.utils.eval_job_utils import (
    cancel_evaluation_job,
    create_evaluation_job,
    get_evaluation_config,
    get_evaluation_job,
    list_evaluation_jobs,
    run_evaluation_job,
)
from app.utils.chunk_id_migration import migrate_collection_chunk_ids
from app.utils.unified_rag_eval import sync_evaluation_dataset
from app.clients.mongo_history_utils import (
    get_recent_messages,
    clear_history,
    get_all_sessions,
)
from app.utils.query_cache_utils import (
    get_current_request_cache_summary,
    get_query_cache_stats,
    query_cache_request_context,
    reset_query_cache,
)
from app.lm.embedding_utils import warmup_embeddings
from app.query_process.agent.main_graph import query_app

# 后续导入启动图对象
# from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="InsightVault Query Service", description="InsightVault 查询服务")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""

    query: str = Field(..., description="查询内容")  # ...必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


class EvaluationRunRequest(BaseModel):
    """统一评测触发请求"""

    dataset_path: str = Field(..., description="评测数据集路径")
    variants: list[str] = Field(default_factory=list, description="评测变体列表")
    output_path: str | None = Field(None, description="可选输出报告路径")


class EvaluationDatasetSyncRequest(BaseModel):
    """评测集 chunk_id 对齐请求"""

    dataset_path: str = Field(..., description="评测数据集路径")
    output_path: str | None = Field(
        None, description="可选输出路径，留空则原地更新数据集"
    )
    create_backup: bool = Field(True, description="原地更新时是否生成备份")


class ChunkIdMigrationRequest(BaseModel):
    """知识库稳定 chunk_id 迁移请求"""

    item_names: list[str] = Field(default_factory=list, description="可选 item_name 范围")
    collection_name: str | None = Field(None, description="可选集合名称")
    dry_run: bool = Field(False, description="仅预演，不实际写回")
    sync_graph: bool = Field(True, description="迁移后是否同步刷新 Neo4j 图谱")


class CacheResetRequest(BaseModel):
    """查询缓存重置请求"""

    reason: str = Field("manual", description="重置原因")


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


@app.post("/warmup/embeddings")
async def warmup_embedding_model():
    """
    主动预热 embedding 模型，避免首个真实查询承担冷启动成本。
    """
    try:
        result = warmup_embeddings("query service warmup")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"embedding warmup failed: {e}"
        ) from e


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    # 性能埋点：开始追踪
    perf_start(session_id, user_query)

    default_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream,
    }
    try:
        # 后期运行
        with query_cache_request_context(default_state):
            result = query_app.invoke(default_state)
            if isinstance(result, dict):
                result["cache_summary"] = result.get("cache_summary") or get_current_request_cache_summary()
        # 整体任务就更新完了！ 接下来就是数据的更新了！
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    finally:
        # 性能埋点：结束追踪并写入 MongoDB
        perf_finish(session_id)


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    print(
        "开始处理流程... 是否流式:",
        is_stream,
        f"其他参数:{user_query}, session_id:{session_id}",
    )

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        # 返回结果
        print("开始处理结果....")
        return {"message": "结果正在处理中...", "session_id": session_id}
    else:
        # 同步运行
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        metadata = get_task_result(session_id, "metadata", {})
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "metadata": metadata,
            "done_list": [],
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print("调用流式/stream...")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append(
                {
                    "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                    "session_id": r.get("session_id", ""),
                    "role": r.get("role", ""),
                    "text": r.get("text", ""),
                    "rewritten_query": r.get("rewritten_query", ""),
                    "item_names": r.get("item_names", []),
                    "image_urls": r.get("image_urls", []),
                    "metadata": r.get("metadata", {}),
                    "ts": r.get("ts"),
                }
            )
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


@app.get("/sessions")
async def list_sessions(limit: int = 50):
    """
    获取所有会话列表，按最后消息时间倒序
    """
    try:
        sessions = get_all_sessions(limit=limit)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sessions error: {e}")


# ─── 性能分析 API ─────────────────────────────────────────────


@app.get("/performance/summary")
async def performance_summary(start_date: str = None, end_date: str = None):
    """获取性能摘要统计"""
    try:
        return get_performance_summary(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"performance summary error: {e}")


@app.get("/performance/time-series")
async def performance_time_series(
    granularity: str = "day", start_date: str = None, end_date: str = None
):
    """获取性能时间序列数据"""
    try:
        return get_performance_time_series(granularity, start_date, end_date)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"performance time-series error: {e}"
        )


@app.get("/performance/stages")
async def performance_stages(start_date: str = None, end_date: str = None):
    """获取阶段耗时分布"""
    try:
        return get_stage_breakdown(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"performance stages error: {e}")


# ─── 量化评测 API ─────────────────────────────────────────────


@app.get("/evaluation/reports")
async def evaluation_reports():
    """获取统一评测报告列表"""
    try:
        return list_evaluation_reports()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation reports error: {e}")


@app.get("/evaluation/reports/latest")
async def latest_evaluation_report():
    """获取最新的统一评测报告"""
    try:
        return {"report": get_latest_evaluation_report()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"latest evaluation report error: {e}"
        )


@app.get("/evaluation/reports/{report_id}")
async def evaluation_report(report_id: str):
    """获取指定统一评测报告"""
    try:
        report = get_evaluation_report(report_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation report error: {e}")
    if report is None:
        raise HTTPException(status_code=404, detail="evaluation report not found")
    return report


@app.delete("/evaluation/reports/{report_id}")
async def delete_evaluation_report_api(report_id: str):
    """删除指定统一评测报告"""
    try:
        return delete_evaluation_report(report_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="evaluation report not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"delete evaluation report error: {e}"
        )


@app.get("/evaluation/config")
async def evaluation_config():
    """获取评测运行配置"""
    try:
        return get_evaluation_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation config error: {e}")


@app.get("/evaluation/jobs")
async def evaluation_jobs(limit: int = 10):
    """获取最近的评测任务列表"""
    try:
        return list_evaluation_jobs(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation jobs error: {e}")


@app.get("/evaluation/jobs/{job_id}")
async def evaluation_job(job_id: str):
    """获取指定评测任务状态"""
    try:
        job = get_evaluation_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation job error: {e}")
    if job is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    return job


@app.post("/evaluation/jobs/{job_id}/cancel")
async def cancel_evaluation_job_api(job_id: str):
    """取消指定评测任务"""
    try:
        job = cancel_evaluation_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cancel evaluation job error: {e}")
    if job is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    return job


@app.post("/evaluation/jobs")
async def create_and_run_evaluation_job(
    background_tasks: BackgroundTasks, request: EvaluationRunRequest
):
    """创建并后台执行统一评测"""
    try:
        job = create_evaluation_job(
            dataset_path=request.dataset_path,
            variants=request.variants,
            output_path=request.output_path,
        )
        background_tasks.add_task(run_evaluation_job, job["job_id"])
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"create evaluation job error: {e}")


@app.post("/evaluation/dataset/sync")
async def sync_evaluation_dataset_api(request: EvaluationDatasetSyncRequest):
    """将评测集中的历史 chunk_id 对齐到当前知识库"""
    try:
        return sync_evaluation_dataset(
            dataset_path=request.dataset_path,
            output_path=request.output_path,
            create_backup=request.create_backup,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"sync evaluation dataset error: {e}"
        )


@app.post("/knowledge-base/chunk-ids/migrate")
async def migrate_chunk_ids_api(request: ChunkIdMigrationRequest):
    """将知识库现有 chunk 对齐到稳定 chunk_id"""
    try:
        result = migrate_collection_chunk_ids(
            item_names=request.item_names,
            collection_name=request.collection_name,
            dry_run=request.dry_run,
            sync_graph=request.sync_graph,
        )
        if not request.dry_run:
            cache_reset = reset_query_cache(reason="knowledge_base_chunk_id_migration")
            result["cache_reset"] = cache_reset
            message = str(result.get("message") or "").strip()
            result["message"] = (
                f"{message} 查询缓存已失效。".strip()
                if message
                else "知识库迁移完成，查询缓存已失效。"
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"migrate chunk ids error: {e}")


@app.get("/cache/stats")
async def query_cache_stats_api():
    """获取查询缓存全局统计"""
    try:
        return get_query_cache_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query cache stats error: {e}")


@app.post("/cache/reset")
async def reset_query_cache_api(request: CacheResetRequest):
    """手动失效查询缓存"""
    try:
        return reset_query_cache(reason=request.reason)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query cache reset error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
