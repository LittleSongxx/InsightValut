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
from app.clients.mongo_history_utils import (
    get_recent_messages,
    clear_history,
    get_all_sessions,
)
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


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


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
        query_app.invoke(default_state)
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
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
