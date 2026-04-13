import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn

# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.clients.milvus_utils import get_milvus_client
from app.clients.neo4j_graph_utils import delete_product_graph
from app.utils.path_util import PROJECT_ROOT
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
    set_task_result,
    get_task_result,
    clear_task,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具
from app.clients.mongo_import_utils import (
    create_import_task,
    update_import_task,
    get_import_task,
    get_import_tasks_by_ids,
    list_import_tasks,
    delete_import_task,
)

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="InsightVault Import Service",
    description="InsightVault 导入服务（PDF/MD -> 解析 -> 切分 -> 向量化 -> 入库）",
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/health")
async def health():
    """检查服务是否正常"""
    return {"ok": True}


class DeleteImportTasksRequest(BaseModel):
    task_ids: List[str]


def _delete_milvus_records_by_item_name(item_name: str, collection_name: str) -> None:
    cleaned_item_name = (item_name or "").strip()
    if not cleaned_item_name or not collection_name:
        return
    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus client is unavailable")
    if not client.has_collection(collection_name=collection_name):
        return
    if hasattr(client, "load_collection"):
        client.load_collection(collection_name=collection_name)
    safe_item_name = escape_milvus_string(cleaned_item_name)
    client.delete(
        collection_name=collection_name,
        filter=f'item_name == "{safe_item_name}"',
    )
    if hasattr(client, "flush"):
        try:
            client.flush(collection_name=collection_name)
        except Exception:
            pass


def _resolve_task_local_dir(task: Dict[str, Any]) -> str:
    local_dir = (task.get("local_dir") or "").strip()
    if local_dir:
        return local_dir
    task_id = (task.get("task_id") or "").strip()
    created_at = task.get("created_at")
    if not task_id or not created_at:
        return ""
    date_dir = datetime.fromtimestamp(created_at).strftime("%Y%m%d")
    return os.path.join(PROJECT_ROOT, "output", date_dir, task_id)


def _delete_task_local_dir(task: Dict[str, Any]) -> None:
    local_dir = _resolve_task_local_dir(task)
    if local_dir and os.path.isdir(local_dir):
        shutil.rmtree(local_dir, ignore_errors=True)


# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    """
    final_state: Dict[str, Any] = {}
    resolved_file_title = os.path.splitext(os.path.basename(local_file_path))[0]
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        update_import_task(
            task_id,
            status="processing",
            done_list=get_done_task_list(task_id),
            running_list=get_running_task_list(task_id),
            file_title=resolved_file_title,
        )
        logger.info(
            f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}"
        )

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径
        final_state = init_state

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                if isinstance(node_result, dict):
                    final_state = node_result
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)
                update_import_task(
                    task_id,
                    status="processing",
                    done_list=get_done_task_list(task_id),
                    running_list=get_running_task_list(task_id),
                    item_name=(final_state.get("item_name") or "").strip(),
                    file_title=(
                        final_state.get("file_title") or resolved_file_title
                    ).strip(),
                )

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        update_import_task(
            task_id,
            status="completed",
            done_list=get_done_task_list(task_id),
            running_list=[],
            item_name=(final_state.get("item_name") or "").strip(),
            file_title=(final_state.get("file_title") or resolved_file_title).strip(),
        )
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        set_task_result(task_id, "error", str(e))
        update_import_task(
            task_id,
            status="failed",
            error_message=str(e),
            done_list=get_done_task_list(task_id),
            running_list=[],
            item_name=(final_state.get("item_name") or "").strip(),
            file_title=(final_state.get("file_title") or resolved_file_title).strip(),
        )
        logger.error(
            f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True
        )


# --------------------------
# 核心接口：文件上传接口
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
@app.post(
    "/upload",
    summary="文件上传接口",
    description="支持多文件批量上传，自动触发知识库导入全流程",
)
async def upload_files(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    date_based_root_dir = os.path.join(
        PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d")
    )
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(
            f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}"
        )

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")
        create_import_task(task_id, file.filename, status="uploading")
        update_import_task(
            task_id,
            status="uploading",
            done_list=get_done_task_list(task_id),
            running_list=get_running_task_list(task_id),
        )

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)  # 目录不存在则创建，存在则不做处理
        # 构建上传文件的本地保存绝对路径
        local_file_abs_path = os.path.join(task_local_dir, file.filename)
        update_import_task(task_id, local_dir=task_local_dir)

        # 5. 将上传的文件保存到本地临时目录（后续MinIO上传/文件解析均基于此文件）
        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 6. 将本地文件上传至MinIO对象存储，做持久化保存
        # 从环境变量获取MinIO的PDF存储目录配置
        minio_pdf_base_dir = os.getenv(
            "MINIO_PDF_DIR", "pdf_files"
        )  # 缺省值：pdf_files
        # 构建MinIO中的文件对象名：配置目录/YYYYMMDD/文件名（按日期分层，和本地一致）
        minio_object_name = (
            f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        )
        try:
            # 获取MinIO客户端实例
            minio_client = get_minio_client()
            if minio_client is None:
                # MinIO客户端获取失败，抛出500服务异常
                raise HTTPException(
                    status_code=500,
                    detail="MinIO service connection failed, please check MinIO config",
                )
            # 从环境变量获取MinIO的桶名配置
            minio_bucket_name = os.getenv(
                "MINIO_BUCKET_NAME", "kb-import-bucket"
            )  # 缺省值：kb-import-bucket

            # 本地文件上传至MinIO（同名文件会自动覆盖，保证文件最新）
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type,  # 传递文件原始MIME类型
            )
            logger.info(
                f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}"
            )
        except Exception as e:
            # MinIO上传失败，记录警告日志（不中断后续流程，本地文件仍可继续处理）
            logger.warning(
                f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}",
                exc_info=True,
            )

        # 7. 标记「文件上传」阶段为「已完成」，前端轮询可查
        add_done_task(task_id, "upload_file")
        update_import_task(
            task_id,
            status="uploading",
            done_list=get_done_task_list(task_id),
            running_list=get_running_task_list(task_id),
        )

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        background_tasks.add_task(
            run_graph_task, task_id, task_local_dir, local_file_abs_path
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID（前端基于TaskID轮询进度）
    logger.info(
        f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}"
    )
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids,
    }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get(
    "/status/{task_id}",
    summary="任务状态查询",
    description="根据TaskID查询单个文件的处理进度和全局状态",
)
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 优先从内存获取（处理中的任务实时性最好）
    mem_status = get_task_status(task_id)
    db_task = get_import_task(task_id)

    if mem_status:
        # 内存中有数据 → 正在处理或刚完成
        task_status_info: Dict[str, Any] = {
            "code": 200,
            "task_id": task_id,
            "status": mem_status,
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
            "error_message": get_task_result(task_id, "error"),
            "item_name": (db_task or {}).get("item_name", ""),
            "file_name": (db_task or {}).get("file_name", ""),
            "file_title": (db_task or {}).get("file_title", ""),
        }
    else:
        # 内存中无数据（服务重启过） → 回退到 MongoDB 持久化记录
        if db_task:
            task_status_info = {
                "code": 200,
                "task_id": task_id,
                "status": db_task.get("status", ""),
                "done_list": db_task.get("done_list", []),
                "running_list": db_task.get("running_list", []),
                "error_message": db_task.get("error_message", ""),
                "item_name": db_task.get("item_name", ""),
                "file_name": db_task.get("file_name", ""),
                "file_title": db_task.get("file_title", ""),
            }
        else:
            task_status_info = {
                "code": 404,
                "task_id": task_id,
                "status": "",
                "done_list": [],
                "running_list": [],
                "error_message": "任务不存在",
                "item_name": "",
                "file_name": "",
                "file_title": "",
            }

    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}"
    )
    return task_status_info


@app.get(
    "/tasks",
    summary="导入任务列表",
    description="查询所有历史导入任务（持久化存储，重启不丢失）",
)
async def get_all_tasks(limit: int = 100, status: str = None):
    """
    导入任务列表接口
    从 MongoDB 读取所有历史导入任务，按创建时间倒序
    """
    tasks = list_import_tasks(limit=limit, status=status)
    return {"code": 200, "tasks": tasks}


@app.post(
    "/tasks/delete",
    summary="批量删除导入任务",
    description="批量删除历史导入任务，并在安全情况下清理关联知识数据",
)
async def delete_tasks(payload: DeleteImportTasksRequest):
    task_ids = []
    for task_id in payload.task_ids or []:
        cleaned_task_id = (task_id or "").strip()
        if cleaned_task_id and cleaned_task_id not in task_ids:
            task_ids.append(cleaned_task_id)

    if not task_ids:
        raise HTTPException(status_code=400, detail="task_ids is required")

    tasks = get_import_tasks_by_ids(task_ids)
    task_map = {task.get("task_id"): task for task in tasks if task.get("task_id")}
    all_tasks = list_import_tasks(limit=1000)
    remaining_completed_items = {
        (task.get("item_name") or "").strip()
        for task in all_tasks
        if task.get("task_id") not in task_ids
        and (task.get("status") or "").strip() == "completed"
        and (task.get("item_name") or "").strip()
    }

    deleted_task_ids = []
    skipped = []
    cleaned_items = set()
    chunks_collection = os.getenv("CHUNKS_COLLECTION") or ""
    item_name_collection = os.getenv("ITEM_NAME_COLLECTION") or ""

    for task_id in task_ids:
        task = task_map.get(task_id)
        if not task:
            skipped.append({"task_id": task_id, "reason": "导入记录不存在或已删除"})
            continue

        task_status = (task.get("status") or "").strip()
        if task_status in {"pending", "uploading", "processing"}:
            skipped.append(
                {"task_id": task_id, "reason": "任务仍在处理中，暂不支持删除"}
            )
            continue

        item_name = (task.get("item_name") or "").strip()

        try:
            if (
                item_name
                and item_name not in remaining_completed_items
                and item_name not in cleaned_items
            ):
                _delete_milvus_records_by_item_name(item_name, chunks_collection)
                _delete_milvus_records_by_item_name(item_name, item_name_collection)
                delete_product_graph(item_name)
                cleaned_items.add(item_name)

            _delete_task_local_dir(task)
            clear_task(task_id)

            deleted_count = delete_import_task(task_id)
            if deleted_count == 0:
                skipped.append({"task_id": task_id, "reason": "导入记录不存在或已删除"})
                continue

            deleted_task_ids.append(task_id)
        except Exception as e:
            logger.error(f"[{task_id}] 删除导入任务失败：{e}", exc_info=True)
            skipped.append({"task_id": task_id, "reason": str(e)})

    return {
        "code": 200,
        "requested_count": len(task_ids),
        "deleted_count": len(deleted_task_ids),
        "deleted_task_ids": deleted_task_ids,
        "skipped": skipped,
    }


# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="0.0.0.0",  # 允许所有IP访问（容器/远程环境必需）
        port=8000,  # 服务端口
    )
