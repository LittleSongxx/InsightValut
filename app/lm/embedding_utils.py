import os
import threading
from pathlib import Path

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None
_bge_m3_lock = threading.Lock()


def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    with _bge_m3_lock:
        if _bge_m3_ef is not None:
            return _bge_m3_ef

        # BGE_M3_PATH: 本地模型路径（相对/绝对），存在则优先使用；不存在则用 repo ID 自动下载
        raw_path = embedding_config.bge_m3_path
        if raw_path:
            resolved = Path(os.path.expanduser(raw_path))
            # 相对路径 → 拼接 PROJECT_ROOT（容器内为 /app）
            if not resolved.is_absolute():
                project_root = os.getenv("PROJECT_ROOT", "/app")
                resolved = Path(project_root) / resolved
            if resolved.exists():
                model_name = str(resolved)
                logger.info(f"检测到本地模型目录，使用本地路径：{resolved}")
            else:
                model_name = embedding_config.bge_m3 or "BAAI/bge-m3"
                logger.warning(f"本地模型目录不存在（{resolved}），回退到 HuggingFace repo ID：{model_name}")
        else:
            model_name = embedding_config.bge_m3 or "BAAI/bge-m3"

        device = embedding_config.bge_device or "cpu"
        use_fp16 = embedding_config.bge_fp16 or False

        logger.info(
            "开始初始化BGE-M3模型",
            extra={
                "model_name": model_name,
                "device": device,
                "use_fp16": use_fp16,
                "normalize_embeddings": True,
            },
        )

        try:
            _bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                normalize_embeddings=True,
            )
            logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
            return _bge_m3_ef
        except Exception as e:
            logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
            raise


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = (
                embeddings["sparse"]
                .indices[
                    embeddings["sparse"].indptr[i] : embeddings["sparse"].indptr[i + 1]
                ]
                .tolist()
            )
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = (
                embeddings["sparse"]
                .data[
                    embeddings["sparse"].indptr[i] : embeddings["sparse"].indptr[i + 1]
                ]
                .tolist()
            )
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            "dense": [
                emb.tolist() for emb in embeddings["dense"]
            ],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse,  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""
