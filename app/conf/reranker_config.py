# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_path: str   # 本地模型路径（优先使用）
    bge_reranker_large: str  # HuggingFace repo ID（本地不存在时下载）
    bge_reranker_device: str  # 运行设备
    bge_reranker_fp16: bool   # 是否开启半精度

reranker_config = RerankerConfig(
    bge_reranker_path=os.getenv("BGE_RERANKER_PATH"),
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1)
)