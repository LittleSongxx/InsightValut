"""
DashScope 重排序工具 - 直接调用百炼 Rerank API，无需本地下载模型。

用法：将 RERANKER_MODE=dashscope 写入 .env，即可启用。
备选模式 RERANKER_MODE=local：使用本地 FlagReranker 模型（需提前下载）。
"""
import os
import threading
from typing import List, Tuple

import requests

from app.core.logger import logger
from app.conf.reranker_config import reranker_config

# 双重加载 dotenv，确保环境变量可用
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

RERANKER_MODE = os.getenv("RERANKER_MODE", "local").strip().lower()
DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ------------------- 本地 FlagReranker -------------------

_local_reranker = None
_local_reranker_lock = threading.Lock()


def _get_local_reranker():
    global _local_reranker
    if _local_reranker is not None:
        return _local_reranker

    with _local_reranker_lock:
        if _local_reranker is not None:
            return _local_reranker

        from FlagEmbedding import FlagReranker
        raw_path = reranker_config.bge_reranker_path
        if raw_path:
            resolved = os.path.expanduser(raw_path)
            if not os.path.isabs(resolved):
                project_root = os.getenv("PROJECT_ROOT", "/app")
                resolved = os.path.join(project_root, resolved)
            if os.path.exists(resolved):
                model_path = resolved
                logger.info(f"使用本地 reranker 模型: {model_path}")
            else:
                model_path = reranker_config.bge_reranker_large or "Alibaba-NLP/gte-rerank-v2"
                logger.warning(f"本地 reranker 不存在（{resolved}），使用 repo ID：{model_path}")
        else:
            model_path = reranker_config.bge_reranker_large or "Alibaba-NLP/gte-rerank-v2"

        device = reranker_config.bge_reranker_device or "cpu"
        use_fp16 = reranker_config.bge_reranker_fp16 or False

        _local_reranker = FlagReranker(
            model_name_or_path=model_path,
            device=device,
            use_fp16=use_fp16,
        )
        return _local_reranker


# ------------------- DashScope API Reranker -------------------


class DashScopeReranker:
    """直接调用 DashScope Rerank API，零模型下载，零显存占用。"""

    def __init__(self, model: str = "gte-rerank"):
        self.model = model
        self.api_key = DASHSCOPE_API_KEY
        if not self.api_key:
            raise ValueError("未设置 OPENAI_API_KEY（DASHSCOPE API Key）")

    def compute_score(self, sentence_pairs: List[List[str]]) -> List[float]:
        """
        批量计算相关性得分。

        :param sentence_pairs: [[query, doc1], [query, doc2], ...]
        :return: [score1, score2, ...]
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        docs = [pair[1] for pair in sentence_pairs]
        query = sentence_pairs[0][0] if sentence_pairs else ""

        payload = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "return_documents": False,
        }

        logger.debug(f"DashScope Rerank 请求，文档数：{len(docs)}")
        resp = requests.post(
            f"{DASHSCOPE_BASE_URL}/rerank",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        results = result.get("results", [])
        # 按 documents 原始顺序返回得分
        out = [0.0] * len(docs)
        for item in results:
            idx = item.get("document_index", -1)
            score = item.get("relevance_score", 0.0)
            if 0 <= idx < len(docs):
                out[idx] = score
        return out


# ------------------- 统一入口 -------------------

_reranker_instance = None


def get_reranker_model():
    """
    根据 RERANKER_MODE 返回重排序模型实例。
    - dashscope: DashScopeReranker（默认，推荐）
    - local: FlagReranker（需本地下载模型）
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    if RERANKER_MODE == "local":
        logger.info("Reranker 模式：本地 FlagReranker")
        _reranker_instance = _get_local_reranker()
    else:
        logger.info("Reranker 模式：DashScope API（无需下载模型）")
        _reranker_instance = DashScopeReranker()

    return _reranker_instance
