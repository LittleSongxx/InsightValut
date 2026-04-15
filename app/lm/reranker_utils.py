"""
DashScope 重排序工具 - 直接调用百炼 Rerank API，无需本地下载模型。

用法：将 RERANKER_MODE=dashscope 写入 .env，即可启用。
备选模式 RERANKER_MODE=local：使用本地 FlagReranker 模型（需提前下载）。
"""
import os
import re
import threading
from typing import Any, Dict, List

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
DASHSCOPE_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
).strip()
DASHSCOPE_RERANK_MODEL = (
    os.getenv("DASHSCOPE_RERANK_MODEL") or "gte-rerank-v2"
).strip()
DASHSCOPE_RERANK_BASE_URL = (
    os.getenv("DASHSCOPE_RERANK_BASE_URL")
    or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
).rstrip("/")
LOCAL_RERANK_MODEL_ID = (
    reranker_config.bge_reranker_large or "BAAI/bge-reranker-v2-m3"
).strip()

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
                model_path = LOCAL_RERANK_MODEL_ID
                logger.warning(f"本地 reranker 不存在（{resolved}），使用 repo ID：{model_path}")
        else:
            model_path = LOCAL_RERANK_MODEL_ID

        device = reranker_config.bge_reranker_device or "cpu"
        use_fp16 = reranker_config.bge_reranker_fp16 or False

        _local_reranker = FlagReranker(
            model_name_or_path=model_path,
            device=device,
            use_fp16=use_fp16,
        )
        return _local_reranker


def _tokenize_fallback_text(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)


def _compute_keyword_overlap_scores(sentence_pairs: List[List[str]]) -> List[float]:
    scores: List[float] = []
    for query, doc in sentence_pairs:
        query_tokens = _tokenize_fallback_text(query)
        doc_tokens = _tokenize_fallback_text(doc)
        if not query_tokens or not doc_tokens:
            scores.append(0.0)
            continue

        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        overlap = len(query_set & doc_set)
        coverage = overlap / max(len(query_set), 1)
        density = overlap / max(len(doc_set), 1)
        scores.append(round(coverage * 0.8 + density * 0.2, 6))
    return scores


# ------------------- DashScope API Reranker -------------------


class DashScopeReranker:
    """直接调用 DashScope Rerank API，零模型下载，零显存占用。"""

    def __init__(self, model: str = DASHSCOPE_RERANK_MODEL):
        self.model = model
        self.api_key = DASHSCOPE_API_KEY
        self.base_url = DASHSCOPE_RERANK_BASE_URL
        if not self.api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY / OPENAI_API_KEY（DashScope API Key）")

    def _build_payload(self, query: str, docs: List[str]) -> Dict[str, Any]:
        top_n = len(docs)
        if self.model == "qwen3-rerank":
            return {
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": top_n,
            }

        return {
            "model": self.model,
            "input": {
                "query": query,
                "documents": docs,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }

    @staticmethod
    def _extract_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        output = result.get("output")
        if isinstance(output, dict) and isinstance(output.get("results"), list):
            return output.get("results") or []
        if isinstance(result.get("results"), list):
            return result.get("results") or []
        return []

    def _compute_remote_scores(self, sentence_pairs: List[List[str]]) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        docs = [pair[1] for pair in sentence_pairs]
        query = sentence_pairs[0][0] if sentence_pairs else ""
        payload = self._build_payload(query=query, docs=docs)

        logger.debug(
            f"DashScope Rerank 请求，模型={self.model}，文档数={len(docs)}，地址={self.base_url}"
        )
        resp = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            logger.error(
                f"DashScope Rerank 请求失败: status={resp.status_code}, body={resp.text[:500]}"
            )
        resp.raise_for_status()
        result = resp.json()

        results = self._extract_results(result)
        out = [0.0] * len(docs)
        for item in results:
            idx = item.get("index", item.get("document_index", -1))
            score = item.get("relevance_score", item.get("score", 0.0))
            if 0 <= idx < len(docs):
                out[idx] = score
        return out

    @staticmethod
    def _compute_local_scores(sentence_pairs: List[List[str]]) -> List[float]:
        local_reranker = _get_local_reranker()
        return [float(score) for score in local_reranker.compute_score(sentence_pairs)]

    def compute_score(self, sentence_pairs: List[List[str]]) -> List[float]:
        """
        批量计算相关性得分。

        :param sentence_pairs: [[query, doc1], [query, doc2], ...]
        :return: [score1, score2, ...]
        """
        if not sentence_pairs:
            return []

        try:
            return self._compute_remote_scores(sentence_pairs)
        except Exception:
            logger.exception("DashScope Rerank 不可用，将尝试本地模型回退")

        try:
            scores = self._compute_local_scores(sentence_pairs)
            logger.warning("DashScope Rerank 已回退到本地 FlagReranker")
            return scores
        except Exception:
            logger.exception("本地 Rerank 模型不可用，将回退到启发式重排")

        logger.warning("Rerank 最终降级为关键词重叠启发式排序")
        return _compute_keyword_overlap_scores(sentence_pairs)


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
