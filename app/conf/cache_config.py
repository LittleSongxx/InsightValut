from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(value, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class QueryCacheConfig:
    enabled: bool
    namespace_prefix: str
    redis_url: str
    l0_enabled: bool
    l1_enabled: bool
    l2_enabled: bool
    l1_max_entries: int
    embedding_ttl_sec: int
    retrieval_ttl_sec: int
    graph_ttl_sec: int
    hyde_ttl_sec: int
    web_ttl_sec: int
    rerank_ttl_sec: int
    answer_ttl_sec: int


DEFAULT_CACHE_NAMESPACES = {
    "embedding",
    "retrieval_embedding",
    "retrieval_bm25",
    "retrieval_kg",
    "hyde_doc",
    "web_search",
    "rerank",
    "answer",
}


def namespace_ttl(namespace: str) -> int:
    mapping = {
        "embedding": query_cache_config.embedding_ttl_sec,
        "retrieval_embedding": query_cache_config.retrieval_ttl_sec,
        "retrieval_bm25": query_cache_config.retrieval_ttl_sec,
        "retrieval_kg": query_cache_config.graph_ttl_sec,
        "hyde_doc": query_cache_config.hyde_ttl_sec,
        "web_search": query_cache_config.web_ttl_sec,
        "rerank": query_cache_config.rerank_ttl_sec,
        "answer": query_cache_config.answer_ttl_sec,
    }
    return max(1, int(mapping.get(namespace, query_cache_config.retrieval_ttl_sec)))


query_cache_config = QueryCacheConfig(
    enabled=_bool(os.getenv("QUERY_CACHE_ENABLED"), True),
    namespace_prefix=(os.getenv("QUERY_CACHE_NAMESPACE_PREFIX") or "insightvault:qcache").strip()
    or "insightvault:qcache",
    redis_url=(os.getenv("QUERY_CACHE_REDIS_URL") or "").strip(),
    l0_enabled=_bool(os.getenv("QUERY_CACHE_L0_ENABLED"), True),
    l1_enabled=_bool(os.getenv("QUERY_CACHE_L1_ENABLED"), True),
    l2_enabled=_bool(os.getenv("QUERY_CACHE_L2_ENABLED"), True),
    l1_max_entries=max(64, _int(os.getenv("QUERY_CACHE_L1_MAX_ENTRIES"), 512)),
    embedding_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_EMBEDDING_TTL_SEC"), 86400)),
    retrieval_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_RETRIEVAL_TTL_SEC"), 21600)),
    graph_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_GRAPH_TTL_SEC"), 21600)),
    hyde_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_HYDE_TTL_SEC"), 86400)),
    web_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_WEB_TTL_SEC"), 1800)),
    rerank_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_RERANK_TTL_SEC"), 21600)),
    answer_ttl_sec=max(60, _int(os.getenv("QUERY_CACHE_ANSWER_TTL_SEC"), 3600)),
)
