import contextlib
import contextvars
import copy
import hashlib
import json
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from app.conf.cache_config import (
    DEFAULT_CACHE_NAMESPACES,
    namespace_ttl,
    query_cache_config,
)
from app.core.logger import logger

try:
    from redis import Redis

    _REDIS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import path depends on runtime env
    Redis = None
    _REDIS_IMPORT_ERROR = exc


_CACHE_POLICY_VAR: contextvars.ContextVar["RequestCachePolicy | None"] = (
    contextvars.ContextVar("query_cache_policy", default=None)
)
_REQUEST_CACHE_VAR: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("query_cache_request_local", default=None)
)
_REQUEST_STATS_VAR: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("query_cache_request_stats", default=None)
)


def _normalize_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_payload(payload)
            for key, payload in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, set):
        return [_normalize_payload(item) for item in sorted(value, key=lambda item: str(item))]
    return str(value)


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class RequestCachePolicy:
    enabled: bool
    namespaces: frozenset[str]

    def allows(self, namespace: str) -> bool:
        return self.enabled and namespace in self.namespaces


class InMemoryTTLCache:
    def __init__(self, max_entries: int):
        self.max_entries = max(16, int(max_entries))
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def _evict_expired(self, now_ts: Optional[float] = None) -> None:
        current = now_ts if now_ts is not None else _now()
        expired_keys = [key for key, (expires_at, _value) in self._store.items() if expires_at <= current]
        for key in expired_keys:
            self._store.pop(key, None)

    def get(self, key: str) -> Any:
        current = _now()
        with self._lock:
            self._evict_expired(current)
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= current:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return _deepcopy(value)

    def set(self, key: str, value: Any, ttl_sec: int) -> None:
        if ttl_sec <= 0:
            return
        current = _now()
        expires_at = current + ttl_sec
        with self._lock:
            self._evict_expired(current)
            self._store[key] = (expires_at, _deepcopy(value))
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        current = _now()
        with self._lock:
            self._evict_expired(current)
            return len(self._store)


class QueryCacheManager:
    def __init__(self) -> None:
        self._l1 = InMemoryTTLCache(query_cache_config.l1_max_entries)
        self._redis_client: Redis | None = None
        self._redis_init_attempted = False
        self._redis_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._global_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._epoch_key = f"{query_cache_config.namespace_prefix}:epoch"
        self._epoch = 1
        self._epoch_lock = threading.Lock()
        self._load_epoch()

    def _redis_enabled(self) -> bool:
        return bool(
            query_cache_config.enabled
            and query_cache_config.l2_enabled
            and query_cache_config.redis_url
            and Redis is not None
        )

    def _get_redis_client(self) -> Redis | None:
        if not self._redis_enabled():
            return None
        if self._redis_client is not None:
            return self._redis_client
        with self._redis_lock:
            if self._redis_client is not None:
                return self._redis_client
            if self._redis_init_attempted:
                return None
            self._redis_init_attempted = True
            try:
                client = Redis.from_url(query_cache_config.redis_url, decode_responses=False)
                client.ping()
                self._redis_client = client
                logger.info("查询缓存：Redis 后端连接成功")
                return self._redis_client
            except Exception as exc:
                logger.warning(f"查询缓存：Redis 不可用，已退化为 L0/L1 本地缓存 ({exc})")
                return None

    def _load_epoch(self) -> None:
        client = self._get_redis_client()
        if client is None:
            self._epoch = 1
            return
        try:
            raw = client.get(self._epoch_key)
            if raw is None:
                client.set(self._epoch_key, b"1")
                self._epoch = 1
                return
            self._epoch = max(1, int(raw))
        except Exception as exc:
            logger.warning(f"查询缓存：读取 epoch 失败，回退到本地 epoch=1 ({exc})")
            self._epoch = 1

    def _record_global(self, namespace: str, field: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._global_stats[namespace][field] += amount

    def _build_cache_key(self, namespace: str, descriptor: Dict[str, Any]) -> str:
        payload = {
            "epoch": self._epoch,
            "namespace": namespace,
            "descriptor": _normalize_payload(descriptor),
        }
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        return f"{query_cache_config.namespace_prefix}:{namespace}:{digest}"

    def _clear_redis_cache_keys(self, client: Redis) -> int:
        deleted_count = 0
        for namespace in sorted(DEFAULT_CACHE_NAMESPACES):
            pattern = f"{query_cache_config.namespace_prefix}:{namespace}:*"
            batch: list[Any] = []
            for key in client.scan_iter(match=pattern, count=500):
                batch.append(key)
                if len(batch) >= 500:
                    deleted_count += int(client.delete(*batch) or 0)
                    batch.clear()
            if batch:
                deleted_count += int(client.delete(*batch) or 0)
        return deleted_count

    def invalidate_all(self, reason: str = "manual") -> Dict[str, Any]:
        deleted_count = 0
        redis_error = ""
        with self._epoch_lock:
            client = self._get_redis_client()
            if client is not None:
                try:
                    deleted_count = self._clear_redis_cache_keys(client)
                except Exception as exc:
                    redis_error = str(exc)
                    logger.warning(f"查询缓存：Redis 清空失败，仅清空本地缓存 ({exc})")
            self._l1.clear()
        with self._stats_lock:
            self._global_stats.clear()
        return {
            "ok": True,
            "reason": reason,
            "epoch": self._epoch,
            "deleted_keys": deleted_count,
            "redis_error": redis_error,
            "message": "查询缓存已清空",
        }

    def summary(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = {
                namespace: dict(sorted(values.items()))
                for namespace, values in sorted(self._global_stats.items())
            }
        total = defaultdict(int)
        for namespace_stats in stats.values():
            for key, value in namespace_stats.items():
                total[key] += int(value)
        lookups = int(total.get("lookups", 0))
        hits = int(total.get("hits", 0))
        return {
            "enabled": bool(query_cache_config.enabled),
            "redis_enabled": self._get_redis_client() is not None,
            "epoch": self._epoch,
            "l1_size": self._l1.size(),
            "default_namespaces": sorted(DEFAULT_CACHE_NAMESPACES),
            "overall": {
                **dict(sorted(total.items())),
                "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
            },
            "namespaces": stats,
            "redis_url": query_cache_config.redis_url if query_cache_config.redis_url else "",
            "redis_import_error": str(_REDIS_IMPORT_ERROR) if _REDIS_IMPORT_ERROR else "",
        }

    def get(self, namespace: str, descriptor: Dict[str, Any]) -> Any:
        policy = _CACHE_POLICY_VAR.get()
        if policy is None or not policy.allows(namespace):
            return None

        key = self._build_cache_key(namespace, descriptor)
        request_local = _REQUEST_CACHE_VAR.get()
        stats = _REQUEST_STATS_VAR.get()
        if stats is not None:
            _record_request_stat(stats, namespace, "lookups")
        self._record_global(namespace, "lookups")

        if query_cache_config.l0_enabled and request_local is not None and key in request_local:
            value = _deepcopy(request_local[key])
            if stats is not None:
                _record_request_stat(stats, namespace, "hits")
                _record_request_stat(stats, namespace, "l0_hits")
            self._record_global(namespace, "hits")
            self._record_global(namespace, "l0_hits")
            return value

        if query_cache_config.l1_enabled:
            value = self._l1.get(key)
            if value is not None:
                if request_local is not None and query_cache_config.l0_enabled:
                    request_local[key] = _deepcopy(value)
                if stats is not None:
                    _record_request_stat(stats, namespace, "hits")
                    _record_request_stat(stats, namespace, "l1_hits")
                self._record_global(namespace, "hits")
                self._record_global(namespace, "l1_hits")
                return value

        client = self._get_redis_client()
        if client is not None:
            try:
                raw = client.get(key)
                if raw is not None:
                    value = pickle.loads(raw)
                    if query_cache_config.l1_enabled:
                        self._l1.set(key, value, namespace_ttl(namespace))
                    if request_local is not None and query_cache_config.l0_enabled:
                        request_local[key] = _deepcopy(value)
                    if stats is not None:
                        _record_request_stat(stats, namespace, "hits")
                        _record_request_stat(stats, namespace, "l2_hits")
                    self._record_global(namespace, "hits")
                    self._record_global(namespace, "l2_hits")
                    return _deepcopy(value)
            except Exception as exc:
                logger.warning(f"查询缓存：读取 Redis 缓存失败(namespace={namespace}) ({exc})")

        if stats is not None:
            _record_request_stat(stats, namespace, "misses")
        self._record_global(namespace, "misses")
        return None

    def set(self, namespace: str, descriptor: Dict[str, Any], value: Any) -> None:
        policy = _CACHE_POLICY_VAR.get()
        if policy is None or not policy.allows(namespace):
            return
        ttl_sec = namespace_ttl(namespace)
        key = self._build_cache_key(namespace, descriptor)
        request_local = _REQUEST_CACHE_VAR.get()
        if request_local is not None and query_cache_config.l0_enabled:
            request_local[key] = _deepcopy(value)
        if query_cache_config.l1_enabled:
            self._l1.set(key, value, ttl_sec)
        client = self._get_redis_client()
        if client is not None:
            try:
                client.setex(key, ttl_sec, pickle.dumps(value))
            except Exception as exc:
                logger.warning(f"查询缓存：写入 Redis 缓存失败(namespace={namespace}) ({exc})")
        stats = _REQUEST_STATS_VAR.get()
        if stats is not None:
            _record_request_stat(stats, namespace, "writes")
        self._record_global(namespace, "writes")


def _record_request_stat(stats: Dict[str, Any], namespace: str, field: str, amount: int = 1) -> None:
    namespace_stats = stats.setdefault(namespace, defaultdict(int))
    namespace_stats[field] += amount


def _policy_from_state(state: Dict[str, Any] | None) -> RequestCachePolicy:
    enabled = bool(query_cache_config.enabled)
    namespaces = set(DEFAULT_CACHE_NAMESPACES)
    if not isinstance(state, dict):
        return RequestCachePolicy(enabled=enabled, namespaces=frozenset(namespaces))

    for container in (
        state,
        state.get("route_overrides") or {},
        state.get("evaluation_overrides") or {},
    ):
        if not isinstance(container, dict):
            continue
        if "cache_enabled" in container:
            enabled = bool(container.get("cache_enabled"))
        control = container.get("cache_control") or {}
        if not isinstance(control, dict):
            continue
        if "enabled" in control:
            enabled = bool(control.get("enabled"))
        requested_namespaces = control.get("namespaces")
        if isinstance(requested_namespaces, (list, tuple, set)):
            cleaned = {
                str(name).strip()
                for name in requested_namespaces
                if str(name).strip() in DEFAULT_CACHE_NAMESPACES
            }
            if cleaned:
                namespaces = cleaned
        disabled_namespaces = control.get("disabled_namespaces")
        if isinstance(disabled_namespaces, (list, tuple, set)):
            namespaces -= {
                str(name).strip()
                for name in disabled_namespaces
                if str(name).strip() in DEFAULT_CACHE_NAMESPACES
            }
    return RequestCachePolicy(enabled=enabled, namespaces=frozenset(sorted(namespaces)))


@contextlib.contextmanager
def query_cache_request_context(state: Dict[str, Any] | None = None):
    policy = _policy_from_state(state)
    token_policy = _CACHE_POLICY_VAR.set(policy)
    token_local = _REQUEST_CACHE_VAR.set({})
    token_stats = _REQUEST_STATS_VAR.set({})
    try:
        yield policy
    finally:
        _REQUEST_STATS_VAR.reset(token_stats)
        _REQUEST_CACHE_VAR.reset(token_local)
        _CACHE_POLICY_VAR.reset(token_policy)


def get_current_request_cache_summary() -> Dict[str, Any]:
    policy = _CACHE_POLICY_VAR.get()
    stats = _REQUEST_STATS_VAR.get()
    if policy is None:
        return {"enabled": False, "namespaces": [], "overall": {}, "namespaces_breakdown": {}}

    breakdown: Dict[str, Dict[str, Any]] = {}
    totals = defaultdict(int)
    for namespace in sorted(policy.namespaces):
        namespace_stats = dict(sorted((stats or {}).get(namespace, {}).items()))
        lookups = int(namespace_stats.get("lookups", 0))
        hits = int(namespace_stats.get("hits", 0))
        namespace_summary = {
            **namespace_stats,
            "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
        }
        breakdown[namespace] = namespace_summary
        for key, value in namespace_stats.items():
            totals[key] += int(value)

    lookups = int(totals.get("lookups", 0))
    hits = int(totals.get("hits", 0))
    l0_hits = int(totals.get("l0_hits", 0))
    l1_hits = int(totals.get("l1_hits", 0))
    l2_hits = int(totals.get("l2_hits", 0))
    return {
        "enabled": policy.enabled,
        "namespaces": sorted(policy.namespaces),
        "overall": {
            **dict(sorted(totals.items())),
            "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
            "l0_hit_rate": round(l0_hits / lookups, 4) if lookups else 0.0,
            "l1_hit_rate": round(l1_hits / lookups, 4) if lookups else 0.0,
            "l2_hit_rate": round(l2_hits / lookups, 4) if lookups else 0.0,
        },
        "namespaces_breakdown": breakdown,
        "hit_namespaces": [
            namespace
            for namespace, namespace_stats in breakdown.items()
            if int(namespace_stats.get("hits", 0)) > 0
        ],
    }


_QUERY_CACHE_MANAGER = QueryCacheManager()


def get_query_cache_manager() -> QueryCacheManager:
    return _QUERY_CACHE_MANAGER


def query_cache_get(namespace: str, descriptor: Dict[str, Any]) -> Any:
    return _QUERY_CACHE_MANAGER.get(namespace, descriptor)


def query_cache_set(namespace: str, descriptor: Dict[str, Any], value: Any) -> None:
    _QUERY_CACHE_MANAGER.set(namespace, descriptor, value)


def reset_query_cache(reason: str = "manual") -> Dict[str, Any]:
    return _QUERY_CACHE_MANAGER.invalidate_all(reason=reason)


def get_query_cache_stats() -> Dict[str, Any]:
    return _QUERY_CACHE_MANAGER.summary()


def normalize_cache_sequence(values: Sequence[Any] | None) -> list[Any]:
    if not values:
        return []
    return [_normalize_payload(value) for value in values]
