import hashlib
import json
from typing import Any

from django.core.cache import cache


def build_cache_key(prefix: str, payload: dict[str, Any] | None = None) -> str:
    if not payload:
        return prefix
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.md5(serialized.encode('utf-8')).hexdigest()
    return f"{prefix}:{digest}"


def get_or_set_json(prefix: str, payload: dict[str, Any], producer, timeout: int = 300):
    cache_key = build_cache_key(prefix, payload)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    value = producer()
    cache.set(cache_key, value, timeout=timeout)
    return value


def invalidate_prefix(prefix: str, candidates: list[dict[str, Any]] | None = None):
    keys = [prefix]
    for payload in candidates or []:
        keys.append(build_cache_key(prefix, payload))
    cache.delete_many(keys)
