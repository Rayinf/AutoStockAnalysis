"""
FeatureStore: 简单的多级缓存（内存为主），用于存储指标/因子序列。
key 维度：symbol|start|end|freq|factor_set|version
注意：为简化演示，此处未实现磁盘持久化；可后续扩展到 parquet。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    value: Dict[str, Any]
    expire_at: float


class FeatureStore:
    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl_seconds

    def _make_key(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str,
        factor_set: str,
        version: str = "v1",
    ) -> str:
        return f"{symbol}|{start}|{end}|{freq}|{factor_set}|{version}"

    def get(
        self, symbol: str, start: str, end: str, freq: str, factor_set: str, version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        key = self._make_key(symbol, start, end, freq, factor_set, version)
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expire_at < time.time():
            # 过期
            self._store.pop(key, None)
            return None
        return entry.value

    def set(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str,
        factor_set: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        version: str = "v1",
    ) -> None:
        key = self._make_key(symbol, start, end, freq, factor_set, version)
        expire = time.time() + (ttl_seconds or self._default_ttl)
        self._store[key] = CacheEntry(value=value, expire_at=expire)


# 单例
feature_store = FeatureStore()



