"""有界的流水线节点图像缓存。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True, slots=True)
class CacheKey:
    """节点缓存键。"""

    source_id: str
    node_index: int
    signature: str


class PipelineCache:
    """按总字节数限制的 LRU 缓存，避免历史预览无限增长。"""

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[CacheKey, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    @property
    def size_bytes(self) -> int:
        """当前缓存占用的图像字节数。"""

        return self._bytes

    def get(self, key: CacheKey) -> np.ndarray | None:
        """获取缓存并更新 LRU 顺序。"""

        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: CacheKey, image: np.ndarray) -> None:
        """存入只读图像，并淘汰最老节点。"""

        cached = np.ascontiguousarray(image)
        cached.setflags(write=False)
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= previous.nbytes
            self._items[key] = cached
            self._bytes += cached.nbytes
            while self._bytes > self.max_bytes and len(self._items) > 1:
                _, removed = self._items.popitem(last=False)
                self._bytes -= removed.nbytes

    def invalidate_from(self, node_index: int) -> None:
        """淘汰指定节点及其所有下游缓存。"""

        with self._lock:
            for key in [item for item in self._items if item.node_index >= node_index]:
                self._bytes -= self._items.pop(key).nbytes

    def clear(self) -> None:
        """清空缓存。"""

        with self._lock:
            self._items.clear()
            self._bytes = 0
