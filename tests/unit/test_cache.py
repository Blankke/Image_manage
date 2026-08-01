"""流水线缓存测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.core.cache import CacheKey, PipelineCache


def test_cache_invalidation_only_removes_downstream_nodes() -> None:
    cache = PipelineCache(max_bytes=10_000)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    keys = [CacheKey("source", index, str(index)) for index in range(3)]
    for key in keys:
        cache.put(key, image)

    cache.invalidate_from(1)

    assert cache.get(keys[0]) is not None
    assert cache.get(keys[1]) is None
    assert cache.get(keys[2]) is None


def test_cache_is_bounded_and_returns_read_only_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    cache = PipelineCache(max_bytes=image.nbytes + 10)
    cache.put(CacheKey("source", 0, "a"), image)
    cache.put(CacheKey("source", 1, "b"), image)
    cached = cache.get(CacheKey("source", 1, "b"))
    assert cached is not None
    assert not cached.flags.writeable
    assert cache.size_bytes <= image.nbytes + 10

