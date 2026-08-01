"""只读原图与代理预览文档。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class ImageDocument:
    """保存只读 RGB 原图、元数据和按需生成的代理图。"""

    path: Path
    original_rgb: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    _proxies: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.original_rgb.dtype != np.uint8 or self.original_rgb.ndim != 3:
            raise ValueError("original_rgb 必须是 H×W×3 的 RGB uint8 图像")
        if self.original_rgb.shape[2] != 3:
            raise ValueError("original_rgb 必须有三个 RGB 通道")
        self.original_rgb = np.ascontiguousarray(self.original_rgb)
        self.original_rgb.setflags(write=False)

    @property
    def width(self) -> int:
        """原图宽度。"""

        return int(self.original_rgb.shape[1])

    @property
    def height(self) -> int:
        """原图高度。"""

        return int(self.original_rgb.shape[0])

    @property
    def estimated_working_bytes(self) -> int:
        """估计全分辨率流水线同时持有四张图时的内存占用。"""

        return int(self.original_rgb.nbytes * 4)

    def proxy(self, max_edge: int = 1600) -> np.ndarray:
        """返回最长边不超过给定值的只读 RGB 代理图。"""

        if max_edge <= 0:
            raise ValueError("max_edge 必须大于 0")
        if max(self.width, self.height) <= max_edge:
            return self.original_rgb
        cached = self._proxies.get(max_edge)
        if cached is not None:
            return cached
        scale = max_edge / max(self.width, self.height)
        size = (max(1, round(self.width * scale)), max(1, round(self.height * scale)))
        resized = cv2.resize(self.original_rgb, size, interpolation=cv2.INTER_AREA)
        resized.setflags(write=False)
        self._proxies[max_edge] = resized
        return resized

