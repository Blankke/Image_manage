"""像素来源标签与多帧观测融合适配。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PixelOrigin(IntEnum):
    """归档输出中每个像素的来源。"""

    OBSERVED = 0
    RECOVERED_OBSERVATION = 1
    ESTIMATED = 2
    GENERATED = 3
    UNRESOLVED = 4


@dataclass(slots=True)
class ProvenanceMap:
    """与工作图同尺寸的紧凑 uint8 来源图。"""

    labels: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.labels)
        if values.ndim != 2:
            raise ValueError("provenance map 必须是二维数组")
        if values.dtype != np.uint8:
            values = values.astype(np.uint8)
        valid_codes = {int(item) for item in PixelOrigin}
        if any(int(value) not in valid_codes for value in np.unique(values)):
            raise ValueError("provenance map 包含未知来源代码")
        self.labels = values.copy()

    @classmethod
    def observed(cls, image_shape: tuple[int, ...]) -> ProvenanceMap:
        """创建全部来自单帧真实观测的来源图。"""

        return cls(np.full(image_shape[:2], int(PixelOrigin.OBSERVED), dtype=np.uint8))

    @classmethod
    def from_fusion_masks(
        cls,
        image_shape: tuple[int, ...],
        recovered_observation_mask: np.ndarray,
        unresolved_mask: np.ndarray,
    ) -> ProvenanceMap:
        """将多帧融合的真实补回与未解决掩码映射为统一标签。"""

        result = cls.observed(image_shape)
        result.mark(recovered_observation_mask, PixelOrigin.RECOVERED_OBSERVATION)
        # unresolved 优先级最高，覆盖此前任何来源标签。
        result.mark(unresolved_mask, PixelOrigin.UNRESOLVED)
        return result

    def mark(self, mask: np.ndarray, origin: PixelOrigin) -> None:
        """在副本持有的标签图上标记来源；mask 不会被修改。"""

        values = np.asarray(mask)
        if values.shape != self.labels.shape:
            raise ValueError("来源掩码尺寸必须与 provenance map 一致")
        self.labels[values.astype(bool)] = int(origin)

    def summary(self) -> dict[str, float]:
        """返回每类来源占比，适合进入 JSON 诊断。"""

        total = max(1, self.labels.size)
        return {
            origin.name.lower(): round(float(np.count_nonzero(self.labels == int(origin)) / total), 8)
            for origin in PixelOrigin
        }

    def copy(self) -> ProvenanceMap:
        return ProvenanceMap(self.labels.copy())
