"""算子共用的 RGB/浮点转换与参数检查。"""

from __future__ import annotations

import numpy as np


def require_rgb_float(image: np.ndarray) -> None:
    """验证算子之间统一的 RGB float32 [0,1] 契约。"""

    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("算子需要 H×W×3 的 RGB float32 图像")
    if not np.all(np.isfinite(image)):
        raise ValueError("算子输入包含非有限值")
    if image.size and (float(image.min()) < 0.0 or float(image.max()) > 1.0):
        raise ValueError("算子输入必须位于 [0,1]")


def to_float(image: np.ndarray) -> np.ndarray:
    """仅在加载/模型边界把 RGB uint8 转为 [0,1] float32。"""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("转换需要 H×W×3 的 RGB uint8 图像")
    return np.ascontiguousarray(image.astype(np.float32) / 255.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """仅在显示、外部模型或导出边界量化为 RGB uint8。"""

    return np.clip(
        np.rint(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0) * 255.0),
        0,
        255,
    ).astype(
        np.uint8,
    )


def clip_float(image: np.ndarray) -> np.ndarray:
    """清理数值并保持连续 RGB float32 输出。"""

    return np.ascontiguousarray(
        np.clip(
            np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
    )


def require_range(name: str, value: float, minimum: float, maximum: float) -> None:
    """验证有限标量范围。"""

    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须位于 {minimum}..{maximum}")
