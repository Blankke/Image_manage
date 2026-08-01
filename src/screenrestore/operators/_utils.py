"""算子共用的 RGB/浮点转换与参数检查。"""

from __future__ import annotations

import numpy as np


def require_rgb_u8(image: np.ndarray) -> None:
    """验证模块边界的统一 RGB uint8 契约。"""

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("算子需要 H×W×3 的 RGB uint8 图像")


def to_float(image: np.ndarray) -> np.ndarray:
    """把 RGB uint8 转为 [0, 1] float32。"""

    require_rgb_u8(image)
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    """裁剪浮点数组并转为 RGB uint8。"""

    return np.clip(np.rint(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0) * 255.0), 0, 255).astype(
        np.uint8
    )


def require_range(name: str, value: float, minimum: float, maximum: float) -> None:
    """验证有限标量范围。"""

    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须位于 {minimum}..{maximum}")

