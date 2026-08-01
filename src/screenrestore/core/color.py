"""标准 sRGB 与线性 RGB 转换。

核心流水线的数组始终是 ``float32 [0,1]``；这两个函数只改变传递函数，不改变
RGB 原色、白点或通道顺序。
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(image_srgb: np.ndarray) -> np.ndarray:
    """按 IEC 61966-2-1 分段 EOTF 把 sRGB 转为线性光。"""

    source = _validated_float_rgb(image_srgb, "sRGB")
    linear = np.where(
        source <= 0.04045,
        source / 12.92,
        np.power((source + 0.055) / 1.055, 2.4),
    )
    return np.ascontiguousarray(linear.astype(np.float32, copy=False))


def linear_to_srgb(image_linear: np.ndarray) -> np.ndarray:
    """按 IEC 61966-2-1 分段 OETF 把线性 RGB 编码为 sRGB。"""

    source = _validated_float_rgb(image_linear, "线性 RGB")
    srgb = np.where(
        source <= 0.0031308,
        source * 12.92,
        1.055 * np.power(source, 1.0 / 2.4) - 0.055,
    )
    return np.ascontiguousarray(np.clip(srgb, 0.0, 1.0).astype(np.float32, copy=False))


def _validated_float_rgb(image: np.ndarray, label: str) -> np.ndarray:
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} 必须是 H×W×3 RGB float32 图像")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{label} 包含非有限值")
    return np.clip(image, 0.0, 1.0)
