"""限制轻量像素网络只能施加小幅残差。"""

from __future__ import annotations

import numpy as np


def apply_bounded_residual(
    corrected_rgb: np.ndarray,
    residual_rgb: np.ndarray,
    alpha: np.ndarray | float,
    *,
    max_delta: float = 0.06,
) -> np.ndarray:
    """计算 ``corrected + alpha * clipped_residual`` 并返回独立数组。"""

    source = np.asarray(corrected_rgb)
    residual = np.asarray(residual_rgb)
    if source.shape != residual.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("校正图与残差必须具有相同的 H×W×3 形状")
    if source.dtype != np.float32 or residual.dtype != np.float32:
        raise ValueError("受限残差需要 float32 输入")
    if not 0.0 < max_delta <= 0.25:
        raise ValueError("max_delta 必须位于 (0, 0.25]")
    alpha_values = np.asarray(alpha, dtype=np.float32)
    if alpha_values.ndim == 2:
        alpha_values = alpha_values[:, :, None]
    try:
        alpha_values = np.broadcast_to(alpha_values, source.shape)
    except ValueError as exc:
        raise ValueError("alpha 无法广播到图像形状") from exc
    bounded = np.clip(residual, -max_delta, max_delta)
    output = source + np.clip(alpha_values, 0.0, 1.0) * bounded
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0).astype(np.float32))
