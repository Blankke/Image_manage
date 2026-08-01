"""基于梯度阈值与 DCT Poisson 求解的单图反光抑制。

本模块移植并修改自 Jan Palasek 的 ``fast-reflection-removal``：
https://github.com/JanPalasek/fast-reflection-removal
原文件：``src/python/frr/base.py``、``src/python/frr/core.py``，MIT License，
Copyright (c) 2024 Jan Palasek。许可证与修改说明见 THIRD_PARTY_NOTICES.md。

主要修改：移除 Matplotlib/文件写入；合并为纯 NumPy/SciPy 函数；修复上游
``remove_reflection`` 重复计算；增加逐通道稳健分位数配准、有限强度混合、
取消检查和 float32/除零保护。算法仍会压制低对比真实纹理，故只能作为实验模式。
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct

from screenrestore.core.cancellation import CancellationToken


def suppress_reflection_dct(
    image_rgb_float: np.ndarray,
    *,
    gradient_threshold: float,
    smoothness_lambda: float,
    curvature_weight: float,
    strength: float,
    cancellation: CancellationToken,
) -> np.ndarray:
    """运行梯度稀疏化与 DCT 求解，返回 ``[0,1] float32`` RGB 图像。"""

    source = np.asarray(image_rgb_float, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("DCT 反光抑制需要 H×W×3 RGB 图像")
    if not np.all(np.isfinite(source)) or float(source.min()) < 0 or float(source.max()) > 1:
        raise ValueError("DCT 反光抑制输入必须是 [0,1] 有限浮点数")
    height, width = source.shape[:2]
    row_cosine = np.cos((np.pi * np.arange(height, dtype=np.float64)) / height)
    column_cosine = np.cos((np.pi * np.arange(width, dtype=np.float64)) / width)
    kappa = 2.0 * (np.add.outer(row_cosine, column_cosine) - 2.0)
    epsilon = 1e-8
    denominator = (
        curvature_weight * np.square(kappa)
        - smoothness_lambda * kappa
        + epsilon
    )
    denominator = np.maximum(denominator, epsilon)
    restored_channels: list[np.ndarray] = []
    for channel_index in range(3):
        cancellation.check()
        channel = source[..., channel_index]
        first_laplacian = _laplacian(channel, gradient_threshold)
        right_hand_side = _laplacian(first_laplacian) + epsilon * channel
        solution = _idct2(_dct2(right_hand_side) / denominator).astype(np.float32)
        restored_channels.append(_match_robust_range(solution, channel))
    restored = np.stack(restored_channels, axis=2)
    return np.clip(source * (1.0 - strength) + restored * strength, 0.0, 1.0).astype(
        np.float32
    )


def _gradient(values: np.ndarray) -> np.ndarray:
    """复制上游前向差分边界约定，返回横向、纵向梯度。"""

    rows, columns = values.shape
    horizontal = np.zeros_like(values)
    horizontal[:, : columns - 1] = np.diff(values, axis=1)
    vertical = np.zeros_like(values)
    vertical[: rows - 1, :] = np.diff(values, axis=0)
    return np.stack((horizontal, vertical), axis=2)


def _divergence(gradients: np.ndarray) -> np.ndarray:
    """计算与前向差分配对的离散散度。"""

    rows, columns = gradients.shape[:2]
    horizontal = gradients[..., 0]
    horizontal_previous = np.zeros((rows, columns), dtype=gradients.dtype)
    horizontal_previous[:, 1:] = horizontal[:, :-1]
    vertical = gradients[..., 1]
    vertical_previous = np.zeros((rows, columns), dtype=gradients.dtype)
    vertical_previous[1:, :] = vertical[:-1, :]
    return horizontal - horizontal_previous + vertical - vertical_previous


def _laplacian(values: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """计算离散 Laplacian；首层可删除低于阈值的梯度。"""

    gradients = _gradient(values)
    if threshold is not None:
        magnitude = np.linalg.norm(gradients, axis=2)
        gradients[magnitude < threshold] = 0
    return _divergence(gradients)


def _dct2(values: np.ndarray) -> np.ndarray:
    """保持上游实现的二维正交 DCT-II。"""

    return dct(dct(values.T, norm="ortho").T, norm="ortho")


def _idct2(values: np.ndarray) -> np.ndarray:
    """保持上游实现的二维正交逆 DCT。"""

    return idct(idct(values.T, norm="ortho").T, norm="ortho")


def _match_robust_range(solution: np.ndarray, source: np.ndarray) -> np.ndarray:
    """把求解结果的稳健动态范围配准到输入，避免孤立极值支配全图。"""

    output_low, output_high = np.percentile(solution, (0.5, 99.5))
    source_low, source_high = np.percentile(source, (0.5, 99.5))
    scale = (source_high - source_low) / max(float(output_high - output_low), 1e-6)
    matched = (solution - output_low) * scale + source_low
    return np.clip(matched, 0.0, 1.0).astype(np.float32)
