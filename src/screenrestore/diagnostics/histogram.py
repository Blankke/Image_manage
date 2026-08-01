"""RGB 直方图计算与轻量渲染。"""

from __future__ import annotations

import cv2
import numpy as np


def rgb_histogram(image_rgb: np.ndarray) -> np.ndarray:
    """返回形状为 3×256 的归一化 RGB 计数。"""

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("直方图需要 H×W×3 RGB uint8 图像")
    histograms = []
    for channel in range(3):
        counts = np.bincount(image_rgb[..., channel].ravel(), minlength=256).astype(np.float64)
        counts /= max(1.0, float(counts.max()))
        histograms.append(counts)
    return np.stack(histograms)


def render_histogram(image_rgb: np.ndarray, width: int = 640, height: int = 280) -> np.ndarray:
    """渲染带网格的 RGB 直方图为 RGB uint8 图。"""

    histograms = rgb_histogram(image_rgb)
    canvas = np.full((height, width, 3), 24, np.uint8)
    for fraction in (0.25, 0.5, 0.75):
        y = round((height - 1) * fraction)
        cv2.line(canvas, (0, y), (width - 1, y), (55, 55, 55), 1)
    colors = ((255, 70, 70), (70, 235, 90), (80, 130, 255))
    x_values = np.linspace(0, width - 1, 256)
    for values, color in zip(histograms, colors, strict=True):
        y_values = (height - 1) - values * (height - 10)
        points = np.column_stack((x_values, y_values)).astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
    return canvas

