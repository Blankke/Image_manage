"""用于算法回归测试的合成图生成器。"""

from __future__ import annotations

import cv2
import numpy as np


def checkerboard(width: int = 320, height: int = 180, cell: int = 16) -> np.ndarray:
    """生成带彩色标记和细线的 RGB 棋盘图。"""

    yy, xx = np.indices((height, width))
    board = (((xx // cell) + (yy // cell)) % 2 * 180 + 35).astype(np.uint8)
    image = np.repeat(board[..., None], 3, axis=2)
    image[..., 1] = np.clip(image[..., 1].astype(np.int16) + xx * 50 // width, 0, 255)
    cv2.line(image, (0, height // 2), (width - 1, height // 2), (255, 0, 0), 1)
    cv2.putText(image, "Screen 123", (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 80), 1)
    return image


def framed_screen(width: int = 640, height: int = 420) -> tuple[np.ndarray, np.ndarray]:
    """生成有清晰人工屏幕边框的 RGB 场景和真实四角。"""

    image = np.full((height, width, 3), 26, dtype=np.uint8)
    corners = np.array([[78, 62], [566, 42], [590, 358], [52, 376]], dtype=np.float32)
    cv2.fillConvexPoly(image, corners.astype(np.int32), (82, 110, 150))
    for inset, color in ((0, (245, 245, 245)), (5, (12, 12, 12))):
        adjusted = corners.copy()
        center = adjusted.mean(axis=0)
        if inset:
            adjusted = center + (adjusted - center) * 0.97
        cv2.polylines(image, [adjusted.astype(np.int32)], True, color, 3)
    return image, corners


def smooth_texture(width: int = 360, height: int = 240, seed: int = 13) -> np.ndarray:
    """生成带缓慢渐变和随机纹理的 RGB 测试图。"""

    generator = np.random.default_rng(seed)
    noise = generator.normal(0, 1, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 3.0)
    texture /= max(1e-6, float(texture.std()))
    yy, xx = np.indices((height, width), dtype=np.float32)
    base = 0.32 + 0.34 * xx / width + 0.12 * yy / height + texture * 0.035
    red = base * 1.04
    green = base
    blue = base * 0.93
    return np.clip(np.stack((red, green, blue), axis=2) * 255, 0, 255).astype(np.uint8)


def add_banding(
    image_rgb: np.ndarray,
    direction: str = "horizontal",
    amplitude: float = 0.16,
    period: float = 15.0,
) -> np.ndarray:
    """叠加正弦条带、逐行/列曝光偏差和缓慢渐变。"""

    height, width = image_rgb.shape[:2]
    length = height if direction == "horizontal" else width
    axis = np.arange(length, dtype=np.float32)
    periodic = amplitude * np.sin(2 * np.pi * axis / period)
    exposure_bias = 0.025 * np.sin(2 * np.pi * axis / (period * 2.7) + 0.7)
    slow_gradient = 0.05 * (axis / max(1, length - 1) - 0.5)
    gain = 1.0 + periodic + exposure_bias + slow_gradient
    field = gain[:, None, None] if direction == "horizontal" else gain[None, :, None]
    return np.clip(image_rgb.astype(np.float32) * field, 0, 255).astype(np.uint8)


def add_color_moire(image_rgb: np.ndarray, amplitude: float = 34.0) -> np.ndarray:
    """叠加简化的二维彩色周期摩尔纹。"""

    height, width = image_rgb.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    pattern = np.sin(2 * np.pi * (xx / 9.5 + yy / 15.5))
    pattern += 0.55 * np.sin(2 * np.pi * (xx / 17.0 - yy / 11.0))
    delta = amplitude * pattern
    degraded = image_rgb.astype(np.float32)
    degraded[..., 0] += delta
    degraded[..., 1] -= delta * 0.35
    degraded[..., 2] -= delta * 0.85
    return np.clip(degraded, 0, 255).astype(np.uint8)

