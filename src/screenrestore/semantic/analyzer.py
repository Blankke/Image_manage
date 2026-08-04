"""语义分析器：编排场景分类、目标定位和退化分析。

SemanticAnalyzer 是语义层的入口，负责协调各个子分析器并将结果
聚合为统一的 SceneContext。
"""

from __future__ import annotations

import numpy as np

from .context import SceneContext
from .scene_classifier import SceneClassifier, classify_scene


class SemanticAnalyzer:
    """语义分析编排器。

    当前阶段使用经典 CV 方法做退化检测和结构分析，
    CLIP/ONNX 模型分类为可选增强路径。
    """

    def __init__(self, classifier: SceneClassifier | None = None):
        self._classifier = classifier

    def analyze(self, image_rgb: np.ndarray) -> SceneContext:
        """对输入图像运行完整语义分析流水线。

        Args:
            image_rgb: H×W×3 uint8 RGB 图像。

        Returns:
            SceneContext: 聚合的语义分析结果。
        """
        ctx = SceneContext()

        # 1) 场景分类
        scene_type, confidence = classify_scene(image_rgb, self._classifier)
        ctx.scene_type = scene_type
        ctx.scene_confidence = confidence

        # 2) 退化检测（经典 CV）
        ctx.properties.update(_detect_degradations(image_rgb))

        # 3) 结构分析
        ctx.semantic_masks.update(_analyze_structure(image_rgb))

        # 4) 反光检测
        reflection_mask = _detect_reflection_region(image_rgb)
        if reflection_mask is not None:
            ctx.artifact_masks["reflection"] = reflection_mask

        # 5) 摩尔纹区域估计
        moire_mask = _estimate_moire_region(image_rgb)
        if moire_mask is not None:
            ctx.artifact_masks["moire"] = moire_mask

        return ctx


# ── 退化检测 (经典 CV，无外部模型依赖) ──────────────────

def _detect_degradations(image_rgb: np.ndarray) -> dict[str, float]:
    """基于经典 CV 的退化指标检测。"""
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    props: dict[str, float] = {}

    # 模糊估计：Laplacian variance 在 uint8 域计算
    gray_u8 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F)
    lap_var = float(lap.var())
    # 映射: var>500→锐利(blur≈0), var<30→模糊(blur≈1)
    props["blur_estimate"] = round(float(np.clip(1.0 - (lap_var - 30) / 470.0, 0.0, 1.0)), 4)

    # 噪声估计：用 median filter 残差的标准差（更稳健）
    blurred = cv2.medianBlur(gray_u8, 5).astype(np.float32)
    residual = gray_u8.astype(np.float32) - blurred
    noise_std = float(np.std(residual))
    # 映射: std<3→干净(≈0), std>15→强噪声(≈1)
    props["noise_estimate"] = round(float(np.clip((noise_std - 3.0) / 12.0, 0.0, 1.0)), 4)

    # 高光/暗部裁剪
    props["highlight_clipping"] = round(float(np.mean(gray > 0.98)), 6)
    props["shadow_clipping"] = round(float(np.mean(gray < 0.02)), 6)

    # 照明不均：分块亮度方差
    block_size = max(32, min(w, h) // 8)
    block_means = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            block_means.append(float(block.mean()))
    if len(block_means) > 4:
        props["illumination_gradient"] = round(
            min(float(np.std(block_means)) * 3.0, 1.0), 4,
        )
    else:
        props["illumination_gradient"] = 0.0

    # 黑位偏移：最暗 3% 像素的 RGB 中位数
    dark_pixels = gray < np.percentile(gray, 3)
    if dark_pixels.sum() > 50:
        dark_rgb = image_rgb[dark_pixels]
        r_med = float(np.median(dark_rgb[:, 0])) / 255.0
        g_med = float(np.median(dark_rgb[:, 1])) / 255.0
        b_med = float(np.median(dark_rgb[:, 2])) / 255.0
        props["black_level_r"] = round(r_med, 4)
        props["black_level_g"] = round(g_med, 4)
        props["black_level_b"] = round(b_med, 4)
    else:
        props["black_level_r"] = 0.0
        props["black_level_g"] = 0.0
        props["black_level_b"] = 0.0

    return props


# ── 结构分析 ─────────────────────────────────────────────

def _analyze_structure(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """检测需要保护的结构区域（平坦区、纹理区）。"""
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 梯度幅度 → 区分平坦区和纹理区（在 uint8 域计算）
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    grad_p95 = np.percentile(grad_mag, 95)
    if grad_p95 < 1:
        grad_p95 = 1.0

    # 平坦区域
    flat_mask = (grad_mag < grad_p95 * 0.08).astype(np.uint8) * 255

    # 细纹理区域（中高梯度但不极端）
    fine_texture = (
        (grad_mag > grad_p95 * 0.15) & (grad_mag < grad_p95 * 0.5)
    ).astype(np.uint8) * 255

    return {
        "flat_region": flat_mask,
        "fine_texture": fine_texture,
    }


# ── 反光检测 ─────────────────────────────────────────────

def _detect_reflection_region(image_rgb: np.ndarray) -> np.ndarray | None:
    """HSV 高亮+低饱和区域检测。仅当面积 > 0.5% 时才视为有效反光。"""
    import cv2

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0
    s = hsv[:, :, 1] / 255.0

    reflection_mask = ((v > 0.85) & (s < 0.2)).astype(np.uint8) * 255
    h, w = reflection_mask.shape

    # 仅当面积 > 0.5% 时才报告
    if reflection_mask.sum() < h * w * 0.005 * 255:
        return None

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
    reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
    return reflection_mask


# ── 摩尔纹区域估计 ───────────────────────────────────────

def _estimate_moire_region(image_rgb: np.ndarray) -> np.ndarray | None:
    """基于 FFT 高频周期性检测摩尔纹区域。"""
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    block_size = min(128, min(h, w) // 3)
    if block_size < 32:
        return None

    moire_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            y2 = min(y + block_size, h)
            x2 = min(x + block_size, w)
            patch = gray[y:y2, x:x2]

            # FFT 周期性分析
            fft = np.fft.fft2(patch)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log(np.abs(fft_shifted) + 1)

            # 检测是否有突出的高频峰值
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            # 排除直流分量周围
            exclude_radius = max(3, min(center_y, center_x) // 6)
            mask = np.ones_like(magnitude, dtype=bool)
            mask[center_y-exclude_radius:center_y+exclude_radius,
                 center_x-exclude_radius:center_x+exclude_radius] = False

            if mask.sum() > 0:
                outer = magnitude[mask]
                peak_ratio = float(outer.max() / (outer.mean() + 1e-8))
                # 高频峰/均值比 → 周期性强度
                periodicity = min(peak_ratio / 10.0, 1.0)

                moire_map[y:y2, x:x2] += periodicity
                weight_map[y:y2, x:x2] += 1.0

    if weight_map.max() < 1:
        return None

    moire_map /= (weight_map + 1e-8)
    # 提高阈值：仅周期性强 (>0.5) 的区域
    moire_mask = (moire_map > 0.5).astype(np.uint8) * 255

    # 要求面积 > 2%
    h, w = moire_mask.shape
    if moire_mask.sum() < h * w * 0.02 * 255:
        return None

    return moire_mask
