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
    gray_u8 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    props: dict[str, float] = {}

    # ── 模糊估计：仅在非摩尔纹、非平坦区计算 ──
    # 先粗略排除 flat_region 和极暗区
    grad_mag = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 1, ksize=3)
    texture_mask = (np.abs(grad_mag) > np.percentile(np.abs(grad_mag), 20))
    if texture_mask.sum() > 500:
        lap = cv2.Laplacian(gray_u8, cv2.CV_32F)
        lap_texture = lap[texture_mask]
        lap_var = float(np.var(lap_texture))
    else:
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_32F).var())
    # 映射: var>600→锐利(blur≈0), var<20→模糊(blur≈1)
    props["blur_estimate"] = round(float(np.clip(1.0 - (lap_var - 20) / 580.0, 0.0, 1.0)), 4)

    # ── 噪声估计：仅在 flat_region 用 MAD ──
    grad_mag_f = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 1, ksize=3)
    flat = (np.abs(grad_mag_f) < np.percentile(np.abs(grad_mag_f), 15))
    if flat.sum() > 200:
        flat_vals = gray_u8[flat].astype(np.float32)
        # MAD (Median Absolute Deviation) → 更鲁棒的噪声估计
        median_val = np.median(flat_vals)
        mad = float(np.median(np.abs(flat_vals - median_val)))
        # 映射: MAD<1.5→干净(≈0), MAD>12→强噪声(≈1)
        props["noise_estimate"] = round(float(np.clip((mad - 1.5) / 10.5, 0.0, 1.0)), 4)
    else:
        props["noise_estimate"] = 0.0

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

    # ── Screen lattice 检测 ──
    lattice = _estimate_screen_lattice(image_rgb)
    props.update(lattice)

    return props


def _estimate_screen_lattice(image_rgb: np.ndarray) -> dict[str, float]:
    """检测屏幕像素栅格 (screen lattice) — v10 确定性版本。

    改动:
    - 用固定 3×3 网格替代 np.random.randint (determinism)
    - 按频谱强度排序峰值 (取前 8 个最强峰)
    - 检测 conjugate symmetric pairs (FFT 对称性)
    - 不再假设方向一定是 0/π/2 (拍摄角度使屏幕可旋转)
    - 检测主导的两个近似正交方向
    """
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    block_size = min(256, min(h, w) // 2)
    if block_size < 64:
        return {"screen_lattice_confidence": 0.0}

    # 固定 3×3 网格 — 确定性!
    grid_positions = []
    for gy in range(3):
        for gx in range(3):
            y = max(0, min(h - block_size, gy * (h - block_size) // 2))
            x = max(0, min(w - block_size, gx * (w - block_size) // 2))
            grid_positions.append((y, x))

    all_peak_pairs = []  # (freq, angle1, angle2, strength)

    for y, x in grid_positions:
        patch = gray[y:y+block_size, x:x+block_size]
        fft = np.fft.fft2(patch)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        log_mag = np.log1p(magnitude)
        cy, cx = log_mag.shape[0] // 2, log_mag.shape[1] // 2
        exclude = max(3, min(cy, cx) // 8)
        log_mag_c = log_mag.copy()
        log_mag_c[cy-exclude:cy+exclude, cx-exclude:cx+exclude] = 0

        if log_mag_c.max() <= 0:
            continue

        # 找最强 8 个峰 — 按强度排序
        local_max = (log_mag_c > cv2.dilate(log_mag_c, np.ones((5,5))) - 1e-8) & (log_mag_c > 0)
        peak_coords = np.argwhere(local_max)
        if len(peak_coords) < 2:
            continue

        peak_vals = np.array([log_mag_c[py, px] for py, px in peak_coords])
        top_idx = np.argsort(peak_vals)[-8:]  # 最强 8 个
        top_peaks = peak_coords[top_idx]

        # 检测 conjugate symmetric pairs
        peaks_info = []
        for py, px in top_peaks:
            dy, dx = py - cy, px - cx
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 4:
                continue
            freq = dist / block_size
            angle = np.arctan2(dy, dx)
            strength = float(log_mag_c[py, px])
            peaks_info.append((freq, angle, strength, py, px))

        # 找 conjugate pairs: 峰 (dy,dx) 和 (-dy,-dx) 应成对出现
        for i, (f1, a1, s1, py1, px1) in enumerate(peaks_info):
            for j, (f2, a2, s2, py2, px2) in enumerate(peaks_info):
                if j <= i:
                    continue
                # conjugate: 频率相近，方向相反
                if abs(f1 - f2) / max(f1 + f2, 1e-8) < 0.2:
                    # 检查是否大致对称: (dx2, dy2) ≈ -(dx1, dy1)
                    dx1, dy1 = px1 - cx, py1 - cy
                    dx2, dy2 = px2 - cx, py2 - cy
                    dot = (dx1 * dx2 + dy1 * dy2) / max(
                        np.sqrt(dx1**2+dy1**2) * np.sqrt(dx2**2+dy2**2), 1e-8
                    )
                    if dot < -0.7:  # 接近 -1 = 完全反向
                        all_peak_pairs.append((f1, a1, a2, (s1 + s2) / 2))

    if len(all_peak_pairs) < 2:
        return {"screen_lattice_confidence": 0.0}

    # 分析主导方向: 检测两个近似正交的 lattice direction
    freqs = np.array([p[0] for p in all_peak_pairs])
    angles1 = np.array([p[1] for p in all_peak_pairs])
    angles2 = np.array([p[2] for p in all_peak_pairs])
    strengths = np.array([p[3] for p in all_peak_pairs])

    # 频率一致性
    freq_mean = float(np.mean(freqs))
    freq_std = float(np.std(freqs))
    freq_consistency = 1.0 - min(freq_std / max(freq_mean, 1e-8), 1.0)

    # 角度一致性: 所有峰应来自两个正交 lattice direction
    # 将角度归一到 [0, π)
    all_angles = np.concatenate([angles1 % np.pi, angles2 % np.pi])
    # 用 circular mean 检测是否有两个主导方向
    # 简化: 检查角度是否聚成两个相差 ~π/2 的簇
    angle_sorted = np.sort(all_angles)
    gaps = np.diff(angle_sorted)
    if len(gaps) >= 2:
        max_gap_idx = np.argmax(gaps)
        cluster1 = angle_sorted[:max_gap_idx + 1]
        cluster2 = angle_sorted[max_gap_idx + 1:]
        if len(cluster1) >= 2 and len(cluster2) >= 2:
            c1_mean = float(np.mean(cluster1))
            c2_mean = float(np.mean(cluster2))
            orthogonality = abs(abs(c1_mean - c2_mean) - np.pi / 2) / (np.pi / 2)
            orientation_consistency = 1.0 - min(orthogonality, 1.0)
        else:
            orientation_consistency = 0.0
    else:
        orientation_consistency = 0.0

    # 强度
    avg_strength = float(np.mean(strengths))

    confidence = float(
        0.35 * freq_consistency
        + 0.35 * orientation_consistency
        + 0.30 * min(avg_strength / 8.0, 1.0)
    )

    return {
        "screen_lattice_confidence": round(min(confidence, 1.0), 4),
        "screen_frequency": round(freq_mean, 4),
        "screen_orientation_consistency": round(orientation_consistency, 4),
        "n_peak_pairs": len(all_peak_pairs),
    }


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
    """基于 FFT 高频周期性检测摩尔纹区域。

    结合 screen_lattice evidence：高周期性与 screen lattice 一致时 moire 置信度更高。
    """
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    block_size = min(128, min(h, w) // 3)
    if block_size < 32:
        return None

    screen_conf = _estimate_screen_lattice(image_rgb).get("screen_lattice_confidence", 0.0)
    # 非 display 场景需要更高阈值
    min_periodicity = 0.35 if screen_conf > 0.5 else 0.6

    moire_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            y2 = min(y + block_size, h)
            x2 = min(x + block_size, w)
            patch = gray[y:y2, x:x2]

            fft = np.fft.fft2(patch)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log(np.abs(fft_shifted) + 1)

            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            exclude_radius = max(3, min(center_y, center_x) // 6)
            mask = np.ones_like(magnitude, dtype=bool)
            mask[center_y-exclude_radius:center_y+exclude_radius,
                 center_x-exclude_radius:center_x+exclude_radius] = False

            if mask.sum() > 0:
                outer = magnitude[mask]
                peak_ratio = float(outer.max() / (outer.mean() + 1e-8))
                periodicity = min(peak_ratio / 10.0, 1.0)

                # screen lattice 证据增强 moire 置信度
                if screen_conf > 0.5:
                    periodicity = min(periodicity * (1.0 + screen_conf * 0.5), 1.0)

                moire_map[y:y2, x:x2] += periodicity
                weight_map[y:y2, x:x2] += 1.0

    if weight_map.max() < 1:
        return None

    moire_map /= (weight_map + 1e-8)
    moire_mask = (moire_map > min_periodicity).astype(np.uint8) * 255

    h_m, w_m = moire_mask.shape
    min_area_ratio = 0.01 if screen_conf > 0.5 else 0.03
    if moire_mask.sum() < h_m * w_m * min_area_ratio * 255:
        return None

    return moire_mask
