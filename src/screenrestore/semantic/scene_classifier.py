"""场景分类器：判断输入图像属于 display/cinema/artwork/glossy_artwork/document/other。

当前使用经典 CV 启发式特征 + 可选 ONNX 模型（如 CLIP）。
无外部模型时也能给出合理推断。
"""

from __future__ import annotations

import numpy as np

# 场景类型常量
SCENE_DISPLAY = "display"
SCENE_CINEMA = "cinema"
SCENE_ARTWORK = "artwork"
SCENE_GLOSSY_ARTWORK = "glossy_artwork"
SCENE_DOCUMENT = "document"
SCENE_OTHER = "other"

ALL_SCENE_TYPES = [
    SCENE_DISPLAY, SCENE_CINEMA, SCENE_ARTWORK,
    SCENE_GLOSSY_ARTWORK, SCENE_DOCUMENT, SCENE_OTHER,
]


class SceneClassifier:
    """场景分类器接口。

    默认使用启发式特征分类；可通过 set_model_backend 接入 ONNX CLIP 等模型。
    """

    def __init__(self):
        self._backend = None

    def set_model_backend(self, backend) -> None:
        """注入 ONNX 模型后端以获得更准确的分类。"""
        self._backend = backend

    def classify(self, image_rgb: np.ndarray) -> tuple[str, float]:
        """分类输入图像。

        Returns:
            (scene_type, confidence) — 场景标签和置信度 [0,1]。
        """
        if self._backend is not None:
            try:
                return self._classify_with_model(image_rgb)
            except Exception:
                pass
        return _classify_heuristic(image_rgb)

    def _classify_with_model(self, image_rgb: np.ndarray) -> tuple[str, float]:
        """使用 ONNX 模型分类（如 CLIP）。"""
        from screenrestore.core.operator import ProcessingContext

        result = self._backend.run_analysis(image_rgb, ProcessingContext(preview=True))
        top_label, confidence = result.top_label()

        # 映射模型输出标签到场景类型
        label_map = {
            "display": SCENE_DISPLAY,
            "screen": SCENE_DISPLAY,
            "monitor": SCENE_DISPLAY,
            "cinema": SCENE_CINEMA,
            "movie": SCENE_CINEMA,
            "artwork": SCENE_ARTWORK,
            "painting": SCENE_ARTWORK,
            "art": SCENE_ARTWORK,
            "glossy": SCENE_GLOSSY_ARTWORK,
            "reflection": SCENE_GLOSSY_ARTWORK,
            "glass": SCENE_GLOSSY_ARTWORK,
            "document": SCENE_DOCUMENT,
            "text": SCENE_DOCUMENT,
        }
        scene_type = label_map.get(top_label.lower(), SCENE_OTHER)
        return scene_type, confidence


def classify_scene(
    image_rgb: np.ndarray,
    classifier: SceneClassifier | None = None,
) -> tuple[str, float]:
    """便捷函数：分类场景类型。"""
    if classifier is not None:
        return classifier.classify(image_rgb)
    return _classify_heuristic(image_rgb)


# ── 启发式分类 ──────────────────────────────────────────

def _classify_heuristic(image_rgb: np.ndarray) -> tuple[str, float]:
    """基于经典 CV 特征的启发式场景分类。

    不依赖任何外部模型，适合作为 baseline 和 fallback。
    """
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 特征 1: 高亮+低饱和区域占比 → 反光概率
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0
    s = hsv[:, :, 1] / 255.0
    highlight_low_sat = float(np.mean((v > 0.85) & (s < 0.2)))

    # 特征 2: 暗区占比 → 电影院/投影概率
    dark_ratio = float(np.mean(v < 0.1))

    # 特征 3: FFT 高频周期性
    fft = np.fft.fft2(gray.astype(np.float32) / 255.0)
    fft_mag = np.abs(np.fft.fftshift(fft))
    center_y, center_x = fft_mag.shape[0] // 2, fft_mag.shape[1] // 2
    exclude_r = min(center_y, center_x) // 4
    mask = np.ones_like(fft_mag, dtype=bool)
    mask[center_y-exclude_r:center_y+exclude_r, center_x-exclude_r:center_x+exclude_r] = False
    high_freq_peak_ratio = float(
        fft_mag[mask].max() / max(fft_mag[mask].mean(), 1e-8)
    ) if mask.sum() > 0 else 1.0

    # 特征 5: 颜色分布
    avg_saturation = float(s.mean())
    # sRGB 是否是窄色域（显示器通常色彩更鲜艳）
    color_std = float(np.std(image_rgb.astype(np.float32), axis=(0, 1)).mean())

    # 特征 6: 对比度
    p5, p95 = np.percentile(gray, [5, 95])
    contrast_range = float(p95 - p5) / 255.0

    # ── 决策逻辑 ──
    scores: dict[str, float] = {}

    # DISPLAY: 高频周期性峰 (摩尔纹) + 中等饱和度 + 正常对比度
    display_score = (
        0.40 * min(high_freq_peak_ratio / 8.0, 1.0) +
        0.25 * min(avg_saturation / 0.3, 1.0) +
        0.20 * (1.0 - min(highlight_low_sat * 20, 1.0)) +
        0.15 * min(contrast_range / 0.7, 1.0)
    )
    scores[SCENE_DISPLAY] = display_score

    # CINEMA: 高暗区占比 + 低对比度 + 正常颜色
    cinema_score = (
        0.40 * min(dark_ratio / 0.15, 1.0) +
        0.30 * (1.0 - min(contrast_range / 0.5, 1.0)) +
        0.15 * (1.0 - min(high_freq_peak_ratio / 6.0, 1.0)) +
        0.15 * (1.0 - min(highlight_low_sat * 15, 1.0))
    )
    scores[SCENE_CINEMA] = cinema_score

    # ARTWORK: 中等对比度 + 低高频峰 + 低反光
    artwork_score = (
        0.30 * min(contrast_range / 0.6, 1.0) +
        0.25 * (1.0 - min(high_freq_peak_ratio / 6.0, 1.0)) +
        0.25 * (1.0 - min(highlight_low_sat * 10, 1.0)) +
        0.10 * min(avg_saturation / 0.25, 1.0) +
        0.10 * min(color_std / 60.0, 1.0)
    )
    scores[SCENE_ARTWORK] = artwork_score

    # GLOSSY_ARTWORK: 高反光 + 低暗区
    glossy_score = (
        0.45 * min(highlight_low_sat * 15, 1.0) +
        0.25 * (1.0 - min(dark_ratio / 0.1, 1.0)) +
        0.15 * (1.0 - min(high_freq_peak_ratio / 6.0, 1.0)) +
        0.15 * min(contrast_range / 0.6, 1.0)
    )
    scores[SCENE_GLOSSY_ARTWORK] = glossy_score

    # DOCUMENT: 高对比度 + 低颜色饱和度 + 低反光
    document_score = (
        0.35 * min(contrast_range / 0.8, 1.0) +
        0.30 * (1.0 - min(avg_saturation / 0.2, 1.0)) +
        0.20 * (1.0 - min(high_freq_peak_ratio / 5.0, 1.0)) +
        0.15 * (1.0 - min(highlight_low_sat * 10, 1.0))
    )
    scores[SCENE_DOCUMENT] = document_score

    # 最高分
    best = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best], 1.0)
    # 如果最高分 < 0.3，置信度太低，标为 other
    if confidence < 0.3:
        return SCENE_OTHER, confidence

    return best, round(confidence, 4)


# ── CLIP 候选重排序 ──────────────────────────────────────

def rerank_candidates_with_clip(
    image_rgb: np.ndarray,
    candidate_previews: list[np.ndarray],
    backend=None,
) -> list[float]:
    """对几何候选区域用 CLIP 重排序。

    对每个 candidate 的 warp preview 询问 "a photograph of an artwork"，
    返回每个候选的语义匹配分数。

    Args:
        image_rgb: 原始图像 (用于上下文)
        candidate_previews: 每个候选区域的 warp preview
        backend: CLIP ONNX 后端 (可选)

    Returns:
        与 candidate_previews 等长的分数列表。
    """
    if backend is None:
        # 无模型时回退到启发式：候选面积+矩形度
        return [_heuristic_candidate_score(preview) for preview in candidate_previews]

    from screenrestore.core.operator import ProcessingContext

    scores = []
    for preview in candidate_previews:
        try:
            result = backend.run_analysis(preview, ProcessingContext(preview=True))
            # 查找 artwork/painting 相关标签
            artwork_score = 0.0
            for label, conf in result.labels.items():
                if label.lower() in ("artwork", "painting", "art", "poster", "print"):
                    artwork_score = max(artwork_score, conf)
            scores.append(artwork_score if artwork_score > 0 else 0.5)
        except Exception:
            scores.append(_heuristic_candidate_score(preview))
    return scores


def _heuristic_candidate_score(preview: np.ndarray) -> float:
    """启发式候选评分（无模型时使用）。"""
    import cv2

    gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    # 梯度丰富度 → 可能有内容
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # 颜色丰富度
    color_std = float(np.std(preview.astype(np.float32)))
    return min((lap_var / 500.0) * 0.5 + (color_std / 80.0) * 0.5, 1.0)
