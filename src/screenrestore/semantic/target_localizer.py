"""目标定位器：在拍摄图中找到目标内容区域。

使用现有 Geometry detector 生成候选四边形，
结合几何评分 + (可选 CLIP 语义重排序) 确定最优候选。
输出 target_bbox / target_polygon / target_mask 供 SceneContext 使用。

无外部模型时使用纯几何评分作为 fallback。
"""

from __future__ import annotations

import numpy as np

from .context import SceneContext


class TargetLocalizer:
    """目标内容区域定位器。

    对拍摄图像检测目标平面（画作/屏幕/海报），输出 SceneContext 中的
    target_bbox、target_polygon 和 target_mask。
    """

    def __init__(self, clip_backend=None):
        self._clip_backend = clip_backend

    def localize(
        self,
        image_rgb: np.ndarray,
        ctx: SceneContext,
    ) -> SceneContext:
        """在图像中定位目标内容区域。

        Args:
            image_rgb: 输入 H×W×3 RGB uint8
            ctx: 现有 SceneContext（更新其 target 字段）

        Returns:
            更新后的 SceneContext
        """
        candidates = self._generate_candidates(image_rgb)

        if not candidates:
            return ctx  # 没有候选，返回原始 ctx

        # 如果有 CLIP backend，用语义重排序
        if self._clip_backend is not None:
            candidates = self._rerank_with_clip(image_rgb, candidates, ctx.scene_type)

        # 选最优候选
        best = max(candidates, key=lambda c: c["score"])

        if best["score"] < 0.3:
            return ctx  # 置信度过低

        # 填 context
        poly = best["polygon"]
        ctx.target_polygon = poly.astype(np.float32)
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        ctx.target_bbox = (
            int(x_min), int(y_min),
            int(x_max - x_min + 1), int(y_max - y_min + 1),
        )

        # 生成 mask
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        ctx.target_mask = mask

        return ctx

    def _generate_candidates(self, image_rgb: np.ndarray) -> list[dict]:
        """用现有 Geometry detector 生成候选四边形。"""
        import cv2

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        # 膨胀连接
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < w * h * 0.02:  # 至少占 2%
                continue
            if area > w * h * 0.95:  # 排除整图轮廓
                continue

            # 多边形拟合
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                poly = approx.reshape(4, 2).astype(np.float32)
            else:
                # 取最小外接矩形
                rect = cv2.minAreaRect(contour)
                poly = cv2.boxPoints(rect)

            # 排序为 TL→TR→BR→BL
            poly = _order_corners(poly)

            # 几何评分
            geo_score = _score_candidate(poly, gray, w, h)

            candidates.append({
                "polygon": poly,
                "score": geo_score,
                "geometry_score": geo_score,
                "semantic_score": 0.0,
            })

        return candidates

    def _rerank_with_clip(
        self,
        image_rgb: np.ndarray,
        candidates: list[dict],
        scene_type: str,
    ) -> list[dict]:
        """用 CLIP backend 对候选重排序。"""
        import cv2

        rh, rw = 224, 224  # CLIP 标准输入

        for cand in candidates:
            poly = cand["polygon"]
            # 做小尺寸 perspective preview
            dst = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]])
            M = cv2.getPerspectiveTransform(poly, dst)
            preview = cv2.warpPerspective(image_rgb, M, (rw, rh))

            try:
                from screenrestore.core.operator import ProcessingContext
                result = self._clip_backend.run_analysis(preview, ProcessingContext(preview=True))
                # 查找与 scene_type 匹配的标签
                semantic_score = 0.0
                target_labels = {
                    "artwork": ("artwork", "painting", "art", "poster", "photograph"),
                    "glossy_artwork": ("artwork", "painting", "art", "poster"),
                    "display": ("screen", "display", "monitor"),
                    "cinema": ("movie", "cinema", "screen"),
                    "document": ("text", "document", "paper"),
                }
                keywords = target_labels.get(scene_type, ("artwork", "screen"))
                for label, conf in result.labels.items():
                    if any(kw in label.lower() for kw in keywords):
                        semantic_score = max(semantic_score, conf)
                cand["semantic_score"] = semantic_score
                cand["score"] = 0.35 * cand["geometry_score"] + 0.65 * semantic_score
            except Exception:
                cand["semantic_score"] = 0.0
                cand["score"] = cand["geometry_score"]

        return candidates


def _order_corners(poly: np.ndarray) -> np.ndarray:
    """将四角排序为 TL→TR→BR→BL。"""
    center = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
    ordered = poly[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[3, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def _score_candidate(poly: np.ndarray, gray: np.ndarray, w: int, h: int) -> float:
    """几何评分：矩形度 + 边缘强度 + 面积合理性 + 中心位置。"""
    import cv2

    # 矩形度：对边长度比
    top_len = np.linalg.norm(poly[1] - poly[0])
    bot_len = np.linalg.norm(poly[2] - poly[3])
    left_len = np.linalg.norm(poly[3] - poly[0])
    right_len = np.linalg.norm(poly[2] - poly[1])
    h_ratio = min(top_len, bot_len) / max(top_len, bot_len, 1)
    v_ratio = min(left_len, right_len) / max(left_len, right_len, 1)
    rectangularity = (h_ratio + v_ratio) / 2.0

    # 边缘强度：沿四条边的梯度均值
    edge_strength = 0.0
    grad_mag = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
    for i in range(4):
        p1, p2 = poly[i], poly[(i + 1) % 4]
        for t in np.linspace(0, 1, 20):
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < w and 0 <= py < h:
                edge_strength += float(grad_mag[py, px])
    edge_strength = min(edge_strength / (80.0 * 255.0), 1.0)

    # 面积合理性：太大/太小扣分
    area = cv2.contourArea(poly.astype(np.float32))
    area_ratio = area / (w * h)
    area_score = 1.0 - abs(area_ratio - 0.5) * 1.5  # 50% 最理想
    area_score = max(0.0, min(1.0, area_score))

    # 中心位置
    cx, cy = poly.mean(axis=0)
    center_score = 1.0 - (abs(cx - w / 2) / (w / 2) * 0.3 + abs(cy - h / 2) / (h / 2) * 0.3)

    return float(
        0.30 * rectangularity
        + 0.25 * edge_strength
        + 0.25 * area_score
        + 0.20 * center_score
    )


# 导入 cv2 供 fillPoly 使用
import cv2
