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
        """用边缘+轮廓检测生成候选四边形。

        改动(v10):
        - RETR_EXTERNAL → RETR_LIST (找到嵌套轮廓)
        - 删除 minAreaRect fallback — 非四边形候选不加惩罚但不参与竞争
        - 删除 50% area prior
        - 梯形不因对边不等长而扣分
        """
        import cv2

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        # 多级 Canny
        candidates = []
        for low_t, high_t in [(30, 100), (50, 150)]:
            edges = cv2.Canny(gray, low_t, high_t)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            # RETR_LIST 保留所有轮廓（包括嵌套）
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < w * h * 0.015:
                    continue
                if area > w * h * 0.92:
                    continue

                # 先取凸包 — 自然边缘轮廓几乎从不凸
                hull = cv2.convexHull(contour)
                if hull is None or len(hull) < 4:
                    continue

                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

                if len(approx) != 4:
                    continue

                poly = approx.reshape(4, 2).astype(np.float32)
                if not _is_valid_polygon(poly, w, h):
                    continue

                poly = _order_corners(poly)
                geo_score = _score_candidate_v2(poly, gray, w, h)

                candidates.append({
                    "polygon": poly,
                    "score": geo_score,
                    "geometry_score": geo_score,
                    "semantic_score": 0.0,
                })

        return candidates

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


def _score_candidate_v2(poly: np.ndarray, gray: np.ndarray, w: int, h: int) -> float:
    """v10 几何评分：凸性 + 最小内角 + 边缘法向强度 + 方向一致性。"""
    import cv2

    # 1) 凸性
    contour = poly.astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return 0.0

    # 2) 最小内角
    min_angle = _min_interior_angle(poly)
    if min_angle < 15 or min_angle > 170:
        return 0.0

    # 3) 面积合法性（仅在排除极端）
    area = cv2.contourArea(poly.astype(np.float32))
    area_ratio = area / (w * h)
    if area_ratio < 0.01 or area_ratio > 0.90:
        return 0.0

    # 4) 边缘法向强度 — 用正确梯度幅值 sqrt(Gx²+Gy²)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x.astype(np.float64)**2 + grad_y.astype(np.float64)**2)

    edge_score = 0.0
    for i in range(4):
        p1, p2 = poly[i], poly[(i + 1) % 4]
        edge_vec = p2 - p1
        length = float(np.linalg.norm(edge_vec))
        if length < 5:
            continue
        # 法向
        normal = np.array([-edge_vec[1], edge_vec[0]]) / length
        # 沿边采样，向外偏移几个像素检测边缘
        for t in np.linspace(0.1, 0.9, 8):
            mid = p1 + t * (p2 - p1)
            for offset in [2, 4, -2, -4]:
                px = int(mid[0] + normal[0] * offset)
                py = int(mid[1] + normal[1] * offset)
                if 0 <= px < w and 0 <= py < h:
                    edge_score += float(grad_mag[py, px])
    edge_score = min(edge_score / (128.0 * 32.0), 1.0)

    # 5) 矩形度：仅检查对边方向一致性（不要求长度相等 = 容忍透视）
    top_dir = poly[1] - poly[0]
    bot_dir = poly[2] - poly[3]
    left_dir = poly[3] - poly[0]
    right_dir = poly[2] - poly[1]
    # 方向相似度 (cosine)
    h_dot = np.dot(top_dir, bot_dir) / max(np.linalg.norm(top_dir) * np.linalg.norm(bot_dir), 1e-8)
    v_dot = np.dot(left_dir, right_dir) / max(np.linalg.norm(left_dir) * np.linalg.norm(right_dir), 1e-8)
    directionality = max(0.0, (h_dot + v_dot) / 2.0)

    return float(0.50 * edge_score + 0.50 * directionality)


def _min_interior_angle(poly: np.ndarray) -> float:
    """计算四边形最小内角(度)。"""
    angles = []
    for i in range(4):
        a = poly[i]
        b = poly[(i + 1) % 4]
        c = poly[(i + 2) % 4]
        v1 = a - b
        v2 = c - b
        cos_angle = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angles.append(float(np.degrees(np.arccos(cos_angle))))
    return min(angles)


def _is_valid_polygon(poly: np.ndarray, w: int, h: int) -> bool:
    """检查四边形是否合法：在边界内、非退化。"""
    # clip check — 允许少量越界（框可能跨出画面）
    if np.any(poly[:, 0] < -w * 0.2) or np.any(poly[:, 0] > w * 1.2):
        return False
    if np.any(poly[:, 1] < -h * 0.2) or np.any(poly[:, 1] > h * 1.2):
        return False
    # 最小边长
    for i in range(4):
        if np.linalg.norm(poly[i] - poly[(i+1)%4]) < 5:
            return False
    return True


# 保留旧版 _score_candidate 作为别名
_score_candidate = _score_candidate_v2


# 导入 cv2 供 fillPoly 使用
import cv2
