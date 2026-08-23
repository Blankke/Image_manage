"""把统一自动几何决策写入语义上下文。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.geometry import (
    AutomaticGeometryService,
    ConfidencePolicy,
    QuadDetector,
    target_class_for_scene,
)

from .context import LocalizationCandidate, SceneContext


class TargetLocalizer:
    """语义分析层到 ``AutomaticGeometryService`` 的轻量适配器。

    旧版 Canny/Hough/CLIP 候选混合已经退出正式路径。所有客户端与评估现在共享同一
    个接受/拒绝决策；只有被接受的 content quad 才会写入 ``target_polygon``。
    """

    def __init__(
        self,
        detector: QuadDetector | None = None,
        *,
        policy: ConfidencePolicy | None = None,
    ) -> None:
        self._service = AutomaticGeometryService(detector, policy=policy)

    def localize(self, image_rgb: np.ndarray, ctx: SceneContext) -> SceneContext:
        """定位内容层并保留拒绝诊断，不原地修改输入图像。"""

        decision = self._service.localize(image_rgb, target_class_for_scene(ctx.scene_type))
        ctx.localization_status = decision.status.value
        ctx.localization_confidence = decision.confidence
        ctx.localization_backend = decision.backend
        ctx.localization_rejection_reasons = tuple(
            reason.value for reason in decision.rejection_reasons
        )
        ctx.aspect_ratio = decision.aspect.ratio if decision.aspect is not None else None
        ctx.aspect_confidence = (
            decision.aspect.confidence if decision.aspect is not None else 0.0
        )
        ctx.outer_polygon = (
            decision.outer_corners.copy() if decision.outer_corners is not None else None
        )
        ctx.localization_candidates = [
            LocalizationCandidate(
                polygon=candidate.corners.copy(),
                source=candidate.source,
                runtime_score=candidate.confidence,
                geometry_score=float(candidate.scores.get("edge_strength", candidate.confidence)),
                semantic_score=float(candidate.scores.get("semantic", 0.0)),
                layer=candidate.layer.value,
            )
            for candidate in decision.candidates
        ]
        ctx.target_polygon = None
        ctx.target_bbox = None
        ctx.target_mask = None
        if not decision.accepted or decision.proposed_corners is None:
            return ctx
        polygon = decision.proposed_corners.astype(np.float32)
        ctx.target_polygon = polygon
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        ctx.target_bbox = (
            int(np.floor(x_min)),
            int(np.floor(y_min)),
            int(np.ceil(x_max - x_min + 1)),
            int(np.ceil(y_max - y_min + 1)),
        )
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        ctx.target_mask = mask
        return ctx
