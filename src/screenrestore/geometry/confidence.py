"""无人值守几何接受/拒绝策略。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import EdgeRefinement, QuadPrediction, RejectionReason, TargetClass


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """产品级 fail-closed 阈值。

    阈值需要在独立真实验证集上校准。当前默认值刻意保守，用来阻止旧版“有候选就采用”
    的行为；它们不构成已达到 99% 正确率的声明。
    """

    min_presence: float = 0.66
    min_class: float = 0.58
    min_corner: float = 0.52
    min_boundary_support: float = 0.16
    min_layer: float = 0.58
    min_combined: float = 0.68
    min_candidate_margin: float = 0.06
    min_area_ratio: float = 0.035
    max_area_ratio: float = 0.985

    def assess(
        self,
        prediction: QuadPrediction,
        refinement: EdgeRefinement | None,
        image_shape: tuple[int, ...],
    ) -> tuple[float, tuple[RejectionReason, ...], dict[str, float]]:
        """返回综合分、拒绝原因和有限分项诊断。"""

        if prediction.content_quad is None:
            return 0.0, (RejectionReason.NO_CANDIDATE,), {}
        corners = refinement.corners if refinement is not None else prediction.content_quad
        area_ratio = abs(float(cv2.contourArea(corners))) / max(
            1.0,
            float(image_shape[0] * image_shape[1]),
        )
        corner_score = float(min(prediction.corner_confidences))
        boundary_score = refinement.mean_support if refinement is not None else 0.0
        candidate_margin = 1.0
        if prediction.candidates:
            candidate_margin = float(
                prediction.candidates[0].scores.get("candidate_margin", 1.0)
            )
        components = np.array(
            [
                prediction.presence_confidence,
                prediction.class_confidence,
                corner_score,
                boundary_score,
                prediction.layer_confidence,
            ],
            dtype=np.float64,
        )
        weights = np.array([0.22, 0.14, 0.24, 0.24, 0.16], dtype=np.float64)
        combined = float(np.dot(components, weights))
        reasons: list[RejectionReason] = []
        if prediction.presence_confidence < self.min_presence:
            reasons.append(RejectionReason.TARGET_ABSENT)
        if prediction.target_class == TargetClass.NONE or prediction.class_confidence < self.min_class:
            reasons.append(RejectionReason.TARGET_CLASS_UNCERTAIN)
        if corner_score < self.min_corner:
            reasons.append(RejectionReason.CORNER_UNCERTAIN)
        if refinement is None or not refinement.accepted or boundary_score < self.min_boundary_support:
            reasons.append(RejectionReason.BOUNDARY_UNCERTAIN)
        if prediction.layer_confidence < self.min_layer:
            reasons.append(RejectionReason.LAYER_AMBIGUOUS)
        if candidate_margin < self.min_candidate_margin:
            reasons.append(RejectionReason.SCORE_AMBIGUOUS)
        if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
            reasons.append(RejectionReason.OUT_OF_SCOPE)
        if combined < self.min_combined and not reasons:
            reasons.append(RejectionReason.CORNER_UNCERTAIN)
        diagnostics = {
            "presence_confidence": float(prediction.presence_confidence),
            "outer_presence_confidence": float(prediction.outer_presence_confidence),
            "class_confidence": float(prediction.class_confidence),
            "minimum_corner_confidence": corner_score,
            "boundary_support": float(boundary_score),
            "layer_confidence": float(prediction.layer_confidence),
            "candidate_margin": candidate_margin,
            "area_ratio": area_ratio,
        }
        return combined, tuple(dict.fromkeys(reasons)), diagnostics
