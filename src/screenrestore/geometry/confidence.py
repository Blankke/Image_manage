"""无人值守几何接受/拒绝策略。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CorrectnessCalibrator
from .types import EdgeRefinement, QuadPrediction, RejectionReason, TargetClass

CORRECTNESS_FEATURE_NAMES = (
    "presence",
    "outer_presence",
    "class",
    "layer_consistency",
    "corner_peak_min",
    "corner_peak_mean",
    "corner_peak_difference_min",
    "corner_peak_ratio_min",
    "corner_entropy_mean",
    "corner_entropy_max",
    "corner_sharpness_min",
    "boundary_support_min",
    "boundary_support_mean",
    "line_residual_median_max",
    "line_residual_p95_max",
    "continuous_coverage_min",
    "gradient_normal_alignment_min",
    "boundary_consistency_mean",
    "refine_displacement_mean",
    "refine_displacement_max",
    "area_ratio",
    "area_drift",
    "aspect_drift",
    "mask_consistency",
)


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
    calibrator: CorrectnessCalibrator | None = None

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
        candidate_margin = 0.0
        if prediction.candidates:
            candidate_margin = float(
                prediction.candidates[0].scores.get("candidate_margin", 0.0)
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
        if prediction.candidates and candidate_margin < self.min_candidate_margin:
            reasons.append(RejectionReason.SCORE_AMBIGUOUS)
        if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
            reasons.append(RejectionReason.OUT_OF_SCOPE)
        if combined < self.min_combined and not reasons:
            reasons.append(RejectionReason.CORNER_UNCERTAIN)
        diagnostics = _correctness_features(
            prediction,
            refinement,
            image_shape,
            area_ratio=area_ratio,
            candidate_margin=candidate_margin,
        )
        diagnostics.update({
            "presence_confidence": float(prediction.presence_confidence),
            "outer_presence_confidence": float(prediction.outer_presence_confidence),
            "class_confidence": float(prediction.class_confidence),
            "minimum_corner_confidence": corner_score,
            "boundary_support": float(boundary_score),
            "layer_confidence": float(prediction.layer_confidence),
            "candidate_margin": candidate_margin,
            "area_ratio": area_ratio,
        })
        if self.calibrator is not None:
            calibrated = self.calibrator.predict_probability(diagnostics)
            diagnostics["calibrated_strict_correct_probability"] = calibrated
            combined = calibrated
            if calibrated < self.calibrator.threshold and not reasons:
                reasons.append(RejectionReason.SCORE_AMBIGUOUS)
        return combined, tuple(dict.fromkeys(reasons)), diagnostics


def _correctness_features(
    prediction: QuadPrediction,
    refinement: EdgeRefinement | None,
    image_shape: tuple[int, ...],
    *,
    area_ratio: float,
    candidate_margin: float,
) -> dict[str, float]:
    """把 decoder、mask、精修与几何证据展开为稳定的校准器数值特征。"""

    corner_items: list[dict[str, object]] = []
    content = prediction.decoder_diagnostics.get("content")
    if isinstance(content, dict) and isinstance(content.get("corners"), list):
        corner_items = [item for item in content["corners"] if isinstance(item, dict)]

    def values(name: str, default: float) -> list[float]:
        result = [float(item.get(name, default)) for item in corner_items]
        return result or [default] * 4

    peak1 = values("peak1", min(prediction.corner_confidences))
    peak_difference = values("peak_difference", candidate_margin)
    peak_ratio = values("peak_ratio", 1.0)
    entropy = values("normalized_entropy", 1.0)
    sharpness = values("local_sharpness", 0.0)
    diagonal = max(1.0, float(np.hypot(image_shape[1], image_shape[0])))
    if refinement is None:
        residual_median = residual_p95 = 999.0
        coverage = alignment = boundary = 0.0
        displacement_mean = displacement_max = 1.0
        area_drift = aspect_drift = 1.0
    else:
        residual_median = float(max(refinement.residual_median))
        residual_p95 = float(max(refinement.residual_p95))
        coverage = float(min(refinement.continuous_coverage))
        alignment = float(min(refinement.gradient_normal_alignment))
        boundary = float(np.mean(refinement.boundary_consistency))
        displacement_mean = float(np.mean(refinement.corner_shifts)) / diagonal
        displacement_max = float(np.max(refinement.corner_shifts)) / diagonal
        area_drift = float(refinement.area_drift)
        aspect_drift = float(refinement.aspect_drift)
    return {
        "presence": float(prediction.presence_confidence),
        "outer_presence": float(prediction.outer_presence_confidence),
        "class": float(prediction.class_confidence),
        "layer_consistency": float(prediction.layer_confidence),
        "corner_peak_min": float(min(peak1)),
        "corner_peak_mean": float(np.mean(peak1)),
        "corner_peak_difference_min": float(min(peak_difference)),
        "corner_peak_ratio_min": float(min(peak_ratio)),
        "corner_entropy_mean": float(np.mean(entropy)),
        "corner_entropy_max": float(max(entropy)),
        "corner_sharpness_min": float(min(sharpness)),
        "boundary_support_min": float(min(refinement.edge_support)) if refinement else 0.0,
        "boundary_support_mean": float(refinement.mean_support) if refinement else 0.0,
        "line_residual_median_max": residual_median,
        "line_residual_p95_max": residual_p95,
        "continuous_coverage_min": coverage,
        "gradient_normal_alignment_min": alignment,
        "boundary_consistency_mean": boundary,
        "refine_displacement_mean": displacement_mean,
        "refine_displacement_max": displacement_max,
        "area_ratio": area_ratio,
        "area_drift": area_drift,
        "aspect_drift": aspect_drift,
        "mask_consistency": _mask_quad_consistency(prediction, image_shape),
    }


def _mask_quad_consistency(
    prediction: QuadPrediction,
    image_shape: tuple[int, ...],
) -> float:
    if prediction.content_mask is None or prediction.content_quad is None:
        return 0.0
    mask = np.asarray(prediction.content_mask, np.float32).squeeze()
    if mask.ndim != 2:
        return 0.0
    polygon = np.zeros(mask.shape, np.uint8)
    scale = np.array(
        [
            (mask.shape[1] - 1) / max(1, image_shape[1] - 1),
            (mask.shape[0] - 1) / max(1, image_shape[0] - 1),
        ],
        np.float32,
    )
    cv2.fillConvexPoly(polygon, np.rint(prediction.content_quad * scale).astype(np.int32), 1)
    observed = mask >= 0.5
    intersection = np.logical_and(polygon > 0, observed).sum()
    union = np.logical_or(polygon > 0, observed).sum()
    return float(intersection / max(1, union))


__all__ = ["CORRECTNESS_FEATURE_NAMES", "ConfidencePolicy"]
