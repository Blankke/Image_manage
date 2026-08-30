"""统一编排语义四边形、原图边缘精修和拒绝策略。"""

from __future__ import annotations

import numpy as np

from .confidence import ConfidencePolicy
from .detector import ClassicQuadDetector, QuadDetector
from .edge_refine import EdgeRefineParameters, refine_quad_edges
from .rectify import estimate_aspect, order_corners
from .types import (
    LocalizationDecision,
    LocalizationStatus,
    QuadPrediction,
    RejectionReason,
    TargetClass,
    TargetLayer,
    quadrilateral_is_valid,
)


class AutomaticGeometryService:
    """CLI、Web、GUI 与 benchmark 共用的唯一自动定位入口。"""

    def __init__(
        self,
        detector: QuadDetector | None = None,
        *,
        policy: ConfidencePolicy | None = None,
        refine_parameters: EdgeRefineParameters | None = None,
    ) -> None:
        self.detector = detector or ClassicQuadDetector()
        self.policy = policy or ConfidencePolicy()
        self.refine_parameters = refine_parameters or EdgeRefineParameters()

    def localize(
        self,
        image_rgb: np.ndarray,
        target_hint: TargetClass | None = None,
    ) -> LocalizationDecision:
        """只依据输入照片生成接受/拒绝结果，绝不读取 clean reference。"""

        prediction = self.detector.predict(image_rgb, target_hint)
        if prediction.content_quad is None:
            return LocalizationDecision(
                status=LocalizationStatus.REJECTED,
                proposed_corners=None,
                coarse_corners=None,
                outer_corners=prediction.outer_quad,
                target_class=prediction.target_class,
                layer=TargetLayer.CONTENT,
                confidence=0.0,
                aspect=None,
                backend=prediction.backend,
                rejection_reasons=(RejectionReason.NO_CANDIDATE,),
                candidates=prediction.candidates,
                diagnostics=_prediction_diagnostics(prediction),
            )
        try:
            coarse = order_corners(prediction.content_quad)
        except ValueError:
            return LocalizationDecision(
                status=LocalizationStatus.REJECTED,
                proposed_corners=None,
                coarse_corners=None,
                outer_corners=prediction.outer_quad,
                target_class=prediction.target_class,
                layer=TargetLayer.CONTENT,
                confidence=0.0,
                aspect=None,
                backend=prediction.backend,
                rejection_reasons=(RejectionReason.INVALID_QUAD,),
                candidates=prediction.candidates,
                diagnostics=_prediction_diagnostics(prediction),
            )
        if not quadrilateral_is_valid(coarse, image_rgb.shape):
            return LocalizationDecision(
                status=LocalizationStatus.REJECTED,
                proposed_corners=coarse,
                coarse_corners=coarse,
                outer_corners=prediction.outer_quad,
                target_class=prediction.target_class,
                layer=TargetLayer.CONTENT,
                confidence=0.0,
                aspect=None,
                backend=prediction.backend,
                rejection_reasons=(RejectionReason.INVALID_QUAD,),
                candidates=prediction.candidates,
                diagnostics=_prediction_diagnostics(prediction),
            )
        refinement = refine_quad_edges(
            image_rgb,
            coarse,
            prediction.boundary_map,
            self.refine_parameters,
        )
        corners = refinement.corners
        confidence, rejection_reasons, components = self.policy.assess(
            prediction,
            refinement,
            image_rgb.shape,
        )
        aspect = estimate_aspect(corners, image_rgb.shape)
        diagnostics: dict[str, object] = {
            **{key: round(value, 6) for key, value in components.items()},
            "candidate_count": len(prediction.candidates),
            "refinement_accepted": refinement.accepted,
            "refinement_reason": refinement.reason,
            "edge_support": [round(value, 6) for value in refinement.edge_support],
            "corner_shifts_px": [round(value, 4) for value in refinement.corner_shifts],
        }
        if rejection_reasons:
            return LocalizationDecision(
                status=LocalizationStatus.REJECTED,
                proposed_corners=corners,
                coarse_corners=coarse,
                outer_corners=prediction.outer_quad,
                target_class=prediction.target_class,
                layer=TargetLayer.CONTENT,
                confidence=confidence,
                aspect=aspect,
                backend=prediction.backend,
                rejection_reasons=rejection_reasons,
                candidates=prediction.candidates,
                diagnostics=diagnostics,
            )
        return LocalizationDecision(
            status=LocalizationStatus.ACCEPTED,
            proposed_corners=corners,
            coarse_corners=coarse,
            outer_corners=prediction.outer_quad,
            target_class=prediction.target_class,
            layer=TargetLayer.CONTENT,
            confidence=confidence,
            aspect=aspect,
            backend=prediction.backend,
            candidates=prediction.candidates,
            diagnostics=diagnostics,
        )


def _prediction_diagnostics(prediction: QuadPrediction) -> dict[str, object]:
    """即使没有可用四角，也保留各 head 的有限诊断供拒绝预览与数据回流。"""

    return {
        "presence_confidence": round(float(prediction.presence_confidence), 6),
        "outer_presence_confidence": round(float(prediction.outer_presence_confidence), 6),
        "class_confidence": round(float(prediction.class_confidence), 6),
        "minimum_corner_confidence": round(float(min(prediction.corner_confidences)), 6),
        "layer_confidence": round(float(prediction.layer_confidence), 6),
        "candidate_count": len(prediction.candidates),
    }
