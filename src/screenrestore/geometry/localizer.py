"""统一编排语义四边形、原图边缘精修和拒绝策略。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .confidence import ConfidencePolicy, mask_quad_consistency
from .detector import ClassicQuadDetector, QuadDetector
from .edge_refine import EdgeRefineParameters, refine_quad_edges
from .rectify import estimate_aspect, order_corners
from .types import (
    EdgeRefinement,
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
        prediction, coarse, refinement, assessment_prediction, assessment_refinement = (
            _select_model_agreement_branch(
                image_rgb,
                prediction,
                coarse,
                refinement,
                self.refine_parameters,
            )
        )
        corners = refinement.corners
        confidence, rejection_reasons, components = self.policy.assess(
            assessment_prediction,
            assessment_refinement,
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
            "refinement_outcome": refinement.outcome,
            "refinement_residual_median_px": [round(value, 4) for value in refinement.residual_median],
            "refinement_residual_p95_px": [round(value, 4) for value in refinement.residual_p95],
            "refinement_continuous_coverage": [round(value, 6) for value in refinement.continuous_coverage],
            "refinement_gradient_normal_alignment": [round(value, 6) for value in refinement.gradient_normal_alignment],
            "refinement_boundary_consistency": [round(value, 6) for value in refinement.boundary_consistency],
            "refinement_area_drift": round(refinement.area_drift, 6),
            "refinement_aspect_drift": round(refinement.aspect_drift, 6),
            "decoder": prediction.decoder_diagnostics,
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


def _select_model_agreement_branch(
    image_rgb: np.ndarray,
    prediction: QuadPrediction,
    agreement_coarse: np.ndarray,
    agreement_refinement: EdgeRefinement,
    parameters: EdgeRefineParameters,
) -> tuple[
    QuadPrediction,
    np.ndarray,
    EdgeRefinement,
    QuadPrediction,
    EdgeRefinement,
]:
    """比较模型原四角与传统候选分支，只采纳有多项运行时证据的吸附。

    该门由公开 validation 冻结：传统候选只有在模型原分支精修失败、候选分支精修
    成功，且候选同时提高边界支持和模型 mask 一致性时才会生效。其余情况保留模型
    原四角，防止传统轮廓凭矩形度覆盖内容层语义。
    """

    snap = prediction.decoder_diagnostics.get("model_agreement_snap")
    if not isinstance(snap, dict) or snap.get("status") != "selected":
        return (
            prediction,
            agreement_coarse,
            agreement_refinement,
            prediction,
            agreement_refinement,
        )
    original = next(
        (
            item
            for item in prediction.candidates
            if item.layer == TargetLayer.CONTENT
            and item.source != "quadlocator_classic_agreement"
        ),
        None,
    )
    if original is None:
        return (
            prediction,
            agreement_coarse,
            agreement_refinement,
            prediction,
            agreement_refinement,
        )
    try:
        model_coarse = order_corners(original.corners)
    except ValueError:
        return (
            prediction,
            agreement_coarse,
            agreement_refinement,
            prediction,
            agreement_refinement,
        )
    if not quadrilateral_is_valid(model_coarse, image_rgb.shape):
        return (
            prediction,
            agreement_coarse,
            agreement_refinement,
            prediction,
            agreement_refinement,
        )
    model_prediction = replace(prediction, content_quad=model_coarse)
    model_refinement = refine_quad_edges(
        image_rgb,
        model_coarse,
        prediction.boundary_map,
        parameters,
    )
    agreement_mask = mask_quad_consistency(prediction, image_rgb.shape)
    model_mask = mask_quad_consistency(model_prediction, image_rgb.shape)
    use_agreement = bool(
        agreement_refinement.accepted
        and not model_refinement.accepted
        and agreement_refinement.mean_support > model_refinement.mean_support + 1e-4
        and agreement_mask > model_mask + 1e-4
    )
    decoder = dict(prediction.decoder_diagnostics)
    enriched = dict(snap)
    enriched.update(
        {
            "runtime_selection": "classic_agreement" if use_agreement else "model",
            "model_refinement_accepted": bool(model_refinement.accepted),
            "agreement_refinement_accepted": bool(agreement_refinement.accepted),
            "model_boundary_support": round(float(model_refinement.mean_support), 8),
            "agreement_boundary_support": round(
                float(agreement_refinement.mean_support), 8
            ),
            "model_mask_consistency": round(model_mask, 8),
            "agreement_mask_consistency": round(agreement_mask, 8),
        }
    )
    decoder["model_agreement_snap"] = enriched
    if use_agreement:
        # 传统候选只改进几何落点。自动接受仍沿用模型原分支证据，防止候选吸附
        # 把原本应拒绝的样本推过高置信度门。
        return (
            replace(prediction, decoder_diagnostics=decoder),
            agreement_coarse,
            agreement_refinement,
            model_prediction,
            model_refinement,
        )
    return (
        replace(model_prediction, decoder_diagnostics=decoder),
        model_coarse,
        model_refinement,
        model_prediction,
        model_refinement,
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
