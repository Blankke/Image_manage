"""自动定位拒绝策略与原分辨率边缘精修数值测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.geometry import (
    AutomaticGeometryService,
    EdgeRefinement,
    LocalizationStatus,
    ModelAgreementQuadDetector,
    QuadPrediction,
    QuadrilateralCandidate,
    RejectionReason,
    TargetClass,
    TargetLayer,
    refine_quad_edges,
)


def test_model_agreement_detector_snaps_to_supported_classic_edges(monkeypatch) -> None:
    import screenrestore.geometry.detector as detector_module

    image = np.zeros((120, 140, 3), np.uint8)
    model_quad = np.asarray([[20, 20], [120, 22], [118, 100], [22, 98]], np.float32)
    edge_quad = np.asarray([[18, 18], [122, 20], [121, 102], [19, 100]], np.float32)
    prediction = QuadPrediction(
        content_quad=model_quad,
        corner_confidences=(0.9, 0.91, 0.92, 0.93),
        presence_confidence=0.95,
        target_class=TargetClass.ARTWORK,
        class_confidence=0.9,
        candidates=(
            QuadrilateralCandidate(
                model_quad,
                0.915,
                {"candidate_margin": 0.2},
                "quadlocator_onnx",
                TargetLayer.CONTENT,
            ),
        ),
        backend="quadlocator_onnx",
    )
    classic = QuadrilateralCandidate(
        edge_quad,
        0.6,
        {},
        "classic_contour",
        TargetLayer.UNKNOWN,
    )
    monkeypatch.setattr(detector_module, "detect_classic_candidates", lambda *_a, **_k: [classic])

    snapped = ModelAgreementQuadDetector(
        _FixedDetector(prediction),
        minimum_agreement_iou=0.8,
    ).predict(image)

    assert np.allclose(snapped.content_quad, edge_quad)
    assert snapped.target_class == TargetClass.ARTWORK
    assert snapped.presence_confidence == prediction.presence_confidence
    assert snapped.candidates[0].source == "quadlocator_classic_agreement"
    assert snapped.decoder_diagnostics["model_agreement_snap"]["status"] == "selected"


def test_model_agreement_detector_keeps_model_quad_when_candidate_disagrees(monkeypatch) -> None:
    import screenrestore.geometry.detector as detector_module

    image = np.zeros((120, 140, 3), np.uint8)
    model_quad = np.asarray([[15, 15], [65, 15], [65, 65], [15, 65]], np.float32)
    prediction = QuadPrediction(content_quad=model_quad, backend="quadlocator_onnx")
    far = QuadrilateralCandidate(
        np.asarray([[75, 70], [130, 70], [130, 115], [75, 115]], np.float32),
        0.8,
        {},
        "classic_contour",
        TargetLayer.UNKNOWN,
    )
    monkeypatch.setattr(detector_module, "detect_classic_candidates", lambda *_a, **_k: [far])

    unchanged = ModelAgreementQuadDetector(_FixedDetector(prediction)).predict(image)

    assert np.allclose(unchanged.content_quad, model_quad)
    assert unchanged.backend == "quadlocator_onnx"
    assert unchanged.decoder_diagnostics["model_agreement_snap"]["status"] == "outside_limits"


def test_model_agreement_runtime_gate_requires_failed_model_refine_and_better_evidence(
    monkeypatch,
) -> None:
    import screenrestore.geometry.localizer as localizer_module

    image = np.zeros((120, 140, 3), np.uint8)
    model_quad = np.asarray([[22, 22], [118, 22], [118, 98], [22, 98]], np.float32)
    agreement_quad = np.asarray([[18, 18], [122, 18], [122, 102], [18, 102]], np.float32)
    prediction = QuadPrediction(
        content_quad=agreement_quad,
        candidates=(
            QuadrilateralCandidate(
                agreement_quad, 0.9, {}, "quadlocator_classic_agreement", TargetLayer.CONTENT
            ),
            QuadrilateralCandidate(
                model_quad, 0.9, {}, "quadlocator_onnx", TargetLayer.CONTENT
            ),
        ),
        decoder_diagnostics={"model_agreement_snap": {"status": "selected"}},
        backend="quadlocator_onnx+classic_agreement",
    )
    agreement_refinement = EdgeRefinement(
        agreement_quad,
        True,
        (0.8, 0.8, 0.8, 0.8),
        (1.0, 1.0, 1.0, 1.0),
    )
    model_refinement = EdgeRefinement(
        model_quad,
        False,
        (0.5, 0.5, 0.5, 0.5),
        (0.0, 0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(localizer_module, "refine_quad_edges", lambda *_a, **_k: model_refinement)
    monkeypatch.setattr(
        localizer_module,
        "mask_quad_consistency",
        lambda item, _shape: 0.9 if np.allclose(item.content_quad, agreement_quad) else 0.8,
    )

    selected, coarse, refinement, assessment, assessment_refinement = (
        localizer_module._select_model_agreement_branch(
            image,
            prediction,
            agreement_quad,
            agreement_refinement,
            localizer_module.EdgeRefineParameters(),
        )
    )

    assert np.allclose(coarse, agreement_quad)
    assert refinement is agreement_refinement
    assert np.allclose(assessment.content_quad, model_quad)
    assert assessment_refinement is model_refinement
    assert selected.decoder_diagnostics["model_agreement_snap"]["runtime_selection"] == "classic_agreement"

    # 模型原分支一旦也能通过精修，保守门必须保留模型语义四角。
    accepted_model = EdgeRefinement(
        model_quad,
        True,
        (0.7, 0.7, 0.7, 0.7),
        (1.0, 1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(localizer_module, "refine_quad_edges", lambda *_a, **_k: accepted_model)
    selected, coarse, refinement, assessment, assessment_refinement = (
        localizer_module._select_model_agreement_branch(
            image,
            prediction,
            agreement_quad,
            agreement_refinement,
            localizer_module.EdgeRefineParameters(),
        )
    )
    assert np.allclose(coarse, model_quad)
    assert refinement is accepted_model
    assert assessment is not prediction
    assert assessment_refinement is accepted_model
    assert selected.decoder_diagnostics["model_agreement_snap"]["runtime_selection"] == "model"


class _FixedDetector:
    def __init__(self, prediction: QuadPrediction) -> None:
        self.prediction = prediction

    def predict(
        self,
        image_rgb: np.ndarray,
        target_hint: TargetClass | None = None,
    ) -> QuadPrediction:
        _ = image_rgb, target_hint
        return self.prediction


def _sharp_target() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((460, 640, 3), 24, dtype=np.uint8)
    expected = np.array([[80, 60], [560, 60], [560, 390], [80, 390]], np.float32)
    cv2.fillConvexPoly(image, expected.astype(np.int32), (172, 188, 196))
    cv2.rectangle(image, (80, 60), (560, 390), (235, 235, 235), 3)
    # 内容纹理用于验证稳健拟合不会吸到内部直线上。
    for x in range(120, 540, 48):
        cv2.line(image, (x, 92), (x + 12, 350), (70, 110, 155), 2)
    coarse = np.array([[89, 68], [551, 69], [552, 381], [88, 380]], np.float32)
    return image, expected, coarse


def test_full_resolution_refinement_reduces_corner_error() -> None:
    image, expected, coarse = _sharp_target()
    source_copy = image.copy()
    before = float(np.mean(np.linalg.norm(coarse - expected, axis=1)))

    result = refine_quad_edges(image, coarse)

    after = float(np.mean(np.linalg.norm(result.corners - expected, axis=1)))
    assert result.accepted
    assert after < 4.5
    assert after < before * 0.4
    assert min(result.edge_support) > 0.16
    assert np.array_equal(image, source_copy)


def test_edge_trace_prefers_continuous_boundary_over_alternating_texture_peaks() -> None:
    from screenrestore.geometry.edge_refine import _straight_edge_peak_indices

    responses = np.full((12, 9), 0.02, np.float64)
    responses[:, 4] = 0.8
    for row in range(len(responses)):
        responses[row, 0 if row % 2 == 0 else 8] = 1.0

    selected = _straight_edge_peak_indices(responses, np.arange(-4, 5), band=4)

    assert np.array_equal(selected, np.full(12, 4))


def test_service_accepts_content_layer_and_keeps_outer_layer_separate() -> None:
    image, expected, coarse = _sharp_target()
    outer = np.array([[55, 35], [585, 35], [585, 415], [55, 415]], np.float32)
    prediction = QuadPrediction(
        content_quad=coarse,
        outer_quad=outer,
        corner_confidences=(0.95, 0.94, 0.96, 0.95),
        presence_confidence=0.97,
        target_class=TargetClass.ARTWORK,
        class_confidence=0.93,
        layer_confidence=0.91,
        backend="test_quadlocator",
    )

    decision = AutomaticGeometryService(_FixedDetector(prediction)).localize(
        image,
        TargetClass.ARTWORK,
    )

    assert decision.status == LocalizationStatus.ACCEPTED
    assert decision.proposed_corners is not None
    assert float(np.mean(np.linalg.norm(decision.proposed_corners - expected, axis=1))) < 4.5
    assert np.allclose(decision.outer_corners, outer)
    assert decision.layer.value == "content"


def test_service_rejects_low_confidence_instead_of_forcing_quad() -> None:
    image, _expected, coarse = _sharp_target()
    prediction = QuadPrediction(
        content_quad=coarse,
        corner_confidences=(0.91, 0.24, 0.88, 0.90),
        presence_confidence=0.92,
        target_class=TargetClass.ARTWORK,
        class_confidence=0.41,
        layer_confidence=0.42,
        backend="test_quadlocator",
    )

    decision = AutomaticGeometryService(_FixedDetector(prediction)).localize(image)

    assert decision.status == LocalizationStatus.REJECTED
    assert RejectionReason.CORNER_UNCERTAIN in decision.rejection_reasons
    assert RejectionReason.TARGET_CLASS_UNCERTAIN in decision.rejection_reasons
    assert RejectionReason.LAYER_AMBIGUOUS in decision.rejection_reasons
    assert decision.proposed_corners is not None


def test_service_rejects_outer_prediction_that_does_not_contain_content() -> None:
    from screenrestore.geometry.detector import _layer_confidence

    image, _expected, coarse = _sharp_target()
    # outer 头错落在 content 内部时，不能根据面积或矩形度继续自动接受。
    invalid_outer = np.array([[250, 160], [390, 160], [390, 280], [250, 280]], np.float32)
    mask = np.ones((32, 32), np.float32)
    assert _layer_confidence(coarse, invalid_outer, mask) < 0.58
    prediction = QuadPrediction(
        content_quad=coarse,
        outer_quad=invalid_outer,
        corner_confidences=(0.95, 0.95, 0.95, 0.95),
        presence_confidence=0.97,
        target_class=TargetClass.ARTWORK,
        class_confidence=0.94,
        layer_confidence=0.0,
        backend="test_quadlocator",
    )

    decision = AutomaticGeometryService(_FixedDetector(prediction)).localize(image)

    assert decision.status == LocalizationStatus.REJECTED
    assert RejectionReason.LAYER_AMBIGUOUS in decision.rejection_reasons


def test_classic_detector_never_claims_content_layer_for_unattended_acceptance() -> None:
    image, _expected, _coarse = _sharp_target()

    decision = AutomaticGeometryService().localize(image, TargetClass.ARTWORK)

    assert decision.status == LocalizationStatus.REJECTED
    assert RejectionReason.LAYER_AMBIGUOUS in decision.rejection_reasons
    assert all(candidate.layer.value == "unknown" for candidate in decision.candidates)
