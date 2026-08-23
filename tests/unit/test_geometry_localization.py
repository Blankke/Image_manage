"""自动定位拒绝策略与原分辨率边缘精修数值测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.geometry import (
    AutomaticGeometryService,
    LocalizationStatus,
    QuadPrediction,
    RejectionReason,
    TargetClass,
    refine_quad_edges,
)


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


def test_classic_detector_never_claims_content_layer_for_unattended_acceptance() -> None:
    image, _expected, _coarse = _sharp_target()

    decision = AutomaticGeometryService().localize(image, TargetClass.ARTWORK)

    assert decision.status == LocalizationStatus.REJECTED
    assert RejectionReason.LAYER_AMBIGUOUS in decision.rejection_reasons
    assert all(candidate.layer.value == "unknown" for candidate in decision.candidates)
