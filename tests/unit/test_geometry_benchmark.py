"""e2e_auto 几何评分与产品 gate 测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.geometry import (
    AspectEstimate,
    LocalizationDecision,
    LocalizationStatus,
    RejectionReason,
    TargetClass,
    TargetLayer,
)
from screenrestore.validation import (
    GeometryGate,
    GeometryGroundTruth,
    aggregate_geometry_results,
    evaluate_geometry_decision,
)


def _decision(corners: np.ndarray, *, accepted: bool = True) -> LocalizationDecision:
    return LocalizationDecision(
        status=LocalizationStatus.ACCEPTED if accepted else LocalizationStatus.REJECTED,
        proposed_corners=corners,
        coarse_corners=corners,
        outer_corners=None,
        target_class=TargetClass.ARTWORK,
        layer=TargetLayer.CONTENT,
        confidence=0.95,
        aspect=AspectEstimate(1.4, 0.8, "test"),
        backend="test",
        rejection_reasons=() if accepted else (RejectionReason.OUT_OF_SCOPE,),
    )


def test_perfect_decision_passes_smoke_gate() -> None:
    corners = np.array([[20, 30], [220, 30], [220, 170], [20, 170]], np.float32)
    truth = GeometryGroundTruth(corners, TargetClass.ARTWORK)
    metrics = evaluate_geometry_decision(_decision(corners), truth)

    summary = aggregate_geometry_results([metrics], GeometryGate(minimum_samples=1))

    assert metrics["correct"]
    assert metrics["corner_nce"] == 0.0
    assert metrics["quad_iou"] == 1.0
    assert summary["status"] == "PASS"


def test_release_gate_rejects_tiny_smoke_sample_even_when_perfect() -> None:
    corners = np.array([[20, 30], [220, 30], [220, 170], [20, 170]], np.float32)
    truth = GeometryGroundTruth(corners, TargetClass.ARTWORK)
    metrics = evaluate_geometry_decision(_decision(corners), truth)

    summary = aggregate_geometry_results([metrics])

    assert summary["status"] == "FAIL"
    assert not summary["gates"]["minimum_samples"]
