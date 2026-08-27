"""e2e_auto 几何评分与产品 gate 测试。"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_release_gate_counts_independent_groups_instead_of_burst_frames() -> None:
    corners = np.array([[20, 30], [220, 30], [220, 170], [20, 170]], np.float32)
    truth = GeometryGroundTruth(corners, TargetClass.ARTWORK)
    results = [evaluate_geometry_decision(_decision(corners), truth) for _ in range(2)]

    summary = aggregate_geometry_results(results, GeometryGate(minimum_samples=2), ["burst-a", "burst-a"])

    assert summary["status"] == "FAIL"
    assert summary["independent_group_count"] == 1
    assert not summary["gates"]["minimum_samples"]


def test_none_hard_negative_is_correct_only_when_automatic_path_rejects() -> None:
    corners = np.array([[20, 30], [220, 30], [220, 170], [20, 170]], np.float32)
    rejected = _decision(corners, accepted=False)

    metrics = evaluate_geometry_decision(rejected, GeometryGroundTruth(None, TargetClass.NONE))

    assert metrics["correct"]
    assert not metrics["in_scope"]


def test_manifest_truth_uses_normalized_content_quad_and_visibility_scope() -> None:
    from benchmarks.geometry_e2e.run import _truth_from_manifest_record

    truth = _truth_from_manifest_record(
        {
            "image": "frames/example.jpg",
            "split": "test",
            "present": True,
            "target_class": "postcard",
            "group_id": "smartdoc-example",
            "visible": False,
            "content_quad": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
        },
        (101, 201, 3),
    )

    assert truth.target_class == TargetClass.POSTCARD
    assert not truth.in_scope
    assert np.allclose(truth.content_quad, [[20, 20], [180, 20], [180, 80], [20, 80]])


def test_manifest_image_is_resolved_from_explicit_dataset_root(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from benchmarks.geometry_e2e.run import _resolve_manifest_image

    root = tmp_path / "screenrestore-data"
    resolved = _resolve_manifest_image(root, "geometry/smartdoc/frames/example.jpg")

    assert resolved == (root / "geometry/smartdoc/frames/example.jpg").resolve()
    with pytest.raises(ValueError, match="不能越出"):
        _resolve_manifest_image(root, "../private/photo.jpg")
