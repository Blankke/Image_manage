"""P4-G3.6 acceptance audit 的数据隔离与 policy 统计契约。"""

from __future__ import annotations

import json

import numpy as np


def _row(group_id: str, correct: bool, margin: float, in_scope: bool = True) -> dict:
    return {
        "group_id": group_id,
        "strict_correct": correct,
        "candidate_margin": margin,
        "in_scope": in_scope,
        "complete_features": True,
    }


def test_group_partition_never_splits_one_group() -> None:
    from scripts.audit_p4_acceptance import _partition_rows

    rows = [_row(f"group-{index}", index % 3 == 0, index / 20) for index in range(30)]
    rows.extend([_row("group-0", True, 0.9), _row("group-1", False, 0.1)])

    partitioned = _partition_rows(rows)
    seen = {}
    for partition, items in partitioned.items():
        for item in items:
            previous = seen.setdefault(item["group_id"], partition)
            assert previous == partition
    assert all(partitioned.values())


def test_threshold_selection_uses_precision_floor_before_coverage() -> None:
    from scripts.audit_p4_acceptance import _select_threshold

    probabilities = np.asarray([0.95, 0.90, 0.80, 0.70])
    labels = np.asarray([True, True, False, True])

    threshold = _select_threshold(probabilities, labels, minimum_precision=0.99)

    assert threshold == 0.90


def test_threshold_selection_keeps_zero_coverage_when_precision_floor_is_impossible() -> None:
    from scripts.audit_p4_acceptance import _select_threshold

    probabilities = np.asarray([0.95, 0.90])
    labels = np.asarray([False, False])

    threshold = _select_threshold(probabilities, labels, minimum_precision=0.99)

    assert threshold > float(probabilities.max())


def test_threshold_selection_applies_structural_eligibility() -> None:
    from scripts.audit_p4_acceptance import _select_threshold

    probabilities = np.asarray([0.99, 0.90, 0.80])
    labels = np.asarray([False, True, True])
    eligible = np.asarray([False, True, True])

    threshold = _select_threshold(probabilities, labels, minimum_precision=0.99, eligible=eligible)

    assert threshold == 0.80


def test_validation_truth_loader_does_not_parse_test_ground_truth(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.audit_p4_acceptance import _truth_by_path

    (tmp_path / "validation.jpg").touch()
    validation = {
        "image": "validation.jpg",
        "split": "validation",
        "present": False,
        "target_class": "none",
        "content_quad": None,
        "group_id": "validation-group",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(validation) + '\n{"split":"test","content_quad": INVALID TEST GT\n',
        encoding="utf-8",
    )

    truths = _truth_by_path(manifest, tmp_path, "validation")

    assert list(truths) == [(tmp_path / "validation.jpg").resolve()]


def test_calibrated_policy_keeps_incomplete_candidates_in_shared_evaluation_denominator() -> None:
    from scripts.audit_p4_acceptance import _fit_and_evaluate_calibrators

    from screenrestore.geometry.confidence import CORRECTNESS_FEATURE_NAMES

    def row(correct: bool, complete: bool) -> dict:
        return {
            "features": {name: float(correct) for name in CORRECTNESS_FEATURE_NAMES},
            "strict_correct": correct,
            "complete_features": complete,
            "has_candidate": complete,
            "area_in_scope": complete,
            "in_scope": True,
            "hard_accepted": False,
            "hard_without_boundary_accepted": False,
            "hard_without_margin_accepted": False,
            "hard_without_margin_boundary_accepted": False,
        }

    partitioned = {
        "fit": [row(True, True), row(False, True)],
        "selection": [row(True, True), row(False, False)],
        "evaluation": [row(True, True), row(False, False)],
    }

    result = _fit_and_evaluate_calibrators(partitioned, "manifest-sha")

    assert result["full"]["selection"]["sample_count"] == 2
    assert result["full"]["evaluation"]["sample_count"] == 2
    assert result["hard_gate_evaluation"]["sample_count"] == 2


def test_stage_nce_uses_target_diagonal_in_pixel_coordinates() -> None:
    from scripts.audit_p4_acceptance import _stage_metrics

    from screenrestore.validation.geometry_benchmark import corner_metrics

    scale = np.asarray([999.0, 799.0], np.float32)
    truth = np.asarray([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
    detected_pixels = truth * scale + np.asarray([40.0, 0.0], np.float32)

    observed = _stage_metrics(detected_pixels, truth, scale)
    expected_nce, expected_iou, _ = corner_metrics(detected_pixels, truth * scale)

    assert observed["corner_nce"] == expected_nce
    assert observed["quad_iou"] == expected_iou
    assert observed["corner_nce"] > 0.04


def test_candidate_margin_metrics_report_fp_fn_at_fixed_threshold() -> None:
    from scripts.audit_p4_acceptance import _threshold_metrics

    rows = [
        _row("a", True, 0.08),
        _row("b", False, 0.07),
        _row("c", True, 0.05),
        _row("d", False, 0.01, in_scope=False),
    ]

    metrics = _threshold_metrics(rows, 0.06)

    assert metrics["accepted_count"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
