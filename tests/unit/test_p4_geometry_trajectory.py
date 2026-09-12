"""P4-G3.6 validation-only checkpoint selection 契约。"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _metrics(nce_median: float, nce_p95: float, iou_median: float, iou_p05: float) -> dict:
    return {
        "corner_nce_median": nce_median,
        "corner_nce_p95": nce_p95,
        "quad_iou_median": iou_median,
        "quad_iou_p05": iou_p05,
    }


def _reports() -> dict:
    from scripts.evaluate_p4_geometry_trajectory import DATASET_ORDER

    output = {}
    for name in DATASET_ORDER:
        output[name] = {
            "checkpoints": [
                {"label": "B0", "metrics": _metrics(0.10, 0.30, 0.70, 0.20)},
                # epoch 1 同时守住 median，并显著改善 tail。
                {"label": "epoch-001", "metrics": _metrics(0.101, 0.25, 0.695, 0.30)},
                # epoch 2 的 tail 更好，但 median 超过预先容差，必须先被淘汰。
                {"label": "epoch-002", "metrics": _metrics(0.12, 0.20, 0.66, 0.40)},
            ]
        }
    return output


def test_selection_applies_b0_eligibility_before_tail_priority() -> None:
    from scripts.evaluate_p4_geometry_trajectory import select_checkpoint

    result = select_checkpoint(
        _reports(),
        {
            "corner_nce_median": 0.002,
            "corner_nce_p95": 0.005,
            "quad_iou_median": 0.01,
            "quad_iou_p05": 0.01,
        },
    )

    assert result["eligible_checkpoints"] == ["epoch-001"]
    assert result["winner"] == "epoch-001"
    assert result["status"] == "FROZEN"


def test_selection_refuses_to_invent_winner_when_all_regress() -> None:
    from scripts.evaluate_p4_geometry_trajectory import DATASET_ORDER, select_checkpoint

    reports = _reports()
    for dataset in DATASET_ORDER:
        reports[dataset]["checkpoints"] = [
            reports[dataset]["checkpoints"][0],
            reports[dataset]["checkpoints"][2],
        ]

    result = select_checkpoint(
        reports,
        {
            "corner_nce_median": 0.002,
            "corner_nce_p95": 0.005,
            "quad_iou_median": 0.01,
            "quad_iou_p05": 0.01,
        },
    )

    assert result["eligible_checkpoints"] == []
    assert result["winner"] is None
    assert result["status"] == "NO_ELIGIBLE_CHECKPOINT"


def test_ineligible_checkpoint_cannot_remove_eligible_pareto_candidate() -> None:
    from scripts.evaluate_p4_geometry_trajectory import DATASET_ORDER, select_checkpoint

    reports = {}
    for name in DATASET_ORDER:
        reports[name] = {
            "checkpoints": [
                {"label": "B0", "metrics": _metrics(0.10, 0.20, 0.80, 0.60)},
                {"label": "epoch-001", "metrics": _metrics(0.10, 0.19, 0.80, 0.61)},
                # median NCE 不合格，但其它三项足以构成偏科的非支配点。
                {"label": "epoch-002", "metrics": _metrics(0.50, 0.10, 0.90, 0.70)},
            ]
        }

    result = select_checkpoint(
        reports,
        {
            "corner_nce_median": 0.0,
            "corner_nce_p95": 0.0,
            "quad_iou_median": 0.0,
            "quad_iou_p05": 0.0,
        },
    )

    assert result["eligible_checkpoints"] == ["epoch-001"]
    assert result["eligible_pareto_front"] == ["epoch-001"]
    assert result["winner"] == "epoch-001"


def test_trajectory_dataset_spec_never_schedules_test_split(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.evaluate_p4_geometry_trajectory import _dataset_spec

    (tmp_path / "validation.jpg").touch()
    (tmp_path / "test.jpg").touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "image": image,
                    "split": split,
                    "present": False,
                    "target_class": "none",
                    "group_id": split,
                }
            )
            + "\n"
            for image, split in (("validation.jpg", "validation"), ("test.jpg", "test"))
        ),
        encoding="utf-8",
    )
    # test 行只需要能投影 split；validation 调度不得继续读取其 image 或标签。
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write('{"split":"test","image": INVALID TEST PAYLOAD\n')

    spec = _dataset_spec("validation", manifest, tmp_path)

    assert spec.paths == ((tmp_path / "validation.jpg").resolve(),)


def test_trajectory_nce_uses_pixel_target_diagonal() -> None:
    from scripts.evaluate_p4_geometry_trajectory import _score_prediction

    scale = np.asarray([999.0, 799.0], np.float32)
    truth_quad = np.asarray([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
    corners = [
        {
            "peak1": 0.9,
            "peak2": 0.1,
            "peak_difference": 0.8,
            "peak_ratio": 9.0,
            "normalized_entropy": 0.1,
            "local_sharpness": 0.8,
        }
        for _ in range(4)
    ]
    prediction = {
        "content_quad": truth_quad * scale + np.asarray([40.0, 0.0], np.float32),
        "image_scale": scale,
        "target_class": "artwork",
        "corner_diagnostics": corners,
        "corner_confidences": [0.9] * 4,
        "candidate_margin": 0.8,
    }
    truth = {
        "present": True,
        "target_class": "artwork",
        "content_quad": truth_quad,
    }

    result = _score_prediction(prediction, truth)

    assert result["corner_nce"] > 0.04


def test_trajectory_forces_one_evaluation_image_size() -> None:
    from scripts.evaluate_p4_geometry_trajectory import _resolve_evaluation_image_size

    assert _resolve_evaluation_image_size(0, baseline_image_size=512) == 512
    assert _resolve_evaluation_image_size(256, baseline_image_size=512) == 256


def test_trajectory_rejects_invalid_evaluation_image_size() -> None:
    from scripts.evaluate_p4_geometry_trajectory import _resolve_evaluation_image_size

    with pytest.raises(ValueError, match="evaluation-image-size"):
        _resolve_evaluation_image_size(250, baseline_image_size=512)
