"""私人开发验证集的对象分组、数据绑定与评分契约。

运行：source .venv/bin/activate && which python && \
XDG_STATE_HOME=/tmp/screenrestore-private-test python -m pytest -q \
tests/unit/test_private_development_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scripts.private_development_validation import (
    _candidate_geometry,
    _dataset_index,
    _draft_annotation,
    _json_sha256,
    _load_bound_inputs,
    _review_tile,
    build_parser,
    discover_frames,
    score_rows,
)


def _make_six_images(directory: Path) -> None:
    directory.mkdir(parents=True)
    for number in range(1, 7):
        Image.new("RGB", (80, 60), (number * 20, 30, 40)).save(directory / f"{number}.JPG")


def test_discovery_groups_six_views_as_one_object(tmp_path: Path) -> None:
    images = tmp_path / "private_set"
    _make_six_images(images / "magazine" / "one")
    _make_six_images(images / "screen" / "two")
    rows = discover_frames(images, tmp_path)
    assert len(rows) == 12
    assert len({row["group_id"] for row in rows}) == 2
    assert [row["capture_view"] for row in rows[:6]] == [
        "frontal",
        "left_oblique",
        "right_oblique",
        "forward_oblique",
        "backward_oblique",
        "far",
    ]
    assert {row["target_class"] for row in rows if row["domain"] == "screen"} == {"screen"}


def test_freeze_parser_records_experimental_candidate_gate() -> None:
    args = build_parser().parse_args(
        [
            "freeze",
            "--data-root",
            "/tmp/data",
            "--index",
            "/tmp/index.json",
            "--quad-model",
            "/tmp/model.onnx",
            "--output-directory",
            "/tmp/output",
            "--classic-agreement-snap",
        ]
    )

    assert args.classic_agreement_snap
    assert args.classic_agreement_min_iou == 0.70
    assert args.classic_agreement_max_corner_nce == 0.12


def test_perfect_frozen_prediction_scores_strict_correct(tmp_path: Path) -> None:
    images = tmp_path / "private_set"
    _make_six_images(images / "posters" / "one")
    frame = discover_frames(images, tmp_path)[0]
    truth = {
        **_draft_annotation(frame),
        "annotation_status": "approved",
        "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
    }
    prediction = {
        frame["image_id"]: {
            "decision": {
                "accepted": True,
                "corners": truth["content_quad"],
                "target_class": "artwork",
                "confidence": 0.99,
                "rejection_reasons": [],
            }
        }
    }
    row = score_rows(prediction, [truth])[0]
    assert row["strict_correct"]
    assert row["has_candidate"]
    assert not row["outer_evaluable"]
    assert row["candidate_closer_to_outer"] is None
    assert row["corner_nce"] == 0.0
    assert row["quad_iou"] == pytest.approx(1.0)
    assert _candidate_geometry([row])["strict_geometry_rate"] == 1.0


def test_bound_loader_rejects_prediction_from_other_dataset(tmp_path: Path) -> None:
    images = tmp_path / "private_set"
    _make_six_images(images / "postcards" / "one")
    frames = discover_frames(images, tmp_path)
    index = _dataset_index(frames)
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    annotations = [
        {**_draft_annotation(frame), "annotation_status": "approved"} for frame in frames
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_path.write_text("".join(json.dumps(row) + "\n" for row in annotations))
    payload = {
        "kind": "screenrestore_private_frozen_predictions",
        "dataset_sha256": "wrong",
        "predictions": [],
    }
    payload["prediction_sha256"] = _json_sha256(payload)
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(payload))
    try:
        _load_bound_inputs(index_path, prediction_path, annotation_path)
    except ValueError as error:
        assert "不属于当前数据索引" in str(error)
    else:
        raise AssertionError("必须拒绝其它数据集的冻结预测")


def test_review_tile_draws_comparison_without_mutating_source() -> None:
    image = Image.new("RGB", (80, 60), (120, 130, 140))
    image_array = np.asarray(image).copy()
    before = image_array.copy()
    truth = {
        "capture_number": 1,
        "capture_view": "frontal",
        "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "outer_quad": None,
    }
    decision = {
        "status": "rejected",
        "corners": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
    }

    tile = _review_tile(image_array, truth, decision, tile_width=120, tile_height=90)

    assert tile.size == (120, 132)
    assert (image_array == before).all()
