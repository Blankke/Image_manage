"""SynGallery 同源匹配试验的数值回归。

运行范例：source .venv/bin/activate && pytest -q tests/unit/test_prepare_syngallery_geometry.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scripts.prepare_syngallery_geometry import (
    _download_rgb,
    propose_content_quad,
    source_full_bleed_evidence,
)
from scripts.promote_syngallery_geometry import main as promote_main


def test_known_perspective_source_quad_is_recovered() -> None:
    rng = np.random.default_rng(17)
    source = rng.integers(0, 256, size=(180, 220, 3), dtype=np.uint8)
    expected = np.asarray([[93, 78], [383, 64], [398, 344], [105, 354]], np.float32)
    corners = np.asarray([[0, 0], [219, 0], [219, 179], [0, 179]], np.float32)
    transform = cv2.getPerspectiveTransform(corners, expected)
    render = np.full((512, 512, 3), 200, np.uint8)
    warped = cv2.warpPerspective(source, transform, (512, 512))
    mask = cv2.warpPerspective(np.full((180, 220), 255, np.uint8), transform, (512, 512))
    render[mask > 0] = warped[mask > 0]

    actual, evidence = propose_content_quad(source, render)

    assert actual is not None
    assert evidence["reason"] == "provisional_match"
    assert evidence["inliers"] >= 20
    assert np.max(np.linalg.norm(actual - expected, axis=1)) < 4.0


def test_featureless_source_is_rejected() -> None:
    source = np.full((180, 220, 3), 160, np.uint8)
    render = np.full((512, 512, 3), 160, np.uint8)

    actual, evidence = propose_content_quad(source, render)

    assert actual is None
    assert evidence["reason"] == "insufficient_features"


def test_uniform_catalog_background_is_not_labeled_as_content() -> None:
    source = np.full((220, 260, 3), 165, np.uint8)
    source[45:175, 55:205] = np.random.default_rng(9).integers(
        0, 256, size=(130, 150, 3), dtype=np.uint8
    )

    accepted, evidence = source_full_bleed_evidence(source)

    assert not accepted
    assert evidence["source_corner_texture"] < 2.0


def test_full_bleed_source_passes_corner_diversity_check() -> None:
    source = np.empty((220, 260, 3), np.uint8)
    source[:110, :130] = (30, 70, 115)
    source[:110, 130:] = (165, 60, 45)
    source[110:, :130] = (65, 180, 80)
    source[110:, 130:] = (190, 150, 45)

    accepted, evidence = source_full_bleed_evidence(source)

    assert accepted
    assert evidence["source_corner_texture"] >= 28.0


def test_cached_image_requires_audited_dataset_revision() -> None:
    try:
        _download_rgb("https://datasets-server.huggingface.co/cached-assets/changed/asset.jpg")
    except ValueError as error:
        assert "版本" in str(error)
    else:
        raise AssertionError("不同版本的图像必须在下载前拒绝")


def test_promotion_excludes_met_texture_overlap_and_keeps_group_split(tmp_path) -> None:
    from scripts.prepare_syngallery_geometry import DATASET_REVISION

    metadata = tmp_path / "textures" / "met-open-access" / "metadata.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"object_id": 101}) + "\n", encoding="utf-8")
    pilot = tmp_path / "geometry" / "syngallery-pilot"
    (pilot / "images").mkdir(parents=True)
    (pilot / "overlays").mkdir()
    (pilot / "review-board-01.jpg").touch()
    rows = []
    for object_id in (101, 202):
        name = f"met-{object_id}-view-60.jpg"
        (pilot / "images" / name).touch()
        (pilot / "overlays" / name).touch()
        rows.append(
            {
                "image": f"images/{name}",
                "met_object_id": object_id,
                "view": 60,
                "split": "train",
                "source_revision": DATASET_REVISION,
                "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "matching_evidence": {"reason": "provisional_match", "inliers": 42},
                "source_evidence": {
                    "source_corner_texture": 38.0,
                    "source_corner_color_spread": 45.0,
                },
            }
        )
    (pilot / "candidate-manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "manifests" / "reviewed.geometry.jsonl"

    assert (
        promote_main(
            [
                "--data-root",
                str(tmp_path),
                "--met-metadata",
                str(metadata),
                "--candidate-directory",
                str(pilot),
                "--reviewer",
                "visual-test",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    promoted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(promoted) == 1
    assert promoted[0]["group_id"] == "syngallery-met-202"
    assert promoted[0]["image"] == "geometry/syngallery-pilot/images/met-202-view-60.jpg"
    schema_path = Path(__file__).resolve().parents[2] / "datasets/schemas/geometry.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert set(schema["required"]) <= set(promoted[0]) <= set(schema["properties"])
    review = json.loads(output.with_suffix(".review.json").read_text(encoding="utf-8"))
    assert review["skipped"]["met_texture_overlap"] == 1
