"""DOCCI 桌面合成的许可身份、分组和真实像素边界回归。

运行范例：source .venv/bin/activate && which python && python -m pytest tests/unit/test_prepare_docci_desk_geometry.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest
from scripts.prepare_docci_desk_geometry import _compose_paper, _sample_quad, main


def test_paper_quad_matches_composited_pixels() -> None:
    background = np.full((800, 640, 3), 100, np.uint8)
    cover = np.zeros((900, 620, 3), np.uint8)
    cover[:] = (20, 30, 220)
    quad = _sample_quad(np.random.default_rng(23), 640, 800, distant=False)
    result = _compose_paper(background, cover, quad, np.random.default_rng(24))
    # 四角标签围出的中心应是封面；远离纸面和阴影的角落保留原照片。
    center = np.rint(quad.mean(axis=0)).astype(int)
    assert result[center[1], center[0], 2] > 140
    assert np.array_equal(result[0, 0], background[0, 0])
    assert cv2.isContourConvex(quad)
    assert cv2.contourArea(quad) / (640 * 800) > 0.09


@pytest.mark.parametrize("width,height", [(640, 800), (1024, 768)])
def test_paper_quad_is_complete_in_portrait_and_landscape(width: int, height: int) -> None:
    rng = np.random.default_rng(290)
    for distant in (False, True):
        for _ in range(30):
            quad = _sample_quad(rng, width, height, distant=distant)
            assert np.all((quad[:, 0] >= 0) & (quad[:, 0] < width))
            assert np.all((quad[:, 1] >= 0) & (quad[:, 1] < height))
            assert cv2.isContourConvex(quad)


def test_docci_generation_keeps_source_family_and_valid_schema(tmp_path: Path) -> None:
    root = tmp_path / "public-data"
    cache = root / "backgrounds" / "docci" / "reviewed-desk-thumbnails"
    cache.mkdir(parents=True)
    metadata = cache.parent / "docci_web_data.jsonlines"
    photos = ["train_00001", "train_00002"]
    metadata.write_text(
        "".join(json.dumps({"example_id": ident, "cluster_id": "21"}) + "\n" for ident in photos),
        encoding="utf-8",
    )
    for number, ident in enumerate(photos):
        image = np.full((800, 640, 3), 95 + number * 15, np.uint8)
        assert cv2.imwrite(str(cache / f"{ident}.jpg"), image)
    index = tmp_path / "reviewed.json"
    index.write_text(json.dumps({
        "dataset": "DOCCI", "source_split": "train", "source_license": "CC BY 4.0",
        "source_page": "https://google.github.io/docci/",
        "photos": [{"example_id": ident, "cluster_id": "21"} for ident in photos],
    }), encoding="utf-8")
    artwork_root = root / "geometry" / "syngallery"
    (artwork_root / "sources").mkdir(parents=True)
    (artwork_root / "images").mkdir()
    artwork_manifest = root / "manifests" / "artwork.geometry.jsonl"
    artwork_manifest.parent.mkdir()
    asset_rows = []
    for number, split in ((1, "train"), (2, "validation")):
        source = artwork_root / "sources" / f"met-{number}.jpg"
        assert cv2.imwrite(str(source), np.full((800, 640, 3), 170, np.uint8))
        asset_rows.append({
            "image": f"geometry/syngallery/images/{number}.jpg",
            "source": "syngallery-reviewed",
            "digital_source_id": f"met-open-access-{number}",
            "split": split,
        })
    artwork_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in asset_rows), encoding="utf-8"
    )
    output = root / "geometry" / "docci-pilot"
    assert main([
        "--data-root", str(root), "--background-index", str(index),
        "--artwork-manifest", str(artwork_manifest),
        "--output-directory", str(output), "--samples-per-photo", "2",
    ]) == 0
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    schema = json.loads(Path("datasets/schemas/geometry.schema.json").read_text())
    assert len(rows) == 4
    for row in rows:
        jsonschema.validate(row, schema)
        assert row["split"] == "train"
        assert row["group_id"] == "docci-photo-family"
        assert row["source_capture_id"] in photos
        assert row["digital_source_id"] == "met-open-access-1"
        quad = np.asarray(row["content_quad"])
        assert np.all((quad >= 0) & (quad <= 1))
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["capture_group_count"] == 1
    assert provenance["metadata_category_cluster_count"] == 1
    assert set(provenance["background_photo_sha256"]) == set(photos)


def test_docci_index_rejects_incorrect_cluster(tmp_path: Path) -> None:
    from scripts.prepare_docci_desk_geometry import _verify_clusters

    metadata = tmp_path / "metadata.jsonlines"
    metadata.write_text('{"example_id":"train_00001","cluster_id":"21"}\n')
    with pytest.raises(ValueError, match="照片簇元数据不一致"):
        _verify_clusters([{"example_id": "train_00001", "cluster_id": "99"}], metadata)
