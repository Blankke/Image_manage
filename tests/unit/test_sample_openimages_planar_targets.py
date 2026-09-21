"""Open Images 候选筛选的标注契约回归。

运行范例：source .venv/bin/activate && which python &&
XDG_STATE_HOME=/tmp/screenrestore-openimages-test pytest -q tests/unit/test_sample_openimages_planar_targets.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.sample_openimages_planar_targets import _eligible_boxes, _image_metadata


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_validation_boxes_are_only_candidate_proposals(tmp_path: Path) -> None:
    # 官方验证集物体框没有四个极值点；遮挡目标和未核实方向的照片都应剔除。
    base = {
        "ImageID": "good",
        "Source": "xclick",
        "LabelName": "/m/01n5jq",
        "Confidence": "1",
        "XMin": "0.2",
        "XMax": "0.8",
        "YMin": "0.2",
        "YMax": "0.8",
        "IsOccluded": "0",
        "IsTruncated": "0",
        "IsGroupOf": "0",
        "IsDepiction": "0",
        "IsInside": "0",
    }
    boxes_path = tmp_path / "boxes.csv"
    _write_csv(boxes_path, [base, {**base, "ImageID": "occluded", "IsOccluded": "1"}])
    proposals = _eligible_boxes(boxes_path, "/m/01n5jq")
    assert proposals == {"good": [{"bbox": [0.2, 0.2, 0.8, 0.8]}]}

    image_base = {
        "ImageID": "good",
        "License": "https://creativecommons.org/licenses/by/2.0/",
        "Rotation": "0",
        "OriginalLandingURL": "https://www.flickr.com/photos/example/1",
        "Thumbnail300KURL": "https://live.staticflickr.com/example.jpg",
    }
    images_path = tmp_path / "images.csv"
    _write_csv(images_path, [image_base, {**image_base, "ImageID": "unknown-rotation", "Rotation": ""}])
    assert list(_image_metadata(images_path, {"good", "unknown-rotation"})) == ["good"]
