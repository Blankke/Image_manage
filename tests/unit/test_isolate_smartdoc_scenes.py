"""SmartDoc 共享拍摄环境的过滤与跨 split 拦截。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.isolate_midv_scenes import isolate_rows as isolate_midv_rows
from scripts.isolate_smartdoc_scenes import isolate_rows

from screenrestore.io.geometry_isolation import (
    MIDV500_SCENE_FAMILY,
    MIDV_HOLO_SCENE_FAMILY,
    SMARTDOC_SCENE_FAMILY,
    validate_geometry_split_isolation,
)


def test_isolation_retains_smartdoc_train_and_other_public_validation(tmp_path: Path) -> None:
    root = tmp_path / "public"
    images = [
        "geometry/smartdoc/frames/background01/book/train.jpg",
        "geometry/smartdoc/frames/background01/letter/validation.jpg",
        "geometry/synthetic/validation.jpg",
    ]
    for relative in images:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"photo")
    rows = [
        {"image": images[0], "source": "smartdoc", "split": "train", "group_id": "book"},
        {"image": images[1], "source": "smartdoc", "split": "validation", "group_id": "letter"},
        {"image": images[2], "source": "synthetic", "split": "validation", "group_id": "synthetic"},
    ]
    manifest = root / "input.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    retained, removed, normalized = isolate_rows(manifest, root)

    assert [row["image"] for row in retained] == [images[0], images[2]]
    assert retained[0]["scene_group_id"] == SMARTDOC_SCENE_FAMILY
    assert removed == {("smartdoc", "validation"): 1}
    assert normalized == 0


def test_old_negative_visibility_is_normalized(tmp_path: Path) -> None:
    root = tmp_path / "public"
    image = root / "geometry/synthetic/negative.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"photo")
    path = root / "input.jsonl"
    path.write_text(json.dumps({
        "image": "geometry/synthetic/negative.jpg", "source": "synthetic",
        "split": "train", "group_id": "negative", "present": False,
        "target_class": "none", "visible": True,
    }) + "\n", encoding="utf-8")

    rows, removed, normalized = isolate_rows(path, root)

    assert not removed
    assert normalized == 1
    assert rows[0]["visible"] is False


def test_smartdoc_scene_identity_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="共享拍摄环境"):
        validate_geometry_split_isolation([
            {"image": "geometry/smartdoc/frames/background01/book/frame.jpg",
             "source": "smartdoc", "split": "train", "group_id": "book",
             "scene_group_id": "unique-desk"}
        ])


@pytest.mark.parametrize("source,family", [
    ("midv500", MIDV500_SCENE_FAMILY),
    ("midv-holo", MIDV_HOLO_SCENE_FAMILY),
])
def test_midv_documents_share_scene_across_splits(source: str, family: str) -> None:
    rows = [
        {"image": "geometry/example/train.jpg", "source": source,
         "split": "train", "group_id": "document-a"},
        {"image": "geometry/example/validation.jpg", "source": source,
         "split": "validation", "group_id": "document-b"},
    ]
    with pytest.raises(ValueError, match="scene_group_id"):
        validate_geometry_split_isolation(rows)
    rows[1]["scene_group_id"] = "invented-unique-scene"
    with pytest.raises(ValueError, match="共享拍摄环境"):
        validate_geometry_split_isolation(rows)


def test_midv_isolation_retains_train_and_other_public_validation(tmp_path: Path) -> None:
    root = tmp_path / "public"
    images = [
        "geometry/midv500/documents/a/train.jpg",
        "geometry/midv500/documents/b/validation.jpg",
        "geometry/midv-holo/subset/a/train.jpg",
        "geometry/midv-holo/subset/b/test.jpg",
        "geometry/synthetic/validation.jpg",
    ]
    for relative in images:
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"photo")
    splits = ["train", "validation", "train", "test", "validation"]
    sources = ["midv500", "midv500", "midv-holo", "midv-holo", "synthetic"]
    rows = [
        {"image": image, "source": source, "split": split, "group_id": f"group-{index}"}
        for index, (image, source, split) in enumerate(zip(images, sources, splits, strict=True))
    ]
    path = root / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    retained, removed = isolate_midv_rows(path, root)

    assert [row["image"] for row in retained] == [images[0], images[2], images[4]]
    assert [row.get("scene_group_id") for row in retained] == [
        MIDV500_SCENE_FAMILY, MIDV_HOLO_SCENE_FAMILY, None,
    ]
    assert removed == {("midv500", "validation"): 1, ("midv-holo", "test"): 1}
