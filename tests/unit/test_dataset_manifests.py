"""公开数据清单的路径、角点和配对契约测试。"""

from __future__ import annotations

import gzip
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from scripts.build_dataset_manifests import (
    _public_root as manifest_public_root,
)
from scripts.build_dataset_manifests import (
    build_div2k_manifest,
    build_smartdoc_manifest,
)
from scripts.prepare_datasets import _is_complete_archive, _safe_extract_members


def test_smartdoc_manifest_converts_official_corner_order_and_skips_partial(tmp_path: Path) -> None:
    data_root = tmp_path / "screenrestore-data"
    frames = data_root / "geometry" / "smartdoc" / "frames" / "background01" / "letter001"
    frames.mkdir(parents=True)
    Image.new("RGB", (9, 7)).save(frames / "frame_0001.jpeg")
    Image.new("RGB", (9, 7)).save(frames / "frame_0002.jpeg")
    metadata = data_root / "geometry" / "smartdoc" / "frames" / "metadata.csv.gz"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "bg_name,model_name,image_path,frame_index,tl_x,tl_y,bl_x,bl_y,br_x,br_y,tr_x,tr_y\n"
    )
    valid = "background01,letter001,background01/letter001/frame_0001.jpeg,1,0,0,0,6,8,6,8,0\n"
    partial = "background01,letter001,background01/letter001/frame_0002.jpeg,2,-1,0,0,6,8,6,8,0\n"
    with gzip.open(metadata, "wt", encoding="utf-8") as handle:
        handle.write(header + valid + partial)

    output = data_root / "manifests" / "smartdoc.geometry.jsonl"
    report = build_smartdoc_manifest(data_root, output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert report["records"] == 1
    assert report["skipped_partial_or_invalid"] == 1
    assert record["image"] == "geometry/smartdoc/frames/background01/letter001/frame_0001.jpeg"
    assert record["content_quad"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert record["group_id"] == "smartdoc:letter001"
    assert record["capture_session"] == "smartdoc:background01:letter001"


def test_div2k_manifest_uses_data_root_relative_hr_and_optional_lr(tmp_path: Path) -> None:
    data_root = tmp_path / "screenrestore-data"
    hr = data_root / "superres" / "div2k" / "DIV2K_train_HR"
    lr = data_root / "superres" / "div2k" / "DIV2K_train_LR_bicubic" / "X2"
    wild = data_root / "superres" / "div2k" / "wild_x4" / "DIV2K_train_LR_wild"
    hr.mkdir(parents=True)
    lr.mkdir(parents=True)
    wild.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(hr / "0001.png")
    Image.new("RGB", (4, 4)).save(lr / "0001x2.png")
    Image.new("RGB", (2, 2)).save(wild / "0001x4w1.png")
    Image.new("RGB", (2, 2)).save(wild / "0001x4w2.png")

    output = data_root / "manifests" / "div2k.restoration.jsonl"
    report = build_div2k_manifest(data_root, output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert report["records"] == 1
    assert report["paired_x2"] == 1
    assert report["paired_wild_x4"] == 1
    assert report["wild_x4_variants"] == 2
    assert record["hr_image"] == "superres/div2k/DIV2K_train_HR/0001.png"
    assert record["lr_x2_image"] == "superres/div2k/DIV2K_train_LR_bicubic/X2/0001x2.png"
    assert record["wild_x4_images"] == [
        "superres/div2k/wild_x4/DIV2K_train_LR_wild/0001x4w1.png",
        "superres/div2k/wild_x4/DIV2K_train_LR_wild/0001x4w2.png",
    ]


def test_manifest_rejects_private_child_but_allows_macos_system_temp_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private"):
        manifest_public_root(tmp_path / "screenrestore-data" / "private" / "gallery")

    # pytest 的 macOS 临时目录位于 /private/var，属于系统路径而非用户数据 private。
    assert manifest_public_root(tmp_path).is_dir()


def test_archive_extraction_rejects_tar_links(tmp_path: Path) -> None:
    link = tarfile.TarInfo("safe-looking-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"

    with pytest.raises(ValueError, match="链接"):
        _safe_extract_members([link], tmp_path, lambda _member: None)


def test_local_archive_must_pass_integrity_check_before_reuse(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr("sample.txt", "ScreenRestore")

    assert _is_complete_archive(archive, checksum=None)
    assert not _is_complete_archive(tmp_path / "missing.zip", checksum=None)
