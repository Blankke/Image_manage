"""公开数据清单的路径、角点和配对契约测试。"""

from __future__ import annotations

import gzip
import json
import subprocess
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
from scripts.prepare_p2_geometry_data import (
    _download,
    _encode_remote_url,
    _find_quad,
    _public_usage,
    _select_midv_holo_clips,
)


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


def test_midv_holo_selection_is_balanced_by_lighting_device_and_document_kind(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    values = {
        "A": {
            "iphone": [
                "images/origins/ID/id01_01_01",
                "images/origins/ID/id02_01_01",
                "images/origins/passport/psp01_01_01",
                "images/origins/passport/psp02_01_01",
            ],
            "samsung": [
                "images/origins/ID/id03_01_01",
                "images/origins/passport/psp03_01_01",
            ],
        },
        "B": {
            "iphone": [
                "images/origins/ID/id04_01_01",
                "images/origins/passport/psp04_01_01",
            ]
        },
    }
    (metadata / "origins.json").write_text(json.dumps(values), encoding="utf-8")

    selected = _select_midv_holo_clips(metadata, clips_per_cell=1)

    assert selected == {
        "images/origins/ID/id01_01_01",
        "images/origins/passport/psp01_01_01",
        "images/origins/ID/id03_01_01",
        "images/origins/passport/psp03_01_01",
        "images/origins/ID/id04_01_01",
        "images/origins/passport/psp04_01_01",
    }


def test_midv_markup_quad_parser_accepts_nested_official_shape() -> None:
    assert _find_quad(
        {"document": {"templates": {"main": {"template_quad": [[1, 2], [3, 2], [3, 4], [1, 4]]}}}}
    ) == [
        [1.0, 2.0],
        [3.0, 2.0],
        [3.0, 4.0],
        [1.0, 4.0],
    ]


def test_public_image_url_encodes_spaces_without_double_encoding() -> None:
    assert _encode_remote_url("https://example.test/a b/already%20encoded.jpg?size=large image") == (
        "https://example.test/a%20b/already%20encoded.jpg?size=large%20image"
    )


def test_public_download_rebuilds_curl_command_to_resume_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "large.tar"
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        commands.append(command)
        if len(commands) == 1:
            destination.write_bytes(b"partial")
            raise subprocess.CalledProcessError(56, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("scripts.prepare_p2_geometry_data.time.sleep", lambda _seconds: None)

    _download("ftp://example.test/large.tar", destination)

    assert "--continue-at" not in commands[0]
    assert commands[1][commands[1].index("--continue-at") + 1] == "-"


def test_public_budget_excludes_private_and_transient_p2_archives(tmp_path: Path) -> None:
    retained = tmp_path / "geometry" / "sample.jpg"
    transient = tmp_path / "downloads" / "p2" / "images.tar"
    private = tmp_path / "private" / "sample.jpg"
    for path, payload in ((retained, b"keep"), (transient, b"archive"), (private, b"secret")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    assert _public_usage(tmp_path) == len(b"keep")


def test_p2_stage_c_keeps_private_test_out_of_gradient_and_replays_public(
    tmp_path: Path,
) -> None:
    from scripts.build_p2_geometry_manifests import main as build_p2_manifests

    data_root = tmp_path / "screenrestore-data"
    (data_root / "manifests").mkdir(parents=True)
    (data_root / "geometry" / "smartdoc").mkdir(parents=True)
    (data_root / "geometry" / "synthetic" / "images").mkdir(parents=True)
    (data_root / "private").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(data_root / "geometry" / "smartdoc" / "sample.jpg")
    Image.new("RGB", (16, 16)).save(data_root / "geometry" / "synthetic" / "images" / "sample.jpg")
    for split in ("train", "validation", "test"):
        Image.new("RGB", (16, 16)).save(data_root / "private" / f"{split}.jpg")

    def record(image: str, split: str, group_id: str, source: str) -> dict[str, object]:
        return {
            "image": image,
            "split": split,
            "group_id": group_id,
            "capture_session": f"session:{group_id}",
            "device": "test",
            "present": False,
            "target_class": "none",
            "content_quad": None,
            "outer_quad": None,
            "visible": False,
            "occlusion": 0.0,
            "glare_level": "none",
            "source": source,
        }

    smartdoc = record("geometry/smartdoc/sample.jpg", "train", "smartdoc:1", "smartdoc")
    synthetic = record("images/sample.jpg", "train", "synthetic:1", "synthetic")
    private = [
        record(f"private/{split}.jpg", split, f"private:{split}", "private-labeled")
        for split in ("train", "validation", "test")
    ]
    (data_root / "manifests" / "smartdoc.geometry.jsonl").write_text(
        json.dumps(smartdoc) + "\n", encoding="utf-8"
    )
    (data_root / "geometry" / "synthetic" / "manifest.jsonl").write_text(
        json.dumps(synthetic) + "\n", encoding="utf-8"
    )
    (data_root / "private" / "geometry.annotations.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in private), encoding="utf-8"
    )

    assert (
        build_p2_manifests(
            [
                "--data-root",
                str(data_root),
                "--stage-a-synthetic-samples",
                "1",
                "--stage-c-replay-samples",
                "1",
            ]
        )
        == 0
    )
    stage_c = [
        json.loads(line)
        for line in (data_root / "manifests" / "p2" / "stage-c.geometry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert all(
        record["source"] == "private-labeled"
        for record in stage_c
        if record["split"] in {"validation", "test"}
    )
    assert any(
        record["source"] != "private-labeled" and record["split"] == "train"
        for record in stage_c
    )


def test_private_grouping_merges_thumbnail_hd_and_content_duplicates(tmp_path: Path) -> None:
    from scripts.label_private_geometry import _group_images

    private = tmp_path / "private"
    private.mkdir()
    pattern = Image.new("RGB", (64, 48), (30, 60, 90))
    pattern.save(private / "artwork.jpg")
    pattern.resize((32, 24)).save(private / "artwork_hd.jpg")
    pattern.save(private / "same_content_different_name.jpg")
    Image.new("RGB", (64, 48), (220, 30, 20)).save(private / "different.jpg")

    groups = _group_images(private)

    assert len(groups) == 2
    assert sorted(len(images) for images in groups.values()) == [1, 3]


def test_private_ambiguous_annotation_is_a_rejection_target() -> None:
    from scripts.label_private_geometry import GroupAnnotation

    annotation = GroupAnnotation(
        target_class="artwork",
        content_quad=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        ambiguous=True,
    )

    assert not annotation.present


def test_private_multi_target_annotation_writes_authoritative_rejection(tmp_path: Path) -> None:
    from scripts.label_private_geometry import GroupAnnotation, _write_annotations

    data_root = tmp_path / "screenrestore-data"
    image = data_root / "private" / "multi.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image)
    group_id = "private:multi"
    annotation = GroupAnnotation(
        target_class="artwork",
        content_quad=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        scene_type="gallery_multi_target",
        # 即使交互状态暂时不一致，多目标语义也不能写成正样本。
        ambiguous=False,
    )
    output = data_root / "private" / "geometry.annotations.jsonl"

    _write_annotations(
        output,
        data_root,
        {group_id: [image]},
        {group_id: "validation"},
        {group_id: annotation},
        {group_id},
    )
    record = json.loads(output.read_text(encoding="utf-8"))

    assert annotation.multiple_targets
    assert not annotation.present
    assert record["scene_type"] == "gallery_multi_target"
    assert record["ambiguous"] is True
    assert record["present"] is False
    assert record["target_class"] == "none"
    assert record["content_quad"] is None
    assert record["outer_quad"] is None
    assert record["in_scope"] is False


def test_p2_manifest_rejects_multi_target_with_arbitrary_single_quad(tmp_path: Path) -> None:
    from scripts.build_p2_geometry_manifests import _validate_record_semantics

    record = {
        "present": True,
        "target_class": "artwork",
        "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "outer_quad": None,
        "visible": True,
        "ambiguous": True,
        "in_scope": True,
        "scene_type": "gallery_multi_target",
    }

    with pytest.raises(ValueError, match="多目标场景必须使用拒绝语义"):
        _validate_record_semantics(record, tmp_path / "private.jsonl", 3)


def test_p2_inventory_exposes_multi_target_slice() -> None:
    from scripts.build_p2_geometry_manifests import _statistics

    records = [
        {
            "group_id": "single",
            "split": "train",
            "target_class": "artwork",
            "present": True,
            "outer_quad": None,
            "scene_type": "gallery_artwork",
            "ambiguous": False,
        },
        {
            "group_id": "multi",
            "split": "validation",
            "target_class": "none",
            "present": False,
            "outer_quad": None,
            "scene_type": "gallery_multi_target",
            "ambiguous": True,
        },
    ]

    statistics = _statistics(records)

    assert statistics["multi_target"] == {"no": 1, "yes": 1}
    assert statistics["ambiguous"] == {"no": 1, "yes": 1}
