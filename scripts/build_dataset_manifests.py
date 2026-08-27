"""为公开训练数据构建 ScreenRestore 清单，绝不读取 private 目录。

使用范例：
    source .venv/bin/activate
    which python
    export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
    python scripts/build_dataset_manifests.py --dataset smartdoc
    python scripts/build_dataset_manifests.py --dataset div2k

SmartDoc 原始标注的角点顺序为 TL、BL、BR、TR。本脚本会明确转换为项目统一的
TL、TR、BR、BL，并按 document model 分配 split，避免同一作品泄漏。DIV2K 清单
只列出公开 HR 与可选 x2 bicubic LR 配对；训练时的相机退化应在线生成。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import struct
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
SMARTDOC_REQUIRED_COLUMNS = {
    "bg_name",
    "model_name",
    "image_path",
    "frame_index",
    "tl_x",
    "tl_y",
    "bl_x",
    "bl_y",
    "br_x",
    "br_y",
    "tr_x",
    "tr_y",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--dataset", choices=("smartdoc", "div2k", "all"), default="all")
    parser.add_argument("--manifest-directory", type=Path)
    parser.add_argument(
        "--smartdoc-target-class",
        choices=("artwork", "postcard", "screen"),
        default="postcard",
        help="SmartDoc 是平面纸质内容，默认作为 postcard 几何代理类别。",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        raise ValueError("frame-stride 必须大于 0")
    data_root = _public_root(args.data_root)
    manifest_directory = (
        _public_root(args.manifest_directory) if args.manifest_directory else data_root / "manifests"
    )
    manifest_directory.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    if args.dataset in {"smartdoc", "all"}:
        reports.append(
            build_smartdoc_manifest(
                data_root,
                manifest_directory / "smartdoc.geometry.jsonl",
                target_class=args.smartdoc_target_class,
                frame_stride=args.frame_stride,
            )
        )
    if args.dataset in {"div2k", "all"}:
        reports.append(build_div2k_manifest(data_root, manifest_directory / "div2k.restoration.jsonl"))
    inventory_path = manifest_directory / "inventory.json"
    inventory_path.write_text(json.dumps({"datasets": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    print(inventory_path)
    return 0


def build_smartdoc_manifest(
    data_root: Path,
    output_path: Path,
    *,
    target_class: str = "postcard",
    frame_stride: int = 1,
) -> dict[str, Any]:
    """将 SmartDoc metadata.csv.gz 转为严格的几何 JSONL 清单。"""

    # 两个官方 archive 都有 metadata/README，同层解压会互相覆盖，因此 frames 独立存放。
    root = _public_root(data_root) / "geometry" / "smartdoc" / "frames"
    metadata_path = root / "metadata.csv.gz"
    if not metadata_path.is_file():
        return _missing_report("smartdoc", output_path, metadata_path)
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not SMARTDOC_REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(SMARTDOC_REQUIRED_COLUMNS - set(reader.fieldnames or []))
            raise ValueError(f"SmartDoc metadata 缺少列：{', '.join(missing)}")
        records, skipped_partial, sampled_out = _smartdoc_records(
            root,
            _public_root(data_root),
            reader,
            target_class,
            frame_stride,
        )
    _write_jsonl(output_path, records)
    splits = Counter(str(record["split"]) for record in records)
    return {
        "dataset": "smartdoc",
        "status": "ready",
        "manifest": str(output_path),
        "records": len(records),
        "splits": dict(sorted(splits.items())),
        "skipped_partial_or_invalid": skipped_partial,
        "sampled_out": sampled_out,
        "corner_order": "TL,TR,BR,BL",
        "target_class": target_class,
        "source_license": "CC-BY-4.0",
    }


def _smartdoc_records(
    root: Path,
    data_root: Path,
    rows: Iterable[dict[str, str]],
    target_class: str,
    frame_stride: int,
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    skipped_partial = 0
    sampled_out = 0
    for row in rows:
        frame_index = _integer(row["frame_index"], "frame_index")
        if (frame_index - 1) % frame_stride:
            sampled_out += 1
            continue
        image_path = _safe_relative_path(row["image_path"])
        image = (root / image_path).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"SmartDoc metadata 引用了不存在的图像：{image_path}")
        width, height = _image_size(image)
        # 官方顺序 TL、BL、BR、TR，统一写入项目顺序 TL、TR、BR、BL。
        quad = [
            [_number(row["tl_x"], "tl_x") / (width - 1), _number(row["tl_y"], "tl_y") / (height - 1)],
            [_number(row["tr_x"], "tr_x") / (width - 1), _number(row["tr_y"], "tr_y") / (height - 1)],
            [_number(row["br_x"], "br_x") / (width - 1), _number(row["br_y"], "br_y") / (height - 1)],
            [_number(row["bl_x"], "bl_x") / (width - 1), _number(row["bl_y"], "bl_y") / (height - 1)],
        ]
        if width < 2 or height < 2 or any(not 0.0 <= value <= 1.0 for point in quad for value in point):
            # 当前 release 的自动路径只训练完整可见目标；部分进入画面的纸张不伪造四角。
            skipped_partial += 1
            continue
        model_name = row["model_name"].strip()
        background = row["bg_name"].strip()
        if not model_name or not background:
            raise ValueError("SmartDoc 的 model_name 与 bg_name 不能为空")
        records.append(
            {
                # 所有数据清单以 SCREENRESTORE_DATA_ROOT 为锚点，供训练与 benchmark 共用。
                "image": image.relative_to(data_root).as_posix(),
                "split": _split_for_group(f"smartdoc:{model_name}"),
                "group_id": f"smartdoc:{model_name}",
                "capture_session": f"smartdoc:{background}:{model_name}",
                "device": "smartdoc-google-nexus-7",
                "present": True,
                "target_class": target_class,
                "content_quad": quad,
                "outer_quad": None,
                "visible": True,
                # 原始 metadata 不含遮挡/反光标签；这两项仅表示无可用标注，不可作质量标签使用。
                "occlusion": 0.0,
                "glare_level": "none",
            }
        )
    if not records:
        raise ValueError("SmartDoc 没有生成可训练样本；请检查解压目录与 frame-stride")
    return records, skipped_partial, sampled_out


def build_div2k_manifest(data_root: Path, output_path: Path) -> dict[str, Any]:
    """建立 DIV2K HR、x2 bicubic 与 x4 wild 配对 inventory，不产生离线副本。"""

    root = _public_root(data_root) / "superres" / "div2k"
    data_root = _public_root(data_root)
    records: list[dict[str, Any]] = []
    missing_hr: list[str] = []
    for split, directory_name in (("train", "DIV2K_train_HR"), ("validation", "DIV2K_valid_HR")):
        directory = root / directory_name
        if not directory.is_dir():
            missing_hr.append(directory_name)
            continue
        lr_directory = root / f"DIV2K_{'train' if split == 'train' else 'valid'}_LR_bicubic" / "X2"
        wild_directory = root / "wild_x4" / f"DIV2K_{'train' if split == 'train' else 'valid'}_LR_wild"
        for image in sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png"):
            sample_id = image.stem
            lr = lr_directory / f"{sample_id}x2.png"
            # wild 训练集每张 HR 有四个真实退化版本（x4w1..x4w4）；验证集为单版本。
            # 以列表保留多对一关系，禁止任意选一张覆盖其它真实观测。
            wild_images = sorted(wild_directory.glob(f"{sample_id}x4w*.png"))
            records.append(
                {
                    "sample_id": f"div2k:{sample_id}",
                    "split": split,
                    "hr_image": image.resolve().relative_to(data_root).as_posix(),
                    "lr_x2_image": (
                        lr.resolve().relative_to(data_root).as_posix()
                        if lr.is_file()
                        else None
                    ),
                    "wild_x4_images": [
                        wild.resolve().relative_to(data_root).as_posix() for wild in wild_images
                    ],
                    "source": "DIV2K",
                    "license": "NTIRE/DIV2K official terms",
                }
            )
    if records:
        _write_jsonl(output_path, records)
    report: dict[str, Any] = {
        "dataset": "div2k",
        "status": "ready" if records else "missing",
        "manifest": str(output_path),
        "records": len(records),
        "splits": dict(sorted(Counter(str(record["split"]) for record in records).items())),
        "paired_x2": sum(record["lr_x2_image"] is not None for record in records),
        "paired_wild_x4": sum(bool(record["wild_x4_images"]) for record in records),
        "wild_x4_variants": sum(len(record["wild_x4_images"]) for record in records),
        "on_the_fly_degradation": True,
    }
    if missing_hr:
        report["missing_hr_directories"] = missing_hr
    return report


def _default_data_root() -> Path:
    return Path(os.environ.get("SCREENRESTORE_DATA_ROOT", "~/screenrestore-data")).expanduser()


def _public_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if _contains_user_private_component(resolved):
        raise ValueError("数据准备工具拒绝读取或写入 private 目录")
    return resolved


def _contains_user_private_component(path: Path) -> bool:
    """拒绝数据根内的 private；仅放行 macOS 系统临时目录固定的 /private/var 前缀。"""

    parts = path.parts
    start = 3 if parts[:3] == ("/", "private", "var") else 0
    return "private" in parts[start:]


def _missing_report(dataset: str, output_path: Path, expected: Path) -> dict[str, Any]:
    return {"dataset": dataset, "status": "missing", "manifest": str(output_path), "expected": str(expected)}


def _split_for_group(group_id: str) -> str:
    """稳定按组切分，避免同一 document model 的视频帧泄漏。"""

    bucket = hashlib.sha256(group_id.encode("utf-8")).digest()[0] % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "private" in path.parts:
        raise ValueError(f"不安全的公开数据相对路径：{value!r}")
    return path


def _number(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"SmartDoc {name} 不是数字") from exc


def _integer(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"SmartDoc {name} 不是整数") from exc


def _image_size(path: Path) -> tuple[int, int]:
    """只解析 PNG/JPEG 头，避免为清单引入 Pillow 或把像素加载进内存。"""

    with path.open("rb") as handle:
        header = handle.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return width, height
        if header[:2] != b"\xff\xd8":
            raise ValueError(f"不支持的图像格式：{path}")
        handle.seek(2)
        while True:
            marker_prefix = handle.read(1)
            while marker_prefix == b"\xff":
                marker_prefix = handle.read(1)
            if not marker_prefix:
                break
            marker = marker_prefix[0]
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            raw_length = handle.read(2)
            if len(raw_length) != 2:
                break
            length = struct.unpack(">H", raw_length)[0]
            if length < 2:
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                values = handle.read(5)
                if len(values) != 5:
                    break
                height, width = struct.unpack(">HH", values[1:5])
                return width, height
            handle.seek(length - 2, 1)
    raise ValueError(f"无法读取 JPEG 尺寸：{path}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _public_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
