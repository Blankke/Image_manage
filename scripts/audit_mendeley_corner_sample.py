#!/usr/bin/env python3
"""核验 Mendeley 文档角点抽样并绘制内容层人工审核拼图。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_mendeley_corner_sample.py \
      --data-root /Users/caozichen/screenrestore-data \
      --sample-directory /Users/caozichen/screenrestore-data/geometry/mendeley-corner-audit-20260918

输出仍是待审核候选；ZIP 提供的文档角点不自动等于 ScreenRestore 的内容四角。
脚本只读公开抽样图片、标签和索引，不读取私人集，不覆盖既有审核结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from screenrestore.geometry.rectify import order_corners

CELL_WIDTH = 290
CELL_HEIGHT = 270
COLUMNS = 6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    sample = args.sample_directory.expanduser().resolve()
    if not sample.is_relative_to(root):
        raise ValueError("sample-directory 必须位于 data-root 内")
    paths = [sample / name for name in ("geometry-audit.json", "review-plain.jpg", "review-labeled.jpg")]
    if any(path.exists() for path in paths):
        raise FileExistsError("审核输出已存在，拒绝覆盖")
    index_path = sample / "sample-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index["samples"]
    if not isinstance(records, list) or not records:
        raise ValueError("抽样索引缺少 samples")
    board_size = (CELL_WIDTH * COLUMNS, CELL_HEIGHT * ((len(records) + COLUMNS - 1) // COLUMNS))
    plain = Image.new("RGB", board_size, "white")
    labeled = Image.new("RGB", board_size, "white")
    details: list[dict[str, object]] = []
    for index_number, record in enumerate(records, 1):
        image_path = _public_path(root, str(record["image"]))
        label_path = _public_path(root, str(record["annotation"]))
        if _sha256(image_path) != record["image_sha256"] or _sha256(label_path) != record["annotation_sha256"]:
            raise ValueError(f"抽样 {record['source_id']} 的图片或标签摘要不符")
        with Image.open(image_path) as source:
            raw_size = source.size
            orientation = int(source.getexif().get(274, 1))
            # 上游 CSV 使用按 EXIF 展示后的照片坐标；同一坐标系中绘制和验证四角。
            photo = ImageOps.exif_transpose(source).convert("RGB")
        quad, annotation_error = _read_quad(label_path, photo.size)
        if quad is None:
            area_fraction = None
            normalized = None
        else:
            xs, ys = quad[:, 0], quad[:, 1]
            area = float(abs(np.sum(xs * np.roll(ys, -1) - ys * np.roll(xs, -1))) / 2)
            area_fraction = round(area / (photo.width * photo.height), 6)
            normalized = (quad / np.asarray(photo.size, dtype=np.float32)).round(6).tolist()
        details.append({
            "source_id": int(record["source_id"]),
            "image": str(record["image"]),
            "annotation": str(record["annotation"]),
            "raw_size": list(raw_size),
            "oriented_size": list(photo.size),
            "exif_orientation": orientation,
            "annotation_valid": annotation_error is None,
            "annotation_error": annotation_error,
            "quad_normalized": normalized,
            "area_fraction": area_fraction,
            "visual_content_layer_review": "pending",
            "underlying_material_rights_review": "pending",
            "independent_scene_review": "pending",
            "evaluation_eligible": False,
        })
        _draw_cell(plain, labeled, photo, quad, index_number - 1, int(record["source_id"]))
        _progress(index_number, len(records))
    payload = {
        "kind": "mendeley_corner_geometry_audit",
        "dataset_url": index["dataset_url"],
        "dataset_version": index["dataset_version"],
        "dataset_license": index["dataset_license"],
        "sample_index_sha256": _sha256(index_path),
        "reviewed_samples": details,
    }
    paths[0].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plain.save(paths[1], quality=90)
    labeled.save(paths[2], quality=90)
    print(json.dumps({"sample_count": len(details), "audit": str(paths[0])}, ensure_ascii=False))
    return 0


def _read_quad(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    fields = path.read_text(encoding="utf-8-sig").strip().split(",")
    if len(fields) < 8:
        return None, "fewer_than_eight_coordinates"
    try:
        quad = order_corners(np.asarray([float(value) for value in fields[:8]], dtype=np.float32))
    except (ValueError, TypeError) as exc:
        return None, f"invalid_quad:{type(exc).__name__}"
    width, height = image_size
    if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] > width) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] > height):
        return None, "coordinates_outside_exif_oriented_image"
    return quad, None


def _draw_cell(
    plain: Image.Image,
    labeled: Image.Image,
    photo: Image.Image,
    quad: np.ndarray | None,
    position: int,
    source_id: int,
) -> None:
    thumbnail = photo.copy()
    thumbnail.thumbnail((CELL_WIDTH - 12, CELL_HEIGHT - 36))
    x = position % COLUMNS * CELL_WIDTH + (CELL_WIDTH - thumbnail.width) // 2
    y = position // COLUMNS * CELL_HEIGHT + 30
    for board in (plain, labeled):
        board.paste(thumbnail, (x, y))
        ImageDraw.Draw(board).text((position % COLUMNS * CELL_WIDTH + 6, y - 23), str(source_id), fill="black")
    # thumbnail 的缩放只用于可视化；geometry-audit.json 始终保存原图归一化坐标。
    draw = ImageDraw.Draw(labeled)
    if quad is None:
        draw.text((x + 5, y + 5), "INVALID LABEL", fill="red")
    else:
        sx, sy = thumbnail.width / photo.width, thumbnail.height / photo.height
        points = [(x + float(px) * sx, y + float(py) * sy) for px, py in quad]
        draw.line([*points, points[0]], fill="lime", width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="red")


def _public_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("抽样路径不存在或越界")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(done: int, total: int) -> None:
    filled = round(24 * done / total)
    print(f"\r审核拼图 [{'#' * filled}{'-' * (24 - filled)}] {done}/{total}",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
