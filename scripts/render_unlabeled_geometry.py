"""生成无 GT 图片的自动几何人工复核效果图。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/render_unlabeled_geometry.py \
        --image-directory "/Users/caozichen/screenrestore-data/private" \
        --quad-model "/Users/caozichen/screenrestore-runs/p1-full/geometry/quadlocator-s.onnx" \
        --output-directory "/Users/caozichen/screenrestore-runs/p1-full/geometry/private-effects"

脚本只读取显式指定的图片目录。默认优先选择同名 ``*_hd`` 图片，避免把缩略图与高清图
重复计入人工复核。accepted 样本额外输出透视校正图；rejected 样本只输出候选叠加图，
不会绕过产品的自动拒绝策略。报告使用匿名 sample 编号，不保存源文件名或图片像素。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from screenrestore.geometry import AutomaticGeometryService, OnnxQuadDetector, warp_perspective
from screenrestore.io.image_loader import load_image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
STATUS_HEIGHT = 92
PREVIEW_MAX_EDGE = 1400
CONTACT_TILE_SIZE = (480, 360)
CONTACT_COLUMNS = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--quad-model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 表示处理全部去重后的图片")
    parser.add_argument(
        "--include-thumbnails",
        action="store_true",
        help="同时处理与 *_hd 对应的低分辨率缩略图",
    )
    args = parser.parse_args(argv)
    if args.max_images < 0:
        raise ValueError("max-images 不能为负数")

    image_directory = args.image_directory.expanduser().resolve()
    if not image_directory.is_dir():
        raise ValueError(f"图片目录不存在：{image_directory}")
    images = _find_images(image_directory, prefer_hd=not args.include_thumbnails)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise ValueError("显式目录中没有可处理图片")

    output_directory = args.output_directory.expanduser().resolve()
    preview_directory = output_directory / "previews"
    rectified_directory = output_directory / "rectified"
    preview_directory.mkdir(parents=True, exist_ok=True)
    rectified_directory.mkdir(parents=True, exist_ok=True)

    service = AutomaticGeometryService(OnnxQuadDetector(args.quad_model))
    records: list[dict[str, object]] = []
    contact_tiles: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()
    accepted_count = 0

    for index, image_path in enumerate(images, start=1):
        _progress(index - 1, len(images), "生成无 GT 几何效果图")
        sample_id = f"sample-{index:03d}"
        document = load_image(image_path)
        decision = service.localize(document.original_rgb)
        reasons = [reason.value for reason in decision.rejection_reasons]
        rejection_counts.update(reasons)
        accepted_count += int(decision.accepted)

        preview = _render_preview(document.original_rgb, decision, sample_id)
        preview_path = preview_directory / f"{sample_id}.jpg"
        _save_rgb(preview_path, preview, quality=92)
        contact_tiles.append(_contact_tile(preview))

        rectified_relative: str | None = None
        if decision.accepted and decision.proposed_corners is not None:
            # AUTO 画幅继续沿用 geometry 权威实现；未知内参时结果仍属于画幅估计。
            rectified, _matrix = warp_perspective(
                document.original_rgb,
                decision.proposed_corners,
            )
            rectified_path = rectified_directory / f"{sample_id}.png"
            _save_rgb(rectified_path, rectified)
            rectified_relative = rectified_path.relative_to(output_directory).as_posix()

        records.append(
            {
                "sample_id": sample_id,
                "status": decision.status.value,
                "target_class": decision.target_class.value,
                "confidence": round(decision.confidence, 6),
                "rejection_reasons": reasons,
                "has_proposed_quad": decision.proposed_corners is not None,
                "preview": preview_path.relative_to(output_directory).as_posix(),
                "rectified": rectified_relative,
            }
        )

    contact_sheet_path = output_directory / "contact-sheet.jpg"
    _save_rgb(contact_sheet_path, _contact_sheet(contact_tiles), quality=92)
    report = {
        "protocol": "unlabeled_geometry_visual_review",
        "model": args.quad_model.expanduser().resolve().name,
        "source_image_count": len(images),
        "accepted_count": accepted_count,
        "rejected_count": len(images) - accepted_count,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "preferred_hd": not args.include_thumbnails,
        "contains_ground_truth": False,
        "contains_source_image_identifiers": False,
        "contact_sheet": contact_sheet_path.name,
        "samples": records,
    }
    report_path = output_directory / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _progress(len(images), len(images), f"完成：{contact_sheet_path}")
    print(contact_sheet_path)
    return 0


def _find_images(directory: Path, *, prefer_hd: bool) -> list[Path]:
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not prefer_hd:
        return images
    selected: dict[tuple[Path, str, str], Path] = {}
    for path in images:
        stem = path.stem
        is_hd = stem.endswith("_hd")
        logical_stem = stem[:-3] if is_hd else stem
        key = (path.parent, logical_stem, path.suffix.lower())
        current = selected.get(key)
        if current is None or (is_hd and not current.stem.endswith("_hd")):
            selected[key] = path
    return sorted(selected.values())


def _render_preview(image_rgb: np.ndarray, decision, sample_id: str) -> np.ndarray:  # type: ignore[no-untyped-def]
    """在缩放后的副本上绘制内容/外框候选与拒绝状态。"""

    height, width = image_rgb.shape[:2]
    scale = min(1.0, PREVIEW_MAX_EDGE / max(height, width))
    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    preview = cv2.resize(image_rgb, (preview_width, preview_height), interpolation=interpolation)
    canvas = np.full((preview_height + STATUS_HEIGHT, preview_width, 3), 245, dtype=np.uint8)
    canvas[STATUS_HEIGHT:] = preview

    # OpenCV 文字只使用 ASCII，避免依赖本机字体；颜色值位于 RGB 数组中。
    status = "ACCEPTED" if decision.accepted else "REJECTED"
    status_color_rgb = (32, 150, 74) if decision.accepted else (200, 55, 45)
    reasons = ",".join(reason.value for reason in decision.rejection_reasons) or "none"
    line_one = f"{sample_id}  {status}  class={decision.target_class.value}  conf={decision.confidence:.3f}"
    line_two = f"reasons={reasons}"
    _put_rgb_text(canvas, line_one, (18, 34), status_color_rgb, 0.72, 2)
    _put_rgb_text(canvas, line_two[:120], (18, 70), (45, 45, 45), 0.58, 1)

    offset = np.array([0.0, float(STATUS_HEIGHT)], dtype=np.float32)
    if decision.outer_corners is not None:
        _draw_quad(canvas, decision.outer_corners * scale + offset, (150, 70, 210), "outer")
    if decision.proposed_corners is not None:
        color = (32, 190, 90) if decision.accepted else (240, 145, 35)
        _draw_quad(canvas, decision.proposed_corners * scale + offset, color, "content")
    return canvas


def _draw_quad(
    canvas_rgb: np.ndarray, corners: np.ndarray, color_rgb: tuple[int, int, int], label: str
) -> None:
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas_rgb, [points], True, color_rgb, thickness=4, lineType=cv2.LINE_AA)
    anchor = tuple(int(value) for value in points[0, 0])
    _put_rgb_text(canvas_rgb, label, (anchor[0] + 6, anchor[1] - 8), color_rgb, 0.62, 2)


def _put_rgb_text(
    canvas_rgb: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color_rgb: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        canvas_rgb,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color_rgb,
        thickness,
        cv2.LINE_AA,
    )


def _contact_tile(preview_rgb: np.ndarray) -> np.ndarray:
    target_width, target_height = CONTACT_TILE_SIZE
    scale = min(target_width / preview_rgb.shape[1], target_height / preview_rgb.shape[0])
    width = max(1, round(preview_rgb.shape[1] * scale))
    height = max(1, round(preview_rgb.shape[0] * scale))
    resized = cv2.resize(preview_rgb, (width, height), interpolation=cv2.INTER_AREA)
    tile = np.full((target_height, target_width, 3), 238, dtype=np.uint8)
    x = (target_width - width) // 2
    y = (target_height - height) // 2
    tile[y : y + height, x : x + width] = resized
    return tile


def _contact_sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = math.ceil(len(tiles) / CONTACT_COLUMNS)
    tile_width, tile_height = CONTACT_TILE_SIZE
    sheet = np.full(
        (rows * tile_height, CONTACT_COLUMNS * tile_width, 3),
        225,
        dtype=np.uint8,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, CONTACT_COLUMNS)
        y, x = row * tile_height, column * tile_width
        sheet[y : y + tile_height, x : x + tile_width] = tile
    return sheet


def _save_rgb(path: Path, image_rgb: np.ndarray, *, quality: int | None = None) -> None:
    options = {"quality": quality} if quality is not None else {}
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(path, **options)


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
