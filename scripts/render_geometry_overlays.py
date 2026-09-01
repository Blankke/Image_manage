"""为有 GT 的 geometry validation/test 生成预测叠加图与 contact sheet。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/render_geometry_overlays.py \
        --manifest "$SCREENRESTORE_DATA_ROOT/manifests/p2/stage-b.geometry.jsonl" \
        --dataset-root "$SCREENRESTORE_DATA_ROOT" --split validation --max-images 50 \
        --quad-model "$SCREENRESTORE_RUN_ROOT/p2/stage-b/quadlocator-s.onnx" \
        --output-directory "$SCREENRESTORE_RUN_ROOT/p2/stage-b/overlays-public-validation"

颜色：青色 content GT、蓝色 outer GT、绿色/橙色 content prediction、紫色 outer
prediction。状态栏展示 corner、content/outer presence、class 和拒绝原因。报告使用匿名
sample ID，不保存源文件名；脚本只读取显式 manifest 指向的数据根内图片。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from screenrestore.geometry import (
    AutomaticGeometryService,
    ConfidencePolicy,
    CorrectnessCalibrator,
    OnnxQuadDetector,
)
from screenrestore.io.image_loader import load_image
from screenrestore.validation import corner_metrics

STATUS_HEIGHT = 146
TILE_WIDTH = 480
TILE_HEIGHT = 360
CONTACT_COLUMNS = 4
MULTI_TARGET_SCENE_TYPES = {
    "gallery_multi_target",
    "multiple_artworks",
    "multiple_equally_plausible",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--quad-model", type=Path, required=True)
    parser.add_argument("--correctness-calibrator", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=50, help="0 表示指定 split 全部")
    args = parser.parse_args(argv)
    if args.max_images < 0:
        raise ValueError("max-images 不能为负数")
    data_root = args.dataset_root.expanduser().resolve()
    records = _read_records(args.manifest, data_root, args.split)
    records = _balanced_records(records, args.max_images)
    calibrator = (
        CorrectnessCalibrator.load(args.correctness_calibrator)
        if args.correctness_calibrator is not None
        else None
    )
    service = AutomaticGeometryService(
        OnnxQuadDetector(args.quad_model),
        policy=ConfidencePolicy(calibrator=calibrator),
    )
    output_directory = args.output_directory.expanduser().resolve()
    preview_directory = output_directory / "previews"
    preview_directory.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []
    visual_records: list[dict[str, object]] = []
    report_records: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    multi_target_count = 0
    multi_target_rejected_count = 0
    for index, record in enumerate(records, start=1):
        _progress(index - 1, len(records), f"渲染 {args.split} overlay")
        sample_id = f"sample-{index:04d}"
        image_path = (data_root / str(record["image"])).resolve()
        document = load_image(image_path)
        decision = service.localize(document.original_rgb)
        scene_type = str(record.get("scene_type", "unknown"))
        if scene_type in MULTI_TARGET_SCENE_TYPES:
            multi_target_count += 1
            multi_target_rejected_count += int(decision.status.value == "rejected")
        reasons = [reason.value for reason in decision.rejection_reasons]
        rejection_counts.update(reasons)
        preview = _render(document.original_rgb, record, decision, sample_id)
        preview_path = preview_directory / f"{sample_id}.jpg"
        Image.fromarray(preview).save(preview_path, quality=92)
        tile = _tile(preview)
        tiles.append(tile)
        visual_metrics = _visual_metrics(record, decision, document.original_rgb.shape)
        visual_records.append({"tile": tile, **visual_metrics, "scene_type": scene_type})
        report_records.append(
            {
                "sample_id": sample_id,
                "group_hash": hashlib.sha256(str(record["group_id"]).encode("utf-8")).hexdigest()[:12],
                "source": str(record.get("source", "unknown")),
                "scene_type": scene_type,
                "status": decision.status.value,
                "target_class_gt": str(record["target_class"]),
                "target_class_prediction": decision.target_class.value,
                "content_presence": decision.diagnostics.get("presence_confidence"),
                "outer_presence": decision.diagnostics.get("outer_presence_confidence"),
                "corner_confidence": decision.diagnostics.get("minimum_corner_confidence"),
                "rejection_reasons": reasons,
                **visual_metrics,
                "preview": preview_path.relative_to(output_directory).as_posix(),
            }
        )
    contact_sheet = output_directory / "contact-sheet.jpg"
    Image.fromarray(_contact_sheet(tiles)).save(contact_sheet, quality=92)
    category_sheets = _category_contact_sheets(visual_records, output_directory)
    report = {
        "protocol": "labeled_geometry_visual_review",
        "split": args.split,
        "sample_count": len(records),
        "independent_group_count": len({str(record["group_id"]) for record in records}),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "multi_target_sample_count": multi_target_count,
        "multi_target_rejected_count": multi_target_rejected_count,
        "multi_target_rejection_rate": (
            multi_target_rejected_count / multi_target_count if multi_target_count else None
        ),
        "contains_source_image_identifiers": False,
        "contact_sheet": contact_sheet.name,
        "category_contact_sheets": category_sheets,
        "samples": report_records,
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _progress(len(records), len(records), f"完成：{contact_sheet}")
    print(contact_sheet)
    return 0


def _read_records(manifest: Path, data_root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("split") != split:
                continue
            image = (data_root / str(value.get("image", ""))).resolve()
            if not image.is_relative_to(data_root) or not image.is_file():
                raise ValueError(f"manifest 第 {line_number} 行图片不存在或越出 data-root")
            records.append(value)
    if not records:
        raise ValueError(f"manifest 没有 split={split} 的可视化样本")
    return records


def _balanced_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not limit or len(records) <= limit:
        return records
    # 先每 group 取一张，再按稳定 hash 补齐，降低长视频帧对 contact sheet 的支配。
    ordered = sorted(records, key=lambda item: hashlib.sha256(str(item["image"]).encode()).digest())
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for record in ordered:
        group = str(record["group_id"])
        if group not in seen_groups:
            selected.append(record)
            seen_groups.add(group)
            if len(selected) == limit:
                return selected
    selected_ids = {id(record) for record in selected}
    selected.extend(record for record in ordered if id(record) not in selected_ids)
    return selected[:limit]


def _render(image_rgb: np.ndarray, record: dict[str, Any], decision, sample_id: str) -> np.ndarray:  # type: ignore[no-untyped-def]
    height, width = image_rgb.shape[:2]
    scale = min(1.0, 1400 / max(height, width))
    resized = cv2.resize(
        image_rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((resized.shape[0] + STATUS_HEIGHT, resized.shape[1], 3), 245, np.uint8)
    canvas[STATUS_HEIGHT:] = resized
    offset = np.array([0.0, float(STATUS_HEIGHT)], np.float32)
    image_scale = np.array([max(1, width - 1), max(1, height - 1)], np.float32)
    content_gt = record.get("content_quad")
    outer_gt = record.get("outer_quad")
    if content_gt is not None:
        _draw_quad(canvas, np.asarray(content_gt, np.float32) * image_scale * scale + offset, (20, 210, 220), "content GT")
    if outer_gt is not None:
        _draw_quad(canvas, np.asarray(outer_gt, np.float32) * image_scale * scale + offset, (30, 100, 230), "outer GT")
    if decision.outer_corners is not None:
        _draw_quad(canvas, decision.outer_corners * scale + offset, (170, 60, 210), "outer pred")
    if decision.coarse_corners is not None:
        _draw_search_band(canvas, decision.coarse_corners * scale + offset, resized.shape[:2])
        _draw_quad(canvas, decision.coarse_corners * scale + offset, (240, 190, 30), "coarse")
    if decision.proposed_corners is not None:
        color = (30, 180, 70) if decision.accepted else (235, 145, 25)
        _draw_quad(canvas, decision.proposed_corners * scale + offset, color, "refined/final")
    diagnostics = decision.diagnostics
    reasons = ",".join(reason.value for reason in decision.rejection_reasons) or "none"
    support = diagnostics.get("edge_support", [])
    residual = diagnostics.get("refinement_residual_p95_px", [])
    lines = (
        f"{sample_id} {decision.status.value} GT={record['target_class']} pred={decision.target_class.value}",
        f"presence={_value(diagnostics, 'presence_confidence')} outer={_value(diagnostics, 'outer_presence_confidence')} corner={_value(diagnostics, 'minimum_corner_confidence')}",
        f"reasons={reasons}"[:150],
        f"edge_support={_short_values(support)} residual_p95={_short_values(residual)} outcome={diagnostics.get('refinement_outcome', 'n/a')}",
    )
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 26 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 2, cv2.LINE_AA)
    return canvas


def _short_values(value: object) -> str:
    if not isinstance(value, list):
        return "n/a"
    return "/".join(f"{float(item):.2f}" for item in value[:4])


def _draw_search_band(
    canvas: np.ndarray,
    corners: np.ndarray,
    image_shape: tuple[int, int],
) -> None:
    """用半透明宽线展示精修搜索窄带；精确像素宽度来自默认 EdgeRefineParameters。"""

    thickness = max(6, round(min(image_shape) * 0.018 * 2.0))
    overlay = canvas.copy()
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [points], True, (230, 90, 210), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)


def _visual_metrics(
    record: dict[str, Any],
    decision,  # type: ignore[no-untyped-def]
    image_shape: tuple[int, ...],
) -> dict[str, object]:
    target = record.get("content_quad")
    if target is None:
        return {
            "corner_nce": None,
            "quad_iou": None,
            "strict_correct": not decision.accepted,
            "false_accept": bool(decision.accepted),
            "false_reject": False,
            "wrong_layer": False,
            "refinement_gt_outcome": "rolled_back"
            if not decision.diagnostics.get("refinement_accepted")
            else "neutral",
        }
    scale = np.array([max(1, image_shape[1] - 1), max(1, image_shape[0] - 1)], np.float32)
    target_pixels = np.asarray(target, np.float32) * scale
    if decision.proposed_corners is None:
        nce, iou = 1.0, 0.0
    else:
        nce, iou, _maximum = corner_metrics(decision.proposed_corners, target_pixels)
    strict = bool(
        decision.target_class.value == str(record["target_class"])
        and nce <= 0.01
        and iou >= 0.93
        and decision.layer.value == "content"
    )
    outcome = "rolled_back" if not decision.diagnostics.get("refinement_accepted") else "neutral"
    if decision.coarse_corners is not None and decision.proposed_corners is not None:
        coarse_nce, coarse_iou, _ = corner_metrics(decision.coarse_corners, target_pixels)
        if nce < coarse_nce - 1e-5 and iou > coarse_iou + 1e-5:
            outcome = "improved"
        elif nce > coarse_nce + 1e-5 or iou < coarse_iou - 1e-5:
            outcome = "worsened"
    return {
        "corner_nce": round(float(nce), 8),
        "quad_iou": round(float(iou), 8),
        "strict_correct": strict,
        "false_accept": bool(decision.accepted and not strict),
        "false_reject": bool(not decision.accepted and record.get("present", True)),
        "wrong_layer": decision.layer.value != "content",
        "refinement_gt_outcome": outcome,
    }


def _category_contact_sheets(
    records: list[dict[str, object]],
    output_directory: Path,
) -> dict[str, str]:
    scored = [record for record in records if isinstance(record.get("corner_nce"), float)]
    scored.sort(key=lambda item: float(item["corner_nce"]))
    count = min(16, len(scored))
    middle = max(0, (len(scored) - count) // 2)
    categories: dict[str, list[dict[str, object]]] = {
        "best": scored[:count],
        "median": scored[middle : middle + count],
        "worst": scored[-count:] if count else [],
        "false-accept": [record for record in records if record.get("false_accept")][:16],
        "false-reject": [record for record in records if record.get("false_reject")][:16],
        "wrong-layer": [record for record in records if record.get("wrong_layer")][:16],
        "nested-frame": [
            record
            for record in records
            if any(token in str(record.get("scene_type", "")) for token in ("nested", "frame", "mat"))
        ][:16],
        "refinement-improved": [
            record for record in records if record.get("refinement_gt_outcome") == "improved"
        ][:16],
        "refinement-worsened": [
            record for record in records if record.get("refinement_gt_outcome") == "worsened"
        ][:16],
    }
    output: dict[str, str] = {}
    for name, selected in categories.items():
        if not selected:
            continue
        path = output_directory / f"contact-{name}.jpg"
        Image.fromarray(_contact_sheet([record["tile"] for record in selected])).save(path, quality=92)  # type: ignore[list-item]
        output[name] = path.name
    return output


def _value(diagnostics: dict[str, object], name: str) -> str:
    value = diagnostics.get(name)
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"


def _draw_quad(
    canvas: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], True, color, 3, cv2.LINE_AA)
    anchor = tuple(int(value) for value in points[0, 0])
    cv2.putText(canvas, label, (anchor[0] + 4, anchor[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def _tile(image: np.ndarray) -> np.ndarray:
    scale = min(TILE_WIDTH / image.shape[1], TILE_HEIGHT / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 235, np.uint8)
    y = (TILE_HEIGHT - resized.shape[0]) // 2
    x = (TILE_WIDTH - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return tile


def _contact_sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = math.ceil(len(tiles) / CONTACT_COLUMNS)
    sheet = np.full((rows * TILE_HEIGHT, CONTACT_COLUMNS * TILE_WIDTH, 3), 225, np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, CONTACT_COLUMNS)
        sheet[
            row * TILE_HEIGHT : (row + 1) * TILE_HEIGHT,
            column * TILE_WIDTH : (column + 1) * TILE_WIDTH,
        ] = tile
    return sheet


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * min(1.0, done / max(1, total)))
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
