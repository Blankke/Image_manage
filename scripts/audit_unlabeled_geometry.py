"""审计无 GT 图片的自动几何行为，绝不把预测伪造为训练标签。

使用范例：
    source .venv/bin/activate
    which python
    python -m pip install -e '.[inference-onnx]'
    python scripts/audit_unlabeled_geometry.py \
        --image-directory "$SCREENRESTORE_DATA_ROOT/private" \
        --quad-model "$SCREENRESTORE_RUN_ROOT/geometry/smartdoc/best.onnx" \
        --output "$SCREENRESTORE_RUN_ROOT/geometry/private-unlabeled-audit.json"

只会读取调用方显式指定目录内的图片，并写入汇总计数、接受率与拒绝原因；报告不包含
文件名、像素、四角预测或可识别的图像内容。无 GT 结果只能用于人工复核和分布审计。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from screenrestore.geometry import AutomaticGeometryService, OnnxQuadDetector
from screenrestore.io.image_loader import load_image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--quad-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 表示审计全部图片")
    args = parser.parse_args(argv)
    if args.max_images < 0:
        raise ValueError("max-images 不能为负数")
    image_directory = args.image_directory.expanduser().resolve()
    if not image_directory.is_dir():
        raise ValueError(f"图片目录不存在：{image_directory}")
    images = [
        item
        for item in sorted(image_directory.rglob("*"))
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise ValueError("显式目录中没有可审计图片")
    service = AutomaticGeometryService(OnnxQuadDetector(args.quad_model))
    accepted = 0
    rejected_reasons: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    failures = 0
    for index, image_path in enumerate(images, start=1):
        _progress(index - 1, len(images), "无 GT 自动定位审计")
        try:
            decision = service.localize(load_image(image_path).original_rgb)
        except (OSError, RuntimeError, ValueError):
            failures += 1
            continue
        classes[decision.target_class.value] += 1
        if decision.accepted:
            accepted += 1
        else:
            rejected_reasons.update(decision.rejection_reasons or ("unspecified",))
    _progress(len(images), len(images), "无 GT 自动定位审计")
    report = {
        "protocol": "unlabeled_geometry_audit",
        "model": args.quad_model.name,
        "input_count": len(images),
        "decode_or_inference_failures": failures,
        "accepted_count": accepted,
        "accepted_rate": round(accepted / max(1, len(images) - failures), 8),
        "predicted_class_counts": dict(sorted(classes.items())),
        "rejection_reason_counts": dict(sorted(rejected_reasons.items())),
        "contains_ground_truth": False,
        "contains_image_identifiers": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>4}/{total:<4} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
