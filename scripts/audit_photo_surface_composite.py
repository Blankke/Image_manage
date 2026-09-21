#!/usr/bin/env python3
"""逐张检查 SmartDoc 纸面替换仅发生在标注四角内，并保存可追溯结果。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_photo_surface_composite.py \
      --data-root /Users/caozichen/screenrestore-data \
      --manifest /Users/caozichen/screenrestore-data/geometry/photo-surface-full-bleed-20260918/manifest.jsonl \
      --output /Users/caozichen/screenrestore-runs/p11-full-bleed-w15-public-pilot-20260918/photo-surface-pixel-audit.json

审计比较原始手机照片与合成图：纸面外的平均绝对差应接近 JPEG 重编码噪声，纸面内
应出现足够的真实内容替换。输入必须是 data-root 下的公开路径，不读取私人照片。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

OUTSIDE_MEAN_LIMIT = 8.0
OUTSIDE_P99_LIMIT = 18.0
INSIDE_MEAN_MINIMUM = 12.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    manifest = _public_file(args.manifest, root)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有审计结果：{output}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("合成清单为空")
    outside_values: list[float] = []
    outside_p99_values: list[float] = []
    inside_values: list[float] = []
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        result = _audit_record(row, root)
        if result["status"] == "PASS":
            outside_values.append(float(result["outside_mean_difference"]))
            outside_p99_values.append(float(result["outside_p99_difference"]))
            inside_values.append(float(result["inside_mean_difference"]))
        else:
            failures.append({"image": str(row.get("image", ""))[:160], **result})
        _progress(index, len(rows))
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "sample_count": len(rows),
        "failed_count": len(failures),
        "outside_mean_limit": OUTSIDE_MEAN_LIMIT,
        "outside_p99_limit": OUTSIDE_P99_LIMIT,
        "inside_mean_minimum": INSIDE_MEAN_MINIMUM,
        "outside_mean_median": statistics.median(outside_values) if outside_values else None,
        "outside_mean_max": max(outside_values) if outside_values else None,
        "outside_p99_max": max(outside_p99_values) if outside_p99_values else None,
        "inside_mean_median": statistics.median(inside_values) if inside_values else None,
        "inside_mean_min": min(inside_values) if inside_values else None,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "failures": failures[:20],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if summary["status"] == "PASS" else 1


def _audit_record(row: dict, root: Path) -> dict[str, object]:
    """在原片像素坐标内核对替换区域；纸面边缘留窄带避开抗锯齿。"""

    if row.get("source") != "smartdoc-photo-surface-composite":
        return {"status": "FAIL", "reason": "source_invalid"}
    original = cv2.imread(str(_public_file(root / str(row["source_capture_id"]), root)))
    composite = cv2.imread(str(_public_file(root / str(row["image"]), root)))
    if original is None or composite is None or original.shape != composite.shape:
        return {"status": "FAIL", "reason": "image_shape_or_read_invalid"}
    height, width = original.shape[:2]
    quad = np.asarray(row.get("content_quad"), np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or np.any((quad < 0) | (quad > 1)):
        return {"status": "FAIL", "reason": "content_quad_invalid"}
    points = np.rint(quad * np.array([width - 1, height - 1])).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    inner = cv2.erode(mask, np.ones((11, 11), np.uint8)) > 0
    outer = cv2.dilate(mask, np.ones((41, 41), np.uint8)) == 0
    if not inner.any() or not outer.any():
        return {"status": "FAIL", "reason": "audit_regions_empty"}
    difference = np.mean(np.abs(original.astype(np.int16) - composite.astype(np.int16)), axis=2)
    outside_mean = float(difference[outer].mean())
    outside_p99 = float(np.percentile(difference[outer], 99))
    inside_mean = float(difference[inner].mean())
    return {
        "status": (
            "PASS"
            if outside_mean <= OUTSIDE_MEAN_LIMIT
            and outside_p99 <= OUTSIDE_P99_LIMIT
            and inside_mean >= INSIDE_MEAN_MINIMUM
            else "FAIL"
        ),
        "outside_mean_difference": round(outside_mean, 6),
        "outside_p99_difference": round(outside_p99, 6),
        "inside_mean_difference": round(inside_mean, 6),
    }


def _public_file(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root) or any(
        "private" in part.lower() for part in resolved.relative_to(root).parts
    ) or not resolved.is_file():
        raise ValueError(f"审计只能读取 data-root 下的公开文件：{path}")
    return resolved


def _progress(done: int, total: int) -> None:
    if done < total and done % max(1, total // 40):
        return
    filled = round(24 * done / total)
    print(f"\r[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} 纸面像素审计",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
