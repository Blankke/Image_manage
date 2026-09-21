#!/usr/bin/env python3
"""审计同一物品多视角预测的照片特征一致性，不读取四角真值。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_multiview_candidate_consistency.py \
      --data-root /Users/caozichen/screenrestore-data \
      --index output/private-validation/b0-20260912/prepare/dataset-index.json \
      --predictions output/private-validation/<run>/frozen-predictions/predictions.json \
      --domain artwork --domain poster \
      --output /Users/caozichen/screenrestore-runs/multiview-audit.json

该脚本仅输出各候选四角在其余照片中的特征匹配证据，不修改自动接受结果。
所有图像只在本机读取，不将图像内容写入报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖审计结果：{output}")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    frozen = json.loads(args.predictions.read_text(encoding="utf-8"))
    if index["dataset_sha256"] != frozen["dataset_sha256"]:
        raise ValueError("照片索引和冻结预测的数据身份不一致")
    predictions = {row["image_id"]: row for row in frozen["predictions"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for frame in index["frames"]:
        if args.domain and frame["domain"] not in args.domain:
            continue
        prediction = predictions.get(frame["image_id"])
        if prediction is None or prediction["image_sha256"] != frame["image_sha256"]:
            raise ValueError(f"冻结预测缺失或图片身份不匹配：{frame['image_id']}")
        path = (root / frame["image"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"照片路径不存在或越界：{frame['image_id']}")
        grouped[frame["group_id"]].append({**frame, "path": path, "decision": prediction["decision"]})
    if not grouped:
        raise ValueError("没有待审计的照片组")
    detector = cv2.SIFT_create(nfeatures=2500)
    groups = []
    for index_number, (group_id, rows) in enumerate(sorted(grouped.items()), start=1):
        features = [_features(row["path"], detector) for row in rows]
        candidates = []
        for seed_index, seed in enumerate(rows):
            corners = seed["decision"].get("corners")
            if corners is None:
                continue
            support = []
            for target_index, target in enumerate(rows):
                if seed_index == target_index:
                    continue
                support.append(
                    _pair_evidence(
                        features[seed_index],
                        features[target_index],
                        np.asarray(corners, np.float32),
                        target["image_id"],
                    )
                )
            candidates.append(
                {
                    "image_id": seed["image_id"],
                    "view_support_count": sum(item["supported"] for item in support),
                    "total_inliers": sum(item["inliers"] for item in support),
                    "mean_seed_coverage": round(
                        float(np.mean([item["seed_coverage"] for item in support])), 4
                    ),
                    "pair_evidence": support,
                }
            )
        candidates.sort(
            key=lambda item: (item["view_support_count"], item["total_inliers"]),
            reverse=True,
        )
        groups.append({"group_id": group_id, "domain": rows[0]["domain"], "candidates": candidates})
        _progress(index_number, len(grouped))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "kind": "photo_only_multiview_candidate_consistency",
                "dataset_sha256": index["dataset_sha256"],
                "prediction_sha256": frozen["prediction_sha256"],
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"审计完成：{len(groups)} 组，输出 {output}")
    return 0


def _features(path: Path, detector: cv2.SIFT) -> dict:
    """统一在最长边 1280 像素的 RGB 无关灰度坐标系提取特征。"""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"无法读取照片：{path}")
    height, width = image.shape
    scale = min(1.0, 1280.0 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return {
        "shape": image.shape,
        "points": np.asarray([point.pt for point in keypoints], np.float32).reshape(-1, 2),
        "descriptors": descriptors,
    }


def _pair_evidence(seed: dict, target: dict, normalized_quad: np.ndarray, image_id: str) -> dict:
    """只匹配种子候选内部特征；足够分布的内点才支持跨视角一致。"""

    quad = normalized_quad * np.asarray([seed["shape"][1] - 1, seed["shape"][0] - 1])
    seed_points = seed["points"]
    inside = np.asarray(
        [cv2.pointPolygonTest(quad.astype(np.float32), tuple(map(float, point)), False) >= 0 for point in seed_points],
        bool,
    )
    source_descriptors = seed["descriptors"]
    target_descriptors = target["descriptors"]
    result = {
        "image_id": image_id,
        "matches": 0,
        "inliers": 0,
        "inlier_fraction": 0.0,
        "seed_coverage": 0.0,
        "supported": False,
    }
    if source_descriptors is None or target_descriptors is None or int(inside.sum()) < 12:
        return result
    selected_points = seed_points[inside]
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_descriptors[inside], target_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    result["matches"] = len(matches)
    if len(matches) < 12:
        return result
    source_xy = np.asarray([selected_points[item.queryIdx] for item in matches], np.float32)
    target_xy = np.asarray([target["points"][item.trainIdx] for item in matches], np.float32)
    matrix, mask = cv2.findHomography(source_xy, target_xy, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return result
    inliers = mask.ravel().astype(bool)
    result["inliers"] = int(inliers.sum())
    result["inlier_fraction"] = round(float(inliers.mean()), 4)
    if not inliers.any():
        return result
    bbox = np.ptp(quad, axis=0)
    coverage = np.ptp(source_xy[inliers], axis=0) / np.maximum(bbox, 1)
    result["seed_coverage"] = round(float(np.min(coverage)), 4)
    projected = cv2.perspectiveTransform(quad.astype(np.float32)[None], matrix)[0]
    target_height, target_width = target["shape"]
    in_frame = bool(
        (projected[:, 0] >= 0).all()
        and (projected[:, 0] < target_width).all()
        and (projected[:, 1] >= 0).all()
        and (projected[:, 1] < target_height).all()
    )
    result["supported"] = bool(
        result["inliers"] >= 12
        and result["inlier_fraction"] >= 0.45
        and result["seed_coverage"] >= 0.35
        and in_frame
    )
    return result


def _progress(done: int, total: int) -> None:
    filled = round(24 * done / total)
    print(
        f"\r[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} 多视角照片审计",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
