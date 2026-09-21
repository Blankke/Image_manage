#!/usr/bin/env python3
"""抽样核验 SynGallery 的来源画作与展厅视图，生成待复核 content 四角。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/prepare_syngallery_geometry.py \
      --output-directory /Users/caozichen/screenrestore-data/geometry/syngallery-pilot \
      --offset 0 --count 20 --view 60 --view 90 --view 120
    python scripts/prepare_syngallery_geometry.py \
      --output-directory /Users/caozichen/screenrestore-data/geometry/syngallery-pilot \
      --review-only

仅访问官方公开 Dataset Viewer API；不依赖 Parquet 工具。输出 candidate-manifest.jsonl
和叠加图供人工复核，不自动进入训练。数据集渲染图许可为 CC BY 4.0，源画作来自
The Met Open Access（CC0）；使用时须保留来源和署名。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

DATASET = "patryk-bartkowiak/SynGallery"
DATASET_PAGE = "https://huggingface.co/datasets/patryk-bartkowiak/SynGallery"
DATASET_REVISION = "278af15f4cfc4f3c4debb088b5b8078a515748d7"
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--view", type=int, action="append", choices=(30, 60, 90, 120, 150))
    parser.add_argument("--review-only", action="store_true", help="只为既有候选绘制审核拼图")
    args = parser.parse_args(argv)
    output = args.output_directory.expanduser().resolve()
    if args.review_only:
        manifest = output / "candidate-manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"候选清单不存在：{manifest}")
        candidates = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        _write_review_boards(output, candidates)
        return 0
    if args.offset < 0 or not 1 <= args.count <= 100:
        raise ValueError("offset 必须非负，count 必须位于 1..100")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖现有样本：{output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "sources").mkdir()
    (output / "overlays").mkdir()
    views = tuple(args.view or (60, 90, 120))
    rows = _dataset_rows(args.offset, args.count)
    candidates: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    total = len(rows) * len(views)
    completed = 0
    for item in rows:
        row = item["row"]
        index = int(row["index"])
        object_id = int(row["met_object_id"])
        if not row.get("is_public_domain"):
            for view in views:
                outcomes.append({"index": index, "view": view, "reason": "not_public_domain"})
                completed += 1
                _progress(completed, total)
            continue
        source_rgb = _download_rgb(str(row["source_image"]["src"]))
        source_path = output / "sources" / f"met-{object_id}.jpg"
        _save_rgb(source_path, source_rgb)
        full_bleed, source_evidence = source_full_bleed_evidence(source_rgb)
        if not full_bleed:
            # Met 原图可能拍到灰底、实体画框或非矩形扇面；其整张图片边界不能
            # 当成画芯四角，故连渲染视角都不下载，防止错误层级进入训练。
            for view in views:
                outcomes.append(
                    {
                        "index": index,
                        "met_object_id": object_id,
                        "view": view,
                        "reason": "source_not_full_bleed",
                        **source_evidence,
                    }
                )
                completed += 1
                _progress(completed, total)
            continue
        for view in views:
            render_rgb = _download_rgb(str(row[f"image_{view}"]["src"]))
            image_path = output / "images" / f"met-{object_id}-view-{view}.jpg"
            _save_rgb(image_path, render_rgb)
            quad, evidence = propose_content_quad(source_rgb, render_rgb)
            outcome: dict[str, object] = {
                "index": index,
                "met_object_id": object_id,
                "view": view,
                "reason": evidence["reason"],
                "matches": evidence["matches"],
                "inliers": evidence["inliers"],
                **source_evidence,
            }
            if quad is not None:
                overlay = render_rgb.copy()
                cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (255, 40, 40), 3)
                _save_rgb(output / "overlays" / image_path.name, overlay)
                height, width = render_rgb.shape[:2]
                # 同一 Met 对象的五个视角只能属于同一 split，防止作品身份泄漏。
                bucket = int(hashlib.sha256(str(object_id).encode()).hexdigest()[:8], 16) % 10
                split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
                candidates.append(
                    {
                        "image": image_path.relative_to(output).as_posix(),
                        "source_image": source_path.relative_to(output).as_posix(),
                        "source": "syngallery-provisional",
                        "source_url": DATASET_PAGE,
                        "source_revision": DATASET_REVISION,
                        "license": "CC-BY-4.0",
                        "met_object_id": object_id,
                        "view": view,
                        "split": split,
                        "group_id": f"syngallery-met-{object_id}",
                        "capture_session": f"syngallery-met-{object_id}",
                        "target_class": "artwork",
                        "content_quad": (quad / np.asarray([width - 1, height - 1])).round(7).tolist(),
                        "outer_quad": None,
                        "label_status": "provisional_match_requires_review",
                        "matching_evidence": evidence,
                        "source_evidence": source_evidence,
                    }
                )
            outcomes.append(outcome)
            completed += 1
            _progress(completed, total)
    (output / "candidate-manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates),
        encoding="utf-8",
    )
    (output / "matching-outcomes.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in outcomes),
        encoding="utf-8",
    )
    _write_review_boards(output, candidates)
    print(f"候选四角 {len(candidates)}/{total}；待人工复核：{output}")
    return 0


def source_full_bleed_evidence(source_rgb: np.ndarray) -> tuple[bool, dict[str, float]]:
    """保守排除明显拍到统一底色/实体外框的 Met 来源图。"""

    height, width = source_rgb.shape[:2]
    span = max(8, round(min(height, width) * 0.08))
    corners = (
        source_rgb[:span, :span],
        source_rgb[:span, -span:],
        source_rgb[-span:, -span:],
        source_rgb[-span:, :span],
    )
    means = np.asarray([corner.mean(axis=(0, 1)) for corner in corners])
    corner_texture = max(float(np.std(corner)) for corner in corners)
    color_spread = float(np.max(np.ptp(means, axis=0)))
    evidence = {
        "source_corner_texture": round(corner_texture, 3),
        "source_corner_color_spread": round(color_spread, 3),
    }
    return corner_texture >= 28.0 and color_spread >= 24.0, evidence


def propose_content_quad(
    source_rgb: np.ndarray, render_rgb: np.ndarray
) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    """用公开同源图像求单应矩阵；低覆盖、裁切或重投影不稳时拒绝。"""

    detector = cv2.SIFT_create(nfeatures=1800)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    source_points, source_desc = detector.detectAndCompute(source_gray, None)
    render_points, render_desc = detector.detectAndCompute(render_gray, None)
    evidence: dict[str, float | int | str] = {"reason": "insufficient_features", "matches": 0, "inliers": 0}
    if source_desc is None or render_desc is None:
        return None, evidence
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(source_desc, render_desc, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    evidence["matches"] = len(matches)
    if len(matches) < 24:
        return None, evidence
    source_xy = np.asarray([source_points[item.queryIdx].pt for item in matches], np.float32)
    render_xy = np.asarray([render_points[item.trainIdx].pt for item in matches], np.float32)
    matrix, inlier_mask = cv2.findHomography(source_xy, render_xy, cv2.RANSAC, 3.0)
    if matrix is None or inlier_mask is None:
        evidence["reason"] = "homography_failed"
        return None, evidence
    inliers = inlier_mask.ravel().astype(bool)
    evidence["inliers"] = int(inliers.sum())
    if inliers.sum() < 20 or float(inliers.mean()) < 0.45:
        evidence["reason"] = "weak_consensus"
        return None, evidence
    source_height, source_width = source_rgb.shape[:2]
    covered = np.ptp(source_xy[inliers], axis=0) / np.asarray([source_width, source_height])
    if float(np.min(covered)) < 0.45:
        evidence["reason"] = "source_coverage_low"
        return None, evidence
    corners = np.asarray(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]],
        np.float32,
    )
    quad = cv2.perspectiveTransform(corners[None], matrix)[0]
    height, width = render_rgb.shape[:2]
    area = abs(float(cv2.contourArea(quad))) / float(width * height)
    if (
        not np.isfinite(quad).all()
        or np.min(quad[:, 0]) < 5
        or np.max(quad[:, 0]) > width - 6
        or np.min(quad[:, 1]) < 5
        or np.max(quad[:, 1]) > height - 6
        or not 0.05 <= area <= 0.85
        or not cv2.isContourConvex(np.rint(quad).astype(np.int32))
    ):
        evidence["reason"] = "incomplete_or_invalid_quad"
        return None, evidence
    projected = cv2.perspectiveTransform(source_xy[inliers][None], matrix)[0]
    reprojection = np.linalg.norm(projected - render_xy[inliers], axis=1)
    median_error = float(np.median(reprojection))
    evidence["reprojection_median_px"] = round(median_error, 4)
    evidence["source_coverage_min"] = round(float(np.min(covered)), 4)
    if median_error > 2.5:
        evidence["reason"] = "reprojection_unstable"
        return None, evidence
    evidence["reason"] = "provisional_match"
    return quad.astype(np.float32), evidence


def _dataset_rows(offset: int, count: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": count}
    )
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=30) as response:
        observed_revision = response.headers.get("X-Revision")
        if observed_revision != DATASET_REVISION:
            raise ValueError(
                f"SynGallery 版本已变化：预期 {DATASET_REVISION}，实得 {observed_revision}"
            )
        payload = json.load(response)
    rows = payload["rows"]
    if len(rows) != count:
        raise ValueError(f"官方 API 只返回 {len(rows)}/{count} 行；拒绝静默截断")
    return rows


def _download_rgb(url: str) -> np.ndarray:
    if not url.startswith("https://datasets-server.huggingface.co/cached-assets/"):
        raise ValueError("只允许 Dataset Viewer 官方缓存图像 URL")
    if f"/--/{DATASET_REVISION}/--/" not in url:
        raise ValueError("缓存图像版本与已核查数据卡不一致")
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("单张图像超过 4 MiB 预算")
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path, quality=93)


def _write_review_boards(output: Path, candidates: list[dict[str, object]]) -> None:
    """将匹配叠加图做成分页面板，便于逐一核查是否选中了真正画芯。"""

    page_size = 24
    for page_number, start in enumerate(range(0, len(candidates), page_size), start=1):
        destination = output / f"review-board-{page_number:02d}.jpg"
        if destination.exists():
            raise FileExistsError(f"拒绝覆盖已有审核面板：{destination}")
        board = Image.new("RGB", (4 * 260, 6 * 280), (34, 38, 43))
        draw = ImageDraw.Draw(board)
        for position, record in enumerate(candidates[start : start + page_size]):
            path = output / "overlays" / Path(str(record["image"])).name
            with Image.open(path) as image:
                tile = image.convert("RGB").resize((256, 256))
            x = position % 4 * 260
            y = position // 4 * 280
            board.paste(tile, (x, y))
            draw.text(
                (x + 4, y + 260),
                f"Met {record['met_object_id']} / {record['view']}",
                fill=(235, 235, 235),
            )
        board.save(destination, quality=90)
        print(f"审核面板 {page_number}: {destination}")


def _progress(done: int, total: int) -> None:
    filled = round(24 * done / total)
    print(f"\r[{'#' * filled}{'-' * (24 - filled)}] {done}/{total}", end="\n" if done == total else "", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
