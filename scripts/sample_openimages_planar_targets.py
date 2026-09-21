#!/usr/bin/env python3
"""从 Open Images 官方验证集抽取平面目标照片，供逐图许可与四角复核。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/sample_openimages_planar_targets.py \
      --data-root /Users/caozichen/screenrestore-data \
      --target-class Poster \
      --output-directory /Users/caozichen/screenrestore-data/geometry/openimages-poster-review

运行前把官方 validation-annotations-bbox.csv 和 validation-images-with-rotation.csv
放在 data-root/backgrounds/openimages/metadata。脚本只下载人工框对应的预览缩略图，
生成标注提案图板和来源索引；验证集框文件不含原始极值点，原始 Flickr 许可、画面内容、
隐私和精确 content 四角仍待逐图核验。
这些候选不能直接进入训练或验收。下载有单图与总量上限，输出目录不能覆盖。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np

CLASS_IDS = {
    "Poster": "/m/01n5jq",
    "Picture frame": "/m/06z37_",
    "Book": "/m/0bt_c3",
    "Television": "/m/07c52",
    "Billboard": "/m/01knjb",
}
LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"
BOXES_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
IMAGES_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-class", choices=tuple(CLASS_IDS), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--max-download-mib", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出必须位于 data-root 下的新目录，禁止覆盖")
    if args.max_images < 1 or args.max_download_mib <= 0:
        raise ValueError("max-images 与 max-download-mib 必须为正数")
    metadata = root / "backgrounds" / "openimages" / "metadata"
    boxes_path = _public_file(metadata / "validation-annotations-bbox.csv", root)
    images_path = _public_file(metadata / "validation-images-with-rotation.csv", root)
    boxes = _eligible_boxes(boxes_path, CLASS_IDS[args.target_class])
    image_info = _image_metadata(images_path, set(boxes))
    selected = [ident for ident in boxes if ident in image_info]
    selected.sort(key=lambda ident: hashlib.sha256(f"{args.seed}:{ident}".encode()).digest())
    selected = selected[: args.max_images]
    if not selected:
        raise ValueError("没有同时满足人工框、来源许可与方向要求的照片")
    output.mkdir(parents=True)
    image_dir = output / "images"
    image_dir.mkdir()
    records: list[dict[str, object]] = []
    downloaded_bytes = 0
    for index, ident in enumerate(selected, 1):
        item = image_info[ident]
        thumbnail_url = item["Thumbnail300KURL"]
        record: dict[str, object] = {
            "image_id": ident,
            "target_class_proposal": args.target_class,
            "boxes": boxes[ident],
            "landing_url": item["OriginalLandingURL"],
            "photo_license_metadata": item["License"],
            "author": item["Author"],
            "title": item["Title"],
            "original_url": item["OriginalURL"],
            "thumbnail_url": thumbnail_url,
            "license_verified_at_source": False,
            "underlying_content_rights": "unverified",
            "quad_review_status": "pending",
            "training_eligible": False,
            "download_status": "unavailable",
        }
        if _allowed_thumbnail_url(thumbnail_url):
            try:
                # 预览图仅供人工筛查；不下载原分辨率或扩大数据预算。
                with urlopen(thumbnail_url, timeout=20) as response:
                    payload = response.read(2 * 1024 * 1024 + 1)
                if len(payload) > 2 * 1024 * 1024:
                    raise ValueError("缩略图超过单张 2 MiB 上限")
                decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or min(decoded.shape[:2]) < 200:
                    raise ValueError("缩略图损坏或过小")
                downloaded_bytes += len(payload)
                if downloaded_bytes > args.max_download_mib * 1024**2:
                    raise ValueError("缩略图超过总下载上限")
                path = image_dir / f"{ident}.jpg"
                path.write_bytes(payload)
                record["thumbnail_path"] = path.relative_to(root).as_posix()
                record["thumbnail_sha256"] = hashlib.sha256(payload).hexdigest()
                record["download_status"] = "downloaded"
            except (OSError, ValueError) as exc:
                record["download_error"] = type(exc).__name__
        records.append(record)
        _progress(index, len(selected))
    boards = _write_boards(records, output, root)
    report = {
        "kind": "openimages_planar_target_review_candidates",
        "target_class": args.target_class,
        "source_split": "validation",
        "official_boxes_url": BOXES_URL,
        "official_images_metadata_url": IMAGES_URL,
        "boxes_sha256": _sha256(boxes_path),
        "image_metadata_sha256": _sha256(images_path),
        "generator_sha256": _sha256(Path(__file__)),
        "seed": args.seed,
        "candidate_count": len(records),
        "downloaded_count": sum(row["download_status"] == "downloaded" for row in records),
        "boards": boards,
        "records": records,
    }
    (output / "sample-index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": len(records), "downloaded": report["downloaded_count"],
                      "index": str(output / "sample-index.json")}, ensure_ascii=False))
    return 0


def _eligible_boxes(path: Path, target_id: str) -> dict[str, list[dict[str, object]]]:
    """筛掉遮挡、截断、群组和图中图；物体框只用作人工复核提案。"""

    output: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["LabelName"] != target_id or row["Source"] != "xclick":
                continue
            if any(row[field] != "0" for field in (
                "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction"
            )):
                continue
            bbox = [float(row[field]) for field in ("XMin", "YMin", "XMax", "YMax")]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if not 0.04 <= area <= 0.85 or not all(0 <= value <= 1 for value in bbox):
                continue
            output[row["ImageID"]].append({"bbox": bbox})
    return dict(output)


def _image_metadata(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    """只收录标为 CC BY 2.0、无需旋转且有可追溯原页的照片。"""

    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ident = row["ImageID"]
            if ident not in wanted:
                continue
            if (
                row["License"] != LICENSE_URL
                or row["Rotation"] not in ("0", "0.0")
                or not row["OriginalLandingURL"].startswith("https://www.flickr.com/photos/")
            ):
                continue
            output[ident] = row
    return output


def _allowed_thumbnail_url(value: str) -> bool:
    url = urlparse(value)
    return url.scheme == "https" and (
        url.hostname == "staticflickr.com" or str(url.hostname).endswith(".staticflickr.com")
    )


def _write_boards(records: list[dict[str, object]], output: Path, root: Path) -> list[str]:
    """每页最多 20 张，蓝框展示上游物体框而非已批准四角。"""

    tiles: list[np.ndarray] = []
    boards = []
    for row in records:
        if row["download_status"] != "downloaded":
            continue
        image = cv2.imread(str(root / str(row["thumbnail_path"])))
        if image is None:
            continue
        height, width = image.shape[:2]
        for box in row["boxes"]:
            coords = np.asarray(box["bbox"], np.float32) * [width - 1, height - 1, width - 1, height - 1]
            x1, y1, x2, y2 = np.rint(coords).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 170, 0), 2)
        tile = np.full((270, 320, 3), 238, np.uint8)
        scale = min(320 / width, 242 / height)
        preview = cv2.resize(image, (round(width * scale), round(height * scale)))
        tile[: preview.shape[0], : preview.shape[1]] = preview
        cv2.putText(tile, str(row["image_id"])[:16], (4, 261), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    for start in range(0, len(tiles), 20):
        page = tiles[start : start + 20]
        while len(page) % 4:
            page.append(np.full((270, 320, 3), 238, np.uint8))
        board = np.concatenate(
            [np.concatenate(page[index : index + 4], axis=1) for index in range(0, len(page), 4)],
            axis=0,
        )
        name = f"review-board-{start // 20 + 1:02d}.jpg"
        if not cv2.imwrite(str(output / name), board, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"无法写入复核图板：{name}")
        boards.append(name)
    return boards


def _public_file(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root) or any(
        "private" in part.lower() for part in resolved.relative_to(root).parts
    ) or not resolved.is_file():
        raise ValueError(f"Open Images 元数据路径不属于公开 data-root：{path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(done: int, total: int) -> None:
    if done < total and done % max(1, total // 40):
        return
    filled = round(24 * done / total)
    print(f"\r[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} Open Images 预览筛查",
          end="\n" if done == total else "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
