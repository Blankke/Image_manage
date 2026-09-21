#!/usr/bin/env python3
"""把视觉复核的 DOCCI 真实桌面照片制成公开杂志几何训练样本。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/prepare_docci_desk_geometry.py \
      --data-root /Users/caozichen/screenrestore-data \
      --background-index datasets/manifests/docci_desk_backgrounds.json \
      --artwork-manifest /Users/caozichen/screenrestore-data/manifests/p9/syngallery-offset1000-reviewed-20260917.geometry.jsonl \
      --artwork-manifest /Users/caozichen/screenrestore-data/manifests/p9/syngallery-offset2000-reviewed-20260917.geometry.jsonl \
      --output-directory /Users/caozichen/screenrestore-data/geometry/docci-desk-magazine-pilot \
      --samples-per-photo 8

只生成 train split；DOCCI 照片保守地共享同一个来源家族 group_id。官方 cluster_id
是内容分类，不能代表独立拍摄场景。下载的官方缩略图、
合成图和哈希清单都留在 data-root。纸面完整覆盖原背景中心的小物体，几何标签由
实际单应矩阵生成；脚本拒绝覆盖既有产物，并显示进度条。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.prepare_photo_surface_geometry import (  # noqa: E402
    _artwork_assets,
    _make_cover,
    _public_file,
)

THUMBNAIL_BASE = "https://storage.googleapis.com/docci/thumbnails"
METADATA_URL = "https://google.github.io/docci/web-data/docci_data.jsonlines"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--background-index", type=Path, required=True)
    parser.add_argument("--artwork-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--samples-per-photo", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-output-gib", type=float, default=0.5)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出必须位于 data-root 下的新目录，禁止覆盖")
    if not 1 <= args.samples_per_photo <= 32 or args.max_output_gib <= 0:
        raise ValueError("samples-per-photo 或 max-output-gib 无效")
    index_path = args.background_index.expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    photos = _validated_photos(index)
    metadata_path = root / "backgrounds" / "docci" / "docci_web_data.jsonlines"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        with urlopen(METADATA_URL, timeout=60) as response:
            payload = response.read(20 * 1024 * 1024 + 1)
        if len(payload) > 20 * 1024 * 1024:
            raise ValueError("DOCCI 官方元数据超过下载上限")
        metadata_path.write_bytes(payload)
    _verify_clusters(photos, metadata_path)
    assets = _artwork_assets(args.artwork_manifest, root)
    cache = root / "backgrounds" / "docci" / "reviewed-desk-thumbnails"
    cache.mkdir(parents=True, exist_ok=True)
    source_paths: dict[str, Path] = {}
    for number, photo in enumerate(photos, 1):
        ident = photo["example_id"]
        path = cache / f"{ident}.jpg"
        if not path.exists():
            # 官方浏览器直接提供 768×1024 缩略图；单张上限防止意外下载归档包。
            with urlopen(f"{THUMBNAIL_BASE}/{ident}.jpg", timeout=30) as response:
                payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError(f"DOCCI 缩略图超过单张上限：{ident}")
            path.write_bytes(payload)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 640:
            raise ValueError(f"DOCCI 缩略图损坏或分辨率不足：{ident}")
        source_paths[ident] = _public_file(path, root)
        _progress(number, len(photos), "核验 DOCCI 桌面照片")

    output.mkdir(parents=True)
    image_dir = output / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []
    asset_hashes: dict[str, str] = {}
    written_bytes = 0
    total = len(photos) * args.samples_per_photo
    for photo_index, photo in enumerate(photos):
        ident = photo["example_id"]
        background_bgr = cv2.imread(str(source_paths[ident]), cv2.IMREAD_COLOR)
        if background_bgr is None:
            raise OSError(f"无法读取 DOCCI 照片：{ident}")
        for sample_index in range(args.samples_per_photo):
            artwork_id, artwork_path = assets[int(rng.integers(0, len(assets)))]
            artwork_bgr = cv2.imread(str(artwork_path), cv2.IMREAD_COLOR)
            if artwork_bgr is None:
                raise OSError(f"无法读取公开画作：{artwork_path}")
            cover = _make_cover(artwork_bgr, rng)
            height, width = background_bgr.shape[:2]
            quad = _sample_quad(rng, width, height, distant=sample_index % 3 == 0)
            composite = _compose_paper(background_bgr, cover, quad, rng)
            relative = Path("images") / f"{ident}-{sample_index:02d}.jpg"
            target = output / relative
            if not cv2.imwrite(str(target), composite, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise OSError(f"无法写入合成图：{target}")
            written_bytes += target.stat().st_size
            if written_bytes > args.max_output_gib * 1024**3:
                raise RuntimeError("合成图超过输出空间预算，保留已有文件供检查")
            asset_hashes[str(artwork_id)] = _sha256(artwork_path)
            records.append({
                "image": target.relative_to(root).as_posix(),
                "split": "train",
                "group_id": "docci-photo-family",
                "capture_session": f"docci-desk:{ident}",
                "device": "docci-camera-unknown",
                "present": True,
                "target_class": "postcard",
                "content_quad": (quad / np.array([width - 1, height - 1], np.float32)).tolist(),
                "outer_quad": None,
                "visible": True,
                "occlusion": 0.0,
                "glare_level": "none",
                "source": "docci-desk-magazine-composite",
                "scene_type": "real_desk_magazine_cover",
                "domain": "postcard",
                "subject_id": f"met-open-access-{artwork_id}",
                "in_scope": True,
                "ambiguous": False,
                "digital_source_id": f"met-open-access-{artwork_id}",
                "source_capture_id": ident,
                "source_license": "DOCCI photo CC BY 4.0; Met Open Access artwork CC0",
                "source_adaptation": "DOCCI 真实桌面照片上合成完整纸面及摄影阴影",
            })
            _progress(photo_index * args.samples_per_photo + sample_index + 1, total, "生成真实桌面杂志")
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    provenance = {
        "kind": "docci_desk_magazine_geometry",
        "source_page": index["source_page"],
        "source_license": index["source_license"],
        "attribution": "Onoe 等，DOCCI，ECCV 2024；Met Open Access CC0",
        "seed": args.seed,
        "samples_per_photo": args.samples_per_photo,
        "sample_count": len(records),
        "capture_group_count": 1,
        "metadata_category_cluster_count": len({photo["cluster_id"] for photo in photos}),
        "metadata_category_cluster_by_capture": {
            photo["example_id"]: photo["cluster_id"] for photo in photos
        },
        "generator_sha256": _sha256(Path(__file__)),
        "background_index_sha256": _sha256(index_path),
        "docci_metadata_sha256": _sha256(metadata_path),
        "background_photo_sha256": {
            ident: _sha256(path) for ident, path in sorted(source_paths.items())
        },
        "artwork_sha256": asset_hashes,
        "manifest_sha256": _sha256(manifest),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sample_count": len(records), "group_count": provenance["capture_group_count"],
                      "manifest": str(manifest)}, ensure_ascii=False))
    return 0


def _validated_photos(index: dict) -> list[dict[str, str]]:
    """只接受人工审查过的 DOCCI train 照片和显式簇标识。"""

    if (
        index.get("dataset") != "DOCCI"
        or index.get("source_split") != "train"
        or index.get("source_license") != "CC BY 4.0"
    ):
        raise ValueError("DOCCI 背景索引缺少来源、训练分区或许可记录")
    photos = index.get("photos")
    if not isinstance(photos, list) or not photos:
        raise ValueError("DOCCI 背景索引缺少照片")
    ids: set[str] = set()
    for photo in photos:
        if not isinstance(photo, dict):
            raise ValueError("DOCCI 照片记录必须是对象")
        ident = str(photo.get("example_id", ""))
        cluster = str(photo.get("cluster_id", ""))
        if not ident.startswith("train_") or not ident[6:].isdigit() or not cluster.isdigit():
            raise ValueError("DOCCI 照片必须来自 train 且有簇 ID")
        if ident in ids:
            raise ValueError(f"重复 DOCCI 照片：{ident}")
        ids.add(ident)
    return photos


def _verify_clusters(photos: list[dict[str, str]], metadata_path: Path) -> None:
    """用 DOCCI 官方可视化元数据验证人工索引中的内容类别簇。"""

    selected = {photo["example_id"]: photo["cluster_id"] for photo in photos}
    observed: dict[str, str] = {}
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ident = str(row.get("example_id", ""))
            if ident in selected:
                observed[ident] = str(row.get("cluster_id", ""))
    if observed != selected:
        raise ValueError("DOCCI 人工索引与官方照片簇元数据不一致")


def _sample_quad(
    rng: np.random.Generator, width: int, height: int, *, distant: bool
) -> np.ndarray:
    """在原始缩略图坐标系中生成完整可见、四角顺序固定的纸面。"""

    for _ in range(100):
        fraction = rng.uniform(0.31, 0.43) if distant else rng.uniform(0.45, 0.62)
        # 横幅照片受画面高度限制；只按宽度采样会产生无法完整放下的纵向纸张。
        height_limit = height * (0.60 if distant else 0.76) * 620.0 / 900.0
        paper_width = min(width * fraction, height_limit)
        paper_height = paper_width * 900.0 / 620.0
        center = np.array([
            width * rng.uniform(0.40, 0.60),
            height * rng.uniform(0.42, 0.58),
        ], np.float32)
        theta = float(rng.uniform(-0.27, 0.27))
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32
        )
        local = np.array([
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
        ], np.float32) * np.array([paper_width, paper_height], np.float32)
        quad = local @ rotation.T + center
        quad += rng.normal(0, paper_width * 0.018, (4, 2)).astype(np.float32)
        margin = min(width, height) * 0.045
        if (
            np.all(quad[:, 0] >= margin)
            and np.all(quad[:, 0] <= width - 1 - margin)
            and np.all(quad[:, 1] >= margin)
            and np.all(quad[:, 1] <= height - 1 - margin)
            and cv2.isContourConvex(quad)
            and cv2.contourArea(quad, oriented=True) > width * height * 0.09
        ):
            return quad
    raise RuntimeError("无法为桌面照片生成完整可见的纸面")


def _compose_paper(
    background_bgr: np.ndarray,
    cover_bgr: np.ndarray,
    quad: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """添加真实背景照明、纸面和有限软阴影；标签仍指向完整纸面边界。"""

    height, width = background_bgr.shape[:2]
    cover_height, cover_width = cover_bgr.shape[:2]
    source = np.array(
        [[0, 0], [cover_width - 1, 0], [cover_width - 1, cover_height - 1], [0, cover_height - 1]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    mask = cv2.warpPerspective(np.full((cover_height, cover_width), 255, np.uint8), transform,
                               (width, height), flags=cv2.INTER_LINEAR)
    shadow = cv2.warpAffine(
        mask,
        np.array([[1, 0, float(rng.uniform(3, 10))], [0, 1, float(rng.uniform(4, 13))]], np.float32),
        (width, height),
    )
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(2.0, width * 0.012)).astype(np.float32) / 255.0
    result = background_bgr.astype(np.float32) * (1.0 - shadow[:, :, None] * float(rng.uniform(0.14, 0.28)))
    gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    illumination = cv2.GaussianBlur(gray, (0, 0), max(14.0, width * 0.09))
    illumination = np.clip(illumination / max(35.0, float(np.median(illumination))), 0.75, 1.18)
    warped = cv2.warpPerspective(cover_bgr, transform, (width, height)).astype(np.float32)
    warped *= illumination[:, :, None] * float(rng.uniform(0.87, 1.04))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.55)[:, :, None]
    return np.clip(result * (1.0 - alpha) + warped * alpha, 0, 255).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(done: int, total: int, title: str) -> None:
    if done < total and done % max(1, total // 50):
        return
    filled = round(24 * done / max(1, total))
    print(
        f"\r[{('#' * filled) + ('-' * (24 - filled))}] {done}/{total} {title}",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
