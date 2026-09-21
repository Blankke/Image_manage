#!/usr/bin/env python3
"""用真实手机拍摄的 SmartDoc 纸面制作公开杂志封面几何样本。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/prepare_photo_surface_geometry.py \
      --data-root /Users/caozichen/screenrestore-data \
      --base-manifest /Users/caozichen/screenrestore-data/manifests/p8/target-v6-v7-v8-replay-20260916.geometry.jsonl \
      --artwork-manifest /Users/caozichen/screenrestore-data/manifests/p9/syngallery-offset1000-reviewed-20260917.geometry.jsonl \
      --artwork-manifest /Users/caozichen/screenrestore-data/manifests/p9/syngallery-offset2000-reviewed-20260917.geometry.jsonl \
      --output-directory /Users/caozichen/screenrestore-data/geometry/photo-surface-v2 \
      --frames-per-model 3

只复用已有本地公开数据。SmartDoc 原图保留桌面、透视、模糊与光照；纸面内替换为
Met Open Access 画作加程序排版，避免把原始文档文字混入新标签。SmartDoc
SmartDoc 背景跨文档复用，所有生成图只进入 train，且只使用原始 train 文档与
train 画作；原始 validation/test 文档不生成。独立验证应使用别的拍摄场景。
生成图、清单和来源都留在 data-root，不进入仓库。程序显示进度条且拒绝覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from screenrestore.io.geometry_isolation import SMARTDOC_SCENE_FAMILY

BACKGROUNDS = frozenset(f"background{index:02d}" for index in range(1, 6))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--artwork-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--frames-per-model", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出须位于 data-root 内的新目录，禁止覆盖")
    if not 1 <= args.frames_per_model <= 12:
        raise ValueError("frames-per-model 必须位于 1..12")
    base = _public_file(args.base_manifest, root)
    assets = _artwork_assets(args.artwork_manifest, root)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in base.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("source") != "smartdoc" or not record.get("present"):
            continue
        parts = Path(str(record["image"])).parts
        if len(parts) < 6 or parts[:3] != ("geometry", "smartdoc", "frames"):
            raise ValueError("SmartDoc 图片路径格式不符合预期")
        background, model = parts[3:5]
        if background not in BACKGROUNDS:
            raise ValueError(f"未知 SmartDoc 背景：{background}")
        groups[(background, model)].append(record)
    model_splits, development_groups = _development_groups(groups)
    if len(groups) != 150 or len(model_splits) != 30:
        raise ValueError(f"预期五个背景×30 种文档，实际 {len(groups)} 组")
    output.mkdir(parents=True)
    image_directory = output / "images"
    image_directory.mkdir()
    manifest = output / "manifest.jsonl"
    total = len(development_groups) * args.frames_per_model
    completed = 0
    written: list[dict] = []
    for background, model in development_groups:
        split = model_splits[model]
        pool = assets
        digest = hashlib.sha256(f"{args.seed}:{background}:{model}".encode()).digest()
        artwork_id, artwork_path = pool[int.from_bytes(digest[:8], "big") % len(pool)]
        # 一个拍摄序列只绑定一个公开画作，连拍不会因内容变换跨 split。
        source_bgr = cv2.imread(str(artwork_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise OSError(f"无法读取来源画作：{artwork_path}")
        frames = sorted(groups[(background, model)], key=lambda row: row["image"])
        indices = np.linspace(0, len(frames) - 1, args.frames_per_model, dtype=int)
        if len(set(indices)) != args.frames_per_model:
            raise ValueError(f"拍摄序列帧数不足：{background}/{model}")
        for frame_number, index in enumerate(indices, start=1):
            row = frames[int(index)]
            path = _public_file(root / str(row["image"]), root)
            photo_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if photo_bgr is None:
                raise OSError(f"无法读取手机照片：{path}")
            height, width = photo_bgr.shape[:2]
            content_quad = np.asarray(row["content_quad"], np.float32)
            if content_quad.shape != (4, 2) or not np.isfinite(content_quad).all():
                raise ValueError(f"无效 SmartDoc 四角：{path}")
            quad_px = content_quad * np.asarray([width - 1, height - 1], np.float32)
            # 同一作品保持一致排版；拍摄造成的光度变化从各帧原始纸面读取。
            cover_rng = np.random.default_rng(int.from_bytes(digest[8:16], "big"))
            cover = _make_cover(source_bgr, cover_rng)
            composite = _replace_page(photo_bgr, quad_px, cover)
            relative = Path("images") / f"{background}-{model}-{frame_number:02d}.jpg"
            target = output / relative
            if not cv2.imwrite(str(target), composite, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise OSError(f"无法写入合成图：{target}")
            # 原始文档模型横跨五个背景，必须沿用其同一分组身份。
            group = f"smartdoc:{model}"
            capture_session = f"photo-surface:{background}:{model}:{artwork_id}"
            written.append(
                {
                    "image": target.relative_to(root).as_posix(),
                    "split": split,
                    "group_id": group,
                    "capture_session": capture_session,
                    "scene_group_id": SMARTDOC_SCENE_FAMILY,
                    "device": row.get("device", "smartdoc-phone"),
                    "present": True,
                    # 类别由实体目标决定：杂志纸面沿用 SmartDoc/postcard 契约。
                    "target_class": "postcard",
                    "content_quad": row["content_quad"],
                    "outer_quad": None,
                    "visible": True,
                    "occlusion": 0.0,
                    "glare_level": "none",
                    "source": "smartdoc-photo-surface-composite",
                    "scene_type": "phone_magazine_cover_surface",
                    "domain": "postcard",
                    "subject_id": f"smartdoc:{model}",
                    "in_scope": True,
                    "ambiguous": False,
                    "digital_source_id": f"met-open-access-{artwork_id}",
                    "source_capture_id": str(row["image"]),
                    "source_license": "SmartDoc CC-BY-4.0; Met Open Access CC0",
                    "source_adaptation": "SmartDoc 纸面替换为 Met Open Access 画作及程序排版",
                }
            )
            completed += 1
            _progress(completed, total)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written),
        encoding="utf-8",
    )
    # 记录输入与生成规则的摘要，后续训练和评分可回溯到同一批数据。
    provenance = {
        "kind": "photo_surface_generation",
        "sample_count": len(written),
        "capture_session_count": len(development_groups),
        "scene_group_id": SMARTDOC_SCENE_FAMILY,
        "seed": args.seed,
        "frames_per_model": args.frames_per_model,
        "generator_sha256": _sha256(Path(__file__)),
        "base_manifest_sha256": _sha256(base),
        "artwork_manifest_sha256": {
            str(path.expanduser().resolve()): _sha256(_public_file(path, root))
            for path in args.artwork_manifest
        },
        "output_manifest_sha256": _sha256(manifest),
        "adaptation": "SmartDoc 纸面替换为 Met Open Access 画作及程序排版",
        "attribution": "SmartDoc: Burie 等，ICDAR 2015；Met Open Access CC0 作品按 digital_source_id 追踪",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"samples": len(written), "groups": len(development_groups), "manifest": str(manifest)}, ensure_ascii=False))
    return 0


def _public_file(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (
        not resolved.is_relative_to(root)
        or any("private" in part.lower() for part in resolved.relative_to(root).parts)
        or not resolved.is_file()
    ):
        raise ValueError(f"公开数据文件不存在或路径越界：{path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_groups(
    groups: dict[tuple[str, str], list[dict]],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """只选原始 train 文档；所有复用背景的衍生图也只进入 train。"""

    model_splits: dict[str, str] = {}
    for (_, model), rows in groups.items():
        splits = {str(row["split"]) for row in rows}
        if len(splits) != 1 or (model in model_splits and model_splits[model] not in splits):
            raise ValueError(f"SmartDoc 文档模型跨 split：{model}")
        model_splits[model] = splits.pop()
    # 同一背景在所有文档间重复，validation/test 文档不能提供独立场景验证。
    development = [
        (background, model) for background, model in sorted(groups)
        if model_splits[model] == "train"
    ]
    return model_splits, development


def _artwork_assets(manifests: list[Path], root: Path) -> list[tuple[int, Path]]:
    train_assets: dict[int, Path] = {}
    split_by_id: dict[int, str] = {}
    for manifest in manifests:
        for line in _public_file(manifest, root).open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("source") != "syngallery-reviewed":
                raise ValueError("只接受视觉复核通过的 SynGallery 来源清单")
            artwork_id = int(str(row["digital_source_id"]).removeprefix("met-open-access-"))
            split = str(row["split"])
            if split not in {"train", "validation", "test"} or (
                artwork_id in split_by_id and split_by_id[artwork_id] != split
            ):
                raise ValueError("Met 作品跨 split 或 split 无效")
            split_by_id[artwork_id] = split
            if split != "train":
                continue
            directory = (root / str(row["image"])).resolve().parent.parent
            source = _public_file(directory / "sources" / f"met-{artwork_id}.jpg", root)
            train_assets[artwork_id] = source
    if not train_assets:
        raise ValueError("需要至少一个已复核的 train 公开画作")
    return sorted(train_assets.items())


def _make_cover(source_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在整页 CC0 画作上排字，避免固定色块把封面误分为嵌套内容层。"""

    height, width = 900, 620
    sh, sw = source_bgr.shape[:2]
    crop_width = min(sw, max(1, round(sh * width / height)))
    crop_height = min(sh, max(1, round(sw * height / width)))
    left = (sw - crop_width) // 2
    top = (sh - crop_height) // 2
    cover = cv2.resize(
        source_bgr[top : top + crop_height, left : left + crop_width],
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    # 用两套排字位置和明暗搭配覆盖不同封面布局；四角均保留原始图像纹理。
    variant = int(rng.integers(0, 3))
    masthead = ("VISUAL", "GALLERY", "ARTBOOK")[variant]
    top_luma = float(cv2.cvtColor(cover[:145], cv2.COLOR_BGR2GRAY).mean())
    bottom_luma = float(cv2.cvtColor(cover[-130:], cv2.COLOR_BGR2GRAY).mean())
    top_color = (25, 25, 25) if top_luma > 150 else (246, 246, 246)
    bottom_color = (25, 25, 25) if bottom_luma > 150 else (246, 246, 246)
    top_outline = (246, 246, 246) if top_luma > 150 else (25, 25, 25)
    bottom_outline = (246, 246, 246) if bottom_luma > 150 else (25, 25, 25)
    title_x = (25, 14, 22)[variant]
    title_size = (2.75, 2.55, 2.75)[variant]
    cv2.putText(cover, masthead, (title_x, 104), cv2.FONT_HERSHEY_DUPLEX,
                title_size, top_outline, 7, cv2.LINE_AA)
    cv2.putText(cover, masthead, (title_x, 104), cv2.FONT_HERSHEY_DUPLEX,
                title_size, top_color, 4, cv2.LINE_AA)
    cv2.putText(cover, "ART  /  ISSUE  01", (30, height - 65), cv2.FONT_HERSHEY_DUPLEX,
                0.85, bottom_outline, 4, cv2.LINE_AA)
    cv2.putText(cover, "ART  /  ISSUE  01", (30, height - 65), cv2.FONT_HERSHEY_DUPLEX,
                0.85, bottom_color, 2, cv2.LINE_AA)
    cv2.putText(cover, "COLLECTION", (30, height - 26), cv2.FONT_HERSHEY_DUPLEX,
                1.0, bottom_outline, 4, cv2.LINE_AA)
    cv2.putText(cover, "COLLECTION", (30, height - 26), cv2.FONT_HERSHEY_DUPLEX,
                1.0, bottom_color, 2, cv2.LINE_AA)
    return cover


def _replace_page(photo_bgr: np.ndarray, quad_px: np.ndarray, cover_bgr: np.ndarray) -> np.ndarray:
    """在真实纸面单应映射封面，并保留大尺度真实照明场。"""

    ph, pw = photo_bgr.shape[:2]
    ch, cw = cover_bgr.shape[:2]
    canonical = np.asarray([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(canonical, quad_px.astype(np.float32))
    inverse = cv2.getPerspectiveTransform(quad_px.astype(np.float32), canonical)
    original_page = cv2.warpPerspective(photo_bgr, inverse, (cw, ch))
    gray = cv2.cvtColor(original_page, cv2.COLOR_BGR2GRAY).astype(np.float32)
    light = cv2.GaussianBlur(gray, (0, 0), 48)
    median = max(35.0, float(np.median(light)))
    illumination = np.clip(light / median, 0.72, 1.20)
    lit_cover = np.clip(cover_bgr.astype(np.float32) * illumination[:, :, None], 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(lit_cover, matrix, (pw, ph))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), matrix, (pw, ph))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.7)[:, :, None]
    return np.clip(photo_bgr * (1 - alpha) + warped * alpha, 0, 255).astype(np.uint8)


def _progress(done: int, total: int) -> None:
    interval = max(1, total // 100)
    if done < total and done % interval:
        return
    filled = round(24 * done / total)
    print(
        f"\r[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} 真实拍摄纸面合成",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
