#!/usr/bin/env python3
"""将已视觉复核的 SynGallery 候选转为隔离的公开 geometry 训练清单。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/promote_syngallery_geometry.py \
      --data-root /Users/caozichen/screenrestore-data \
      --met-metadata /Users/caozichen/screenrestore-data/textures/met-open-access/metadata.jsonl \
      --candidate-directory /Users/caozichen/screenrestore-data/geometry/syngallery-pilot-offset1000-20260917 \
      --reviewer codex-visual-qa \
      --output /Users/caozichen/screenrestore-data/manifests/p9/syngallery-reviewed.geometry.jsonl

使用前必须逐页检查 review-board-*.jpg；误标图可用重复的 --exclude-image 指定
`met-<object_id>-view-<angle>.jpg`。同一 Met 作品若已存在项目纹理池，就整个跳过，
防止同一来源图在旧合成样本和新展厅样本之间跨 split 泄漏。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_syngallery_geometry import DATASET_REVISION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--met-metadata", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, action="append", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--exclude-image", action="append", default=[])
    parser.add_argument("--light-glare-image", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.data_root.expanduser().resolve()
    metadata = args.met_metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not metadata.is_relative_to(root) or not metadata.is_file():
        raise ValueError("Met 来源 metadata 必须位于 data-root 内且已存在")
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("输出须位于 data-root 内的新路径，禁止覆盖")
    if not args.reviewer.strip():
        raise ValueError("reviewer 不能为空")
    existing_met_ids = {
        int(json.loads(line)["object_id"])
        for line in metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    directories = [path.expanduser().resolve() for path in args.candidate_directory]
    if len(set(directories)) != len(directories):
        raise ValueError("candidate-directory 不得重复")
    candidates: list[tuple[Path, dict]] = []
    board_hashes: dict[str, dict[str, str]] = {}
    for directory in directories:
        if not directory.is_relative_to(root):
            raise ValueError(f"候选数据必须在 data-root 内：{directory}")
        manifest = directory / "candidate-manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        boards = sorted(directory.glob("review-board-*.jpg"))
        if len(boards) != math.ceil(len(rows) / 24):
            raise ValueError(f"审核拼图页数不完整：{directory}")
        board_hashes[directory.relative_to(root).as_posix()] = {
            board.name: _sha256(board) for board in boards
        }
        candidates.extend((directory, row) for row in rows)
    exclusions = set(args.exclude_image)
    light_glare = set(args.light_glare_image)
    known_names = {Path(str(row["image"])).name for _, row in candidates}
    if (exclusions | light_glare) - known_names:
        raise ValueError("排除或反光覆盖清单包含未知文件名")
    records: list[dict] = []
    skipped = Counter()
    seen_images: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, (directory, candidate) in enumerate(candidates, start=1):
        image_name = Path(str(candidate["image"])).name
        if image_name in seen_images:
            raise ValueError(f"重复观察视角：{image_name}")
        seen_images.add(image_name)
        object_id = int(candidate["met_object_id"])
        if image_name in exclusions:
            skipped["visual_review_excluded"] += 1
            _progress(index, len(candidates))
            continue
        if object_id in existing_met_ids:
            skipped["met_texture_overlap"] += 1
            _progress(index, len(candidates))
            continue
        if candidate.get("source_revision") != DATASET_REVISION:
            raise ValueError(f"数据版本不匹配：{image_name}")
        evidence = candidate.get("matching_evidence", {})
        source_evidence = candidate.get("source_evidence", {})
        if (
            evidence.get("reason") != "provisional_match"
            or int(evidence.get("inliers", 0)) < 20
            or float(source_evidence.get("source_corner_texture", 0)) < 28
            or float(source_evidence.get("source_corner_color_spread", 0)) < 24
        ):
            raise ValueError(f"候选证据不达标：{image_name}")
        image = (directory / str(candidate["image"])).resolve()
        overlay = directory / "overlays" / image_name
        if not image.is_relative_to(root) or not image.is_file() or not overlay.is_file():
            raise ValueError(f"图像或叠加图缺失：{image_name}")
        quad = candidate["content_quad"]
        if len(quad) != 4 or any(
            len(point) != 2 or any(not 0 <= float(value) <= 1 for value in point)
            for point in quad
        ):
            raise ValueError(f"四角归一化格式错误：{image_name}")
        group = f"syngallery-met-{object_id}"
        split = str(candidate["split"])
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"同一作品跨 split：{group}")
        group_splits[group] = split
        records.append(
            {
                "image": image.relative_to(root).as_posix(),
                "split": split,
                "group_id": group,
                "capture_session": group,
                "device": "synthetic-gallery-camera",
                "present": True,
                "target_class": "artwork",
                "content_quad": quad,
                "outer_quad": None,
                "visible": True,
                "occlusion": 0.0,
                "glare_level": "light" if image_name in light_glare else "none",
                "source": "syngallery-reviewed",
                "scene_type": "rendered_gallery_artwork",
                "in_scope": True,
                "ambiguous": False,
                "digital_source_id": f"met-open-access-{object_id}",
                "subject_id": group,
                "domain": "artwork",
            }
        )
        _progress(index, len(candidates))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    review = {
        "kind": "syngallery_visual_review_provenance",
        "reviewer": args.reviewer,
        "dataset_revision": DATASET_REVISION,
        "candidate_count": len(candidates),
        "approved_count": len(records),
        "skipped": dict(skipped),
        "excluded_images": sorted(exclusions),
        "light_glare_images": sorted(light_glare),
        "review_boards_sha256": board_hashes,
        "manifest_sha256": _sha256(output),
    }
    output.with_suffix(".review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"approved": len(records), "groups": len(group_splits), "skipped": dict(skipped)}, ensure_ascii=False))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(done: int, total: int) -> None:
    if done < total and done % max(1, math.ceil(total / 100)):
        return
    width = 24
    filled = round(width * done / max(1, total))
    print(
        f"\r[{'#' * filled}{'-' * (width - filled)}] {done}/{total} 审核来源与 split",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
