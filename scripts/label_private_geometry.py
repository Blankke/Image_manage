"""为本地 private 图片建立固定分组并交互标注几何真值。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/label_private_geometry.py \
        --data-root "$SCREENRESTORE_DATA_ROOT" \
        --image-directory "$SCREENRESTORE_DATA_ROOT/private" \
        --output "$SCREENRESTORE_DATA_ROOT/private/geometry.annotations.jsonl"

窗口快捷键：1/2/3/0 选择 artwork/postcard/screen/none；C 点击 content 四角；
O 点击 outer 四角；M 切换“多个同等合理目标”；A 切换其它 ambiguous；
U 切换 unusable；S 保存当前组；Q 安全退出。
四角按左上、右上、右下、左下依次点击。脚本优先以 ``*_hd`` 作为同名缩略图组代表，
固定 split 只按 unique group 建立；已有输出中的 split 会保持不变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_KEYS = {ord("1"): "artwork", ord("2"): "postcard", ord("3"): "screen", ord("0"): "none"}


@dataclass(slots=True)
class GroupAnnotation:
    target_class: str = "artwork"
    content_quad: list[list[float]] | None = None
    outer_quad: list[list[float]] | None = None
    scene_type: str = "gallery_artwork"
    ambiguous: bool = False
    unusable: bool = False

    @property
    def multiple_targets(self) -> bool:
        return self.scene_type == "gallery_multi_target"

    @property
    def present(self) -> bool:
        # 层级或目标选择存在歧义时，自动模式的权威语义是拒绝。这里直接把它写成
        # presence 负样本，避免训练器忽略 ``ambiguous`` 元数据后反而学习“必须接受”。
        return (
            self.target_class != "none"
            and not self.multiple_targets
            and not self.ambiguous
            and not self.unusable
            and self.content_quad is not None
        )

    def toggle_multiple_targets(self) -> None:
        """多目标场景没有权威单一四角，当前自动模式应将其作为拒绝样本。"""

        if self.multiple_targets:
            self.scene_type = _default_scene_type(self.target_class)
            self.ambiguous = False
        else:
            self.scene_type = "gallery_multi_target"
            self.ambiguous = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    image_directory = args.image_directory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not image_directory.is_dir() or not image_directory.is_relative_to(data_root):
        raise ValueError("image-directory 必须位于显式 data-root 内")
    if not output.is_relative_to(data_root / "private"):
        raise ValueError("private 标注必须写入 data-root/private")
    groups = _group_images(image_directory)
    if not groups:
        raise ValueError("private 目录中没有可标注图片")
    existing = _read_existing(output)
    splits = _fixed_splits(groups, existing)
    annotations = _existing_annotations(existing)
    saved_groups = set(annotations)
    for index, (group_id, images) in enumerate(groups.items(), start=1):
        annotation = annotations.get(group_id, GroupAnnotation())
        representative = _representative(images)
        image_bgr = cv2.imread(str(representative), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"无法解码 private sample-{index:03d}")
        action = _label_group(image_bgr, annotation, index, len(groups), splits[group_id])
        if action == "quit":
            break
        annotations[group_id] = annotation
        saved_groups.add(group_id)
        _write_annotations(output, data_root, groups, splits, annotations, saved_groups)
        _progress(index, len(groups), "保存 private geometry 标注")
    cv2.destroyAllWindows()
    _write_annotations(output, data_root, groups, splits, annotations, saved_groups)
    print(output)
    return 0


def _group_images(directory: Path) -> dict[str, list[Path]]:
    """先合并 ``*_hd`` 逻辑对，再用低分辨率内容指纹合并其它近重复。"""

    logical_groups: dict[str, list[Path]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(directory)
        stem = relative.stem[:-3] if relative.stem.lower().endswith("_hd") else relative.stem
        logical = (relative.parent / stem).as_posix().lower()
        logical_groups.setdefault(logical, []).append(path)
    clusters: list[tuple[str, tuple[int, np.ndarray, float], list[Path]]] = []
    for logical, images in sorted(logical_groups.items()):
        representative = _representative(images)
        fingerprint = _content_fingerprint(representative)
        matched = None
        for index, (_cluster_key, cluster_fingerprint, _cluster_images) in enumerate(clusters):
            if _fingerprints_match(fingerprint, cluster_fingerprint):
                matched = index
                break
        if matched is None:
            clusters.append((logical, fingerprint, list(images)))
        else:
            cluster_key, cluster_fingerprint, cluster_images = clusters[matched]
            clusters[matched] = (
                min(cluster_key, logical),
                cluster_fingerprint,
                [*cluster_images, *images],
            )
    grouped: dict[str, list[Path]] = {}
    for cluster_key, _fingerprint, images in clusters:
        digest = hashlib.sha256(cluster_key.encode("utf-8")).hexdigest()[:16]
        grouped[f"private:{digest}"] = sorted(images)
    return dict(sorted(grouped.items()))


def _content_fingerprint(path: Path) -> tuple[int, np.ndarray, float]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("无法解码 private 图片以建立内容分组")
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] >= resized[:, :-1]).reshape(-1)
    dhash = sum(int(value) << index for index, value in enumerate(bits))
    thumbnail = cv2.resize(image_bgr, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    signature = np.concatenate(
        (thumbnail.mean(axis=(0, 1)), thumbnail.std(axis=(0, 1)))
    ).astype(np.float32)
    return dhash, signature, width / max(1, height)


def _fingerprints_match(
    first: tuple[int, np.ndarray, float],
    second: tuple[int, np.ndarray, float],
) -> bool:
    hash_distance = (first[0] ^ second[0]).bit_count()
    color_distance = float(np.max(np.abs(first[1] - second[1])))
    aspect_difference = abs(first[2] - second[2]) / max(first[2], second[2], 1e-6)
    return hash_distance <= 6 and color_distance <= 16.0 and aspect_difference <= 0.025


def _representative(images: list[Path]) -> Path:
    hd = [path for path in images if path.stem.lower().endswith("_hd")]
    return sorted(hd or images)[0]


def _fixed_splits(
    groups: dict[str, list[Path]],
    existing: list[dict[str, Any]],
) -> dict[str, str]:
    preserved = {str(record["group_id"]): str(record["split"]) for record in existing}
    ordered = sorted(groups, key=lambda value: hashlib.sha256(value.encode("utf-8")).digest())
    train_end = round(len(ordered) * 0.60)
    validation_end = train_end + round(len(ordered) * 0.20)
    generated = {
        group_id: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, group_id in enumerate(ordered)
    }
    generated.update({group_id: split for group_id, split in preserved.items() if group_id in groups})
    return generated


def _label_group(
    image_bgr: np.ndarray,
    annotation: GroupAnnotation,
    index: int,
    total: int,
    split: str,
) -> str:
    window = "ScreenRestore private geometry labeling"
    mode: str | None = None
    pending: list[tuple[int, int]] = []

    def mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        nonlocal mode, pending
        if event == cv2.EVENT_LBUTTONDOWN and mode is not None and len(pending) < 4:
            pending.append((x, y))
            if len(pending) == 4:
                normalized = _normalize_points(pending, image_bgr.shape)
                if mode == "content":
                    annotation.content_quad = normalized
                else:
                    annotation.outer_quad = normalized
                mode, pending = None, []

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    while True:
        canvas = _render_annotation(image_bgr, annotation, pending, index, total, split, mode)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in CLASS_KEYS:
            annotation.target_class = CLASS_KEYS[key]
            if not annotation.multiple_targets:
                annotation.scene_type = _default_scene_type(annotation.target_class)
            if annotation.target_class == "none":
                annotation.content_quad = None
                annotation.outer_quad = None
            mode, pending = None, []
        elif key in (ord("c"), ord("C")):
            mode, pending = "content", []
        elif key in (ord("o"), ord("O")):
            mode, pending = "outer", []
        elif key in (ord("a"), ord("A")):
            annotation.ambiguous = not annotation.ambiguous
        elif key in (ord("m"), ord("M")):
            annotation.toggle_multiple_targets()
        elif key in (ord("u"), ord("U")):
            annotation.unusable = not annotation.unusable
        elif key in (8, 127):
            if pending:
                pending.pop()
        elif key in (ord("s"), ord("S")):
            if annotation.target_class != "none" and annotation.content_quad is None:
                continue
            return "save"
        elif key in (ord("q"), ord("Q")):
            return "quit"


def _render_annotation(
    image_bgr: np.ndarray,
    annotation: GroupAnnotation,
    pending: list[tuple[int, int]],
    index: int,
    total: int,
    split: str,
    mode: str | None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    for quad, color in ((annotation.outer_quad, (210, 70, 170)), (annotation.content_quad, (30, 150, 245))):
        if quad is not None:
            points = np.rint(np.asarray(quad) * [width - 1, height - 1]).astype(np.int32)
            cv2.polylines(canvas, [points], True, color, max(2, round(max(height, width) / 500)))
    for point_index, point in enumerate(pending, start=1):
        cv2.circle(canvas, point, max(4, round(max(height, width) / 250)), (20, 240, 50), -1)
        cv2.putText(canvas, str(point_index), point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    status = (
        f"sample {index}/{total} split={split} class={annotation.target_class} "
        f"multi={int(annotation.multiple_targets)} ambiguous={int(annotation.ambiguous)} "
        f"unusable={int(annotation.unusable)} mode={mode or '-'}"
    )
    cv2.rectangle(canvas, (0, 0), (width, 42), (245, 245, 245), -1)
    cv2.putText(canvas, status, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
    return canvas


def _normalize_points(points: list[tuple[int, int]], shape: tuple[int, ...]) -> list[list[float]]:
    height, width = shape[:2]
    normalized = np.asarray(points, np.float32) / np.array([max(1, width - 1), max(1, height - 1)])
    return np.clip(normalized, 0.0, 1.0).astype(float).tolist()


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _existing_annotations(records: list[dict[str, Any]]) -> dict[str, GroupAnnotation]:
    values: dict[str, GroupAnnotation] = {}
    for record in records:
        group_id = str(record["group_id"])
        values.setdefault(
            group_id,
            GroupAnnotation(
                target_class=str(record["target_class"]),
                content_quad=record.get("content_quad"),
                outer_quad=record.get("outer_quad"),
                scene_type=str(record.get("scene_type", "gallery_artwork")),
                ambiguous=bool(record.get("ambiguous", False))
                or record.get("scene_type") == "gallery_multi_target",
                unusable=bool(record.get("unusable", False)),
            ),
        )
    return values


def _write_annotations(
    path: Path,
    data_root: Path,
    groups: dict[str, list[Path]],
    splits: dict[str, str],
    annotations: dict[str, GroupAnnotation],
    saved_groups: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for group_id in sorted(saved_groups):
        annotation = annotations[group_id]
        for image in groups[group_id]:
            records.append(
                {
                    "image": image.relative_to(data_root).as_posix(),
                    "split": splits[group_id],
                    "group_id": group_id,
                    "capture_session": group_id,
                    "device": "private-unknown",
                    "present": annotation.present,
                    "target_class": annotation.target_class if annotation.present else "none",
                    "content_quad": annotation.content_quad if annotation.present else None,
                    "outer_quad": annotation.outer_quad if annotation.present else None,
                    "visible": annotation.present,
                    "occlusion": 0.0,
                    "glare_level": "none",
                    "scene_type": annotation.scene_type,
                    "ambiguous": annotation.ambiguous or annotation.multiple_targets,
                    "unusable": annotation.unusable,
                    "in_scope": annotation.present,
                    "source": "private-labeled",
                }
            )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_scene_type(target_class: str) -> str:
    return {
        "artwork": "gallery_artwork",
        "postcard": "tabletop_postcard",
        "screen": "screen",
        "none": "no_target",
    }[target_class]


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * min(1.0, done / max(1, total)))
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {message}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
