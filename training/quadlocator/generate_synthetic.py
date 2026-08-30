"""生成无版权依赖的 QuadLocator 嵌套矩形合成训练集。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.quadlocator.generate_synthetic --output-directory /tmp/quad-synth --count 200

输出 ``manifest.jsonl`` 和 ``images/``。可选从 Met/DIV2K 内容池和 COCO/DIV2K 背景池
读取本地纹理；未提供时使用 NumPy/OpenCV 程序纹理。脚本不会访问网络或读取 private，
执行期间始终显示文本进度条。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

TARGET_CLASSES = ("artwork", "postcard", "screen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--negative-ratio", type=float, default=0.12)
    parser.add_argument(
        "--content-directory",
        type=Path,
        action="append",
        default=[],
        help="可重复指定 Met/DIV2K 等本地公开内容纹理目录",
    )
    parser.add_argument(
        "--background-directory",
        type=Path,
        action="append",
        default=[],
        help="可重复指定 COCO val/DIV2K 等本地公开背景目录",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--max-output-gib",
        type=float,
        default=6.0,
        help="合成图片与 manifest 的最大输出预算，默认 6 GiB",
    )
    args = parser.parse_args(argv)
    if args.count < 3:
        raise ValueError("count 至少为 3")
    if args.size < 128 or args.size % 32:
        raise ValueError("size 必须不小于 128 且为 32 的倍数")
    if not 0.0 <= args.negative_ratio <= 0.5:
        raise ValueError("negative-ratio 必须位于 0..0.5")
    if args.max_output_gib <= 0:
        raise ValueError("max-output-gib 必须大于 0")
    output_directory = args.output_directory.expanduser().resolve()
    image_directory = output_directory / "images"
    manifest_path = output_directory / "manifest.jsonl"
    if manifest_path.exists() or (image_directory.is_dir() and any(image_directory.iterdir())):
        raise ValueError("输出目录已有合成数据；请使用新的目录，避免覆盖或混入历史样本")
    image_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    content_paths = _image_paths(args.content_directory)
    background_paths = _image_paths(args.background_directory)
    # 同一公开纹理只允许进入一个 split，避免换了透视与退化后仍把同一作品或场景
    # 同时泄漏到训练与验证测试。
    content_paths_by_split = _partition_paths_by_split(content_paths)
    background_paths_by_split = _partition_paths_by_split(background_paths)
    output_budget = int(args.max_output_gib * 1024**3)
    written_bytes = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(args.count):
            _progress(index, args.count, "生成嵌套矩形与退化")
            negative = bool(rng.random() < args.negative_ratio)
            image, record = _sample(
                rng,
                args.size,
                index,
                negative,
                content_paths_by_split=content_paths_by_split,
                background_paths_by_split=background_paths_by_split,
            )
            relative_path = Path("images") / f"sample_{index:06d}.jpg"
            target_path = output_directory / relative_path
            ok = cv2.imwrite(
                str(target_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(78, 98))],
            )
            if not ok:
                raise OSError(f"无法写入合成图：{target_path}")
            written_bytes += target_path.stat().st_size
            if written_bytes > output_budget:
                raise RuntimeError(
                    f"合成数据已超过 max-output-gib={args.max_output_gib}；"
                    "保留当前部分产物供人工检查，不继续写入"
                )
            record["image"] = relative_path.as_posix()
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    _progress(args.count, args.count, f"完成：{manifest_path}")
    return 0


def _sample(
    rng: np.random.Generator,
    size: int,
    index: int,
    negative: bool,
    *,
    content_paths_by_split: dict[str, list[Path]] | None = None,
    background_paths_by_split: dict[str, list[Path]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    split_value = index % 10
    split = "train" if split_value < 8 else "validation" if split_value == 8 else "test"
    content_paths = (content_paths_by_split or {}).get(split, [])
    background_paths = (background_paths_by_split or {}).get(split, [])
    background = _background(rng, size, background_paths)
    base_record: dict[str, object] = {
        "split": split,
        "group_id": f"synthetic-{index:06d}",
        # 每个会话只能属于一个 split，便于训练入口直接执行泄漏检查。
        "capture_session": f"procedural-{split}-{index // 100:04d}",
        "device": "synthetic-pinhole",
        "visible": True,
        "occlusion": 0.0,
        "glare_level": "none",
        "source": "synthetic-planar-v2",
    }
    if negative:
        hard_negative = str(
            rng.choice(
                (
                    "windows",
                    "doors",
                    "books",
                    "signs",
                    "tables",
                    "generic_rectangles",
                    "multiple_equally_plausible",
                    "heavily_truncated_target",
                    "no_planar_target",
                )
            )
        )
        _draw_hard_negative(background, rng, hard_negative)
        return background, {
            **base_record,
            "present": False,
            "target_class": "none",
            "content_quad": None,
            "outer_quad": None,
            "scene_type": hard_negative,
            "ambiguous": hard_negative == "multiple_equally_plausible",
            "in_scope": False,
        }

    target_class = str(rng.choice(TARGET_CLASSES))
    scene_type = _scene_type(rng, target_class)
    if scene_type in {"partial_artwork", "partial_screen", "multiple_artworks"}:
        _draw_hard_negative(
            background,
            rng,
            "heavily_truncated_target"
            if scene_type.startswith("partial")
            else "multiple_equally_plausible",
        )
        return background, {
            **base_record,
            "present": False,
            "target_class": "none",
            "content_quad": None,
            "outer_quad": None,
            "scene_type": scene_type,
            "ambiguous": scene_type == "multiple_artworks",
            "in_scope": False,
        }
    small = scene_type == "small_artwork"
    minimum = 0.20 if small else 0.44
    maximum = 0.38 if small else 0.80
    patch_width = int(rng.integers(round(size * minimum), round(size * maximum)))
    patch_height = int(rng.integers(round(size * minimum), round(size * maximum)))
    patch, source_content, has_outer = _nested_patch(
        rng,
        patch_width,
        patch_height,
        target_class,
        scene_type,
        content_paths or [],
    )
    if scene_type == "multiple_rectangular_distractors":
        _draw_hard_negative(background, rng, "generic_rectangles")
    margin = size * (0.012 if scene_type == "near_border_artwork" else 0.06)
    center = np.array(
        [rng.uniform(size * 0.43, size * 0.57), rng.uniform(size * 0.43, size * 0.57)],
        dtype=np.float32,
    )
    half = np.array([patch_width / 2, patch_height / 2], dtype=np.float32)
    base_quad = np.array(
        [center - half, center + [-half[0], half[1]], center + half, center + [half[0], -half[1]]],
        dtype=np.float32,
    )[[0, 3, 2, 1]]
    jitter_scale = 0.06 if scene_type in {"perspective", "reflection", "glass_glare"} else 0.032
    jitter = rng.normal(0.0, size * jitter_scale, size=(4, 2)).astype(np.float32)
    outer_quad = base_quad + jitter
    outer_quad[:, 0] = np.clip(outer_quad[:, 0], margin, size - 1 - margin)
    outer_quad[:, 1] = np.clip(outer_quad[:, 1], margin, size - 1 - margin)
    source_outer = np.array(
        [[0, 0], [patch_width - 1, 0], [patch_width - 1, patch_height - 1], [0, patch_height - 1]],
        np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_outer, outer_quad)
    warped = cv2.warpPerspective(patch, homography, (size, size), borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full((patch_height, patch_width), 255, np.uint8), homography, (size, size))
    background[valid > 0] = warped[valid > 0]
    content_quad = cv2.perspectiveTransform(source_content[None], homography)[0]
    glare = "none"
    if scene_type in {"glass_glare", "reflection"} or rng.random() < 0.30:
        glare = "light" if rng.random() < 0.75 else "medium"
        overlay = background.copy()
        axes = (int(rng.integers(size // 8, size // 3)), int(rng.integers(size // 20, size // 8)))
        center_glare = tuple(int(value) for value in rng.integers(size // 5, size * 4 // 5, size=2))
        cv2.ellipse(overlay, center_glare, axes, float(rng.integers(0, 180)), 0, 360, (245, 245, 245), -1)
        background = cv2.addWeighted(overlay, float(rng.uniform(0.08, 0.22)), background, 1.0, 0)
    noise = rng.normal(0.0, rng.uniform(1.0, 5.0), background.shape)
    background = np.clip(background.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return background, {
        **base_record,
        "present": True,
        "target_class": target_class,
        "content_quad": _normalized(content_quad, size),
        "outer_quad": _normalized(outer_quad, size) if has_outer else None,
        "glare_level": glare,
        "scene_type": scene_type,
        "ambiguous": False,
        "in_scope": True,
    }


def _background(rng: np.random.Generator, size: int, paths: list[Path]) -> np.ndarray:
    if paths and rng.random() < 0.78:
        return _texture_crop(paths[int(rng.integers(0, len(paths)))], size, size, rng)
    yy, xx = np.indices((size, size), dtype=np.float32)
    base = rng.uniform(45, 185, size=3)
    slope_x = rng.uniform(-35, 35, size=3)
    slope_y = rng.uniform(-35, 35, size=3)
    image = base + (xx[:, :, None] / size - 0.5) * slope_x + (yy[:, :, None] / size - 0.5) * slope_y
    return np.clip(image, 0, 255).astype(np.uint8)


def _nested_patch(
    rng: np.random.Generator,
    width: int,
    height: int,
    target_class: str,
    scene_type: str,
    content_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, bool]:
    patch = np.full((height, width, 3), rng.integers(18, 90, size=3), dtype=np.uint8)
    minimum_edge = min(width, height)
    frameless = scene_type in {"frameless_artwork", "postcard_document"}
    thin = scene_type in {"thin_frame", "thin_bezel"}
    thick = scene_type in {"thick_frame", "dark_bezel"}
    has_mat = scene_type in {"mat_artwork", "mat_frame_artwork"}
    frame_ratio = 0.0 if frameless else rng.uniform(0.012, 0.028) if thin else rng.uniform(0.06, 0.12) if thick else rng.uniform(0.025, 0.065)
    frame = max(0, round(minimum_edge * frame_ratio))
    mat_ratio = rng.uniform(0.11, 0.20) if has_mat else rng.uniform(0.035, 0.085)
    mat = 0 if frameless else max(frame + 2, round(minimum_edge * mat_ratio))
    cv2.rectangle(patch, (frame, frame), (width - frame - 1, height - frame - 1), (220, 216, 205), -1)
    y0, y1 = mat, height - mat
    x0, x1 = mat, width - mat
    yy, xx = np.indices((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.float32)
    if content_paths and target_class == "artwork" and rng.random() < 0.88:
        content = _texture_crop(
            content_paths[int(rng.integers(0, len(content_paths)))],
            max(1, x1 - x0),
            max(1, y1 - y0),
            rng,
        )
    elif target_class == "screen":
        content = np.stack(
            ((xx * 1.7) % 255, (yy * 2.3) % 255, ((xx + yy) * 1.1) % 255), axis=2
        )
        if scene_type == "dark_screen":
            content *= 0.16
        elif scene_type == "bright_screen":
            content = np.clip(content * 0.45 + 145, 0, 255)
    else:
        color_a = rng.uniform(25, 225, size=3)
        color_b = rng.uniform(25, 225, size=3)
        wave = (np.sin(xx / rng.uniform(8, 25)) + np.cos(yy / rng.uniform(10, 28)))[:, :, None]
        content = color_a + (wave + 2.0) / 4.0 * (color_b - color_a)
    patch[y0:y1, x0:x1] = np.clip(content, 0, 255).astype(np.uint8)
    cv2.rectangle(patch, (x0, y0), (x1 - 1, y1 - 1), (12, 12, 12), 2)
    content_quad = np.array(
        [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]],
        dtype=np.float32,
    )
    return patch, content_quad, not frameless


def _scene_type(rng: np.random.Generator, target_class: str) -> str:
    values = {
        "artwork": (
            "artwork",
            "frameless_artwork",
            "thin_frame",
            "thick_frame",
            "mat_artwork",
            "mat_frame_artwork",
            "glass_glare",
            "multiple_artworks",
            "small_artwork",
            "near_border_artwork",
            "partial_artwork",
            "perspective",
        ),
        "screen": (
            "screen_content",
            "dark_bezel",
            "thin_bezel",
            "bright_screen",
            "dark_screen",
            "reflection",
            "perspective",
            "partial_screen",
            "multiple_rectangular_distractors",
        ),
        "postcard": ("postcard_document", "perspective", "near_border_document"),
    }[target_class]
    return str(rng.choice(values))


def _draw_hard_negative(
    image: np.ndarray,
    rng: np.random.Generator,
    scene_type: str,
) -> None:
    size = image.shape[0]
    rectangle_count = 0 if scene_type == "no_planar_target" else int(rng.integers(1, 7))
    if scene_type == "multiple_equally_plausible":
        rectangle_count = int(rng.integers(3, 6))
    for index in range(rectangle_count):
        width = int(rng.integers(size // 7, size // 2))
        height = int(rng.integers(size // 8, size // 2))
        if scene_type == "heavily_truncated_target" and index == 0:
            x0 = int(rng.choice((-width * 3 // 4, size - width // 4)))
            y0 = int(rng.integers(-height // 2, size - height // 2))
        else:
            x0 = int(rng.integers(0, max(1, size - width)))
            y0 = int(rng.integers(0, max(1, size - height)))
        color = tuple(int(value) for value in rng.integers(25, 230, size=3))
        thickness = -1 if scene_type in {"books", "signs"} else int(rng.integers(2, 10))
        cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), color, thickness)
    if scene_type == "no_planar_target":
        for _ in range(int(rng.integers(4, 12))):
            center = tuple(int(value) for value in rng.integers(0, size, size=2))
            cv2.circle(image, center, int(rng.integers(4, size // 8)), (80, 90, 110), -1)


def _image_paths(directories: list[Path]) -> list[Path]:
    suffixes = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    paths: list[Path] = []
    for directory in directories:
        resolved = directory.expanduser().resolve()
        if not resolved.is_dir() or "private" in resolved.parts:
            raise ValueError(f"内容/背景目录必须是存在的公开目录：{resolved}")
        paths.extend(path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    return sorted(paths)


def _partition_paths_by_split(paths: list[Path]) -> dict[str, list[Path]]:
    """按数据集末两级相对标识固定纹理 split；空分区由程序纹理自然补位。"""

    result: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}
    for path in paths:
        # 忽略机器上的 data-root 前缀，使同一份公开数据迁移目录后仍保持原 split。
        source_id = Path(*path.parts[-2:]).as_posix()
        bucket = hashlib.sha256(source_id.encode("utf-8")).digest()[0] % 10
        split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        result[split].append(path)
    return result


def _texture_crop(
    path: Path,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取公开纹理：{path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (max(width, round(source_width * scale)), max(height, round(source_height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x = int(rng.integers(0, resized.shape[1] - width + 1))
    y = int(rng.integers(0, resized.shape[0] - height + 1))
    return resized[y : y + height, x : x + width].copy()


def _normalized(points: np.ndarray, size: int) -> list[list[float]]:
    return np.clip(points / max(1, size - 1), 0.0, 1.0).astype(float).tolist()


def _progress(done: int, total: int, message: str) -> None:
    width = 28
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if done >= total else "\r"
    print(f"[{bar}] {done:>6}/{total:<6} {message}", end=ending, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
