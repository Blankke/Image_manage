"""生成 QuadLocator v9 目标域合成训练集。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.quadlocator.generate_synthetic --output-directory /tmp/quad-synth --count 200

输出 ``manifest.jsonl`` 和 ``images/``。覆盖远拍小目标、偏心构图、墙面/桌面、常见
物体画幅、图片查看器/网页中的内层图像、显示器暗色边框、薄卡片纸边、墙面海报集群，
以及外框超出画面但内层图片完整可见的近拍屏幕。屏幕支架与画架互为反事实，墙上邻近
海报只作为干扰目标，不能改变主目标的 content 语义。v9 加入程序生成的木纹课桌、
远拍杂志、墙角与胶带，并继续使用同 split 的公开内容纹理构造邻近海报。
可选从 Met/DIV2K 内容池和 COCO/DIV2K 背景池读取本地公开纹理；
未提供时使用 NumPy/OpenCV 程序纹理。脚本不会访问网络或读取 private，执行期间始终
显示文本进度条。
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
        "source": "synthetic-target-v9",
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
            "visible": False,
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
            "visible": False,
            "target_class": "none",
            "content_quad": None,
            "outer_quad": None,
            "scene_type": scene_type,
            "ambiguous": scene_type == "multiple_artworks",
            "in_scope": False,
        }
    if scene_type in {
        "monitor_photo_viewer",
        "monitor_browser_image",
        "cropped_monitor_photo_viewer",
        "cropped_monitor_browser_image",
    }:
        _draw_screen_environment(background, rng)
    minimum, maximum = _target_scale_range(scene_type)
    patch_width, patch_height = _sample_patch_size(
        rng, size, minimum, maximum, target_class
    )
    patch, source_content, has_outer = _nested_patch(
        rng,
        patch_width,
        patch_height,
        target_class,
        scene_type,
        content_paths or [],
    )
    if scene_type in {
        "multiple_rectangular_distractors",
        "multi_screen_ui",
        "wall_poster",
        "desk_postcard",
    }:
        _draw_hard_negative(background, rng, "generic_rectangles")
    closeup_scene = scene_type.startswith("closeup_") or scene_type in {
        "magazine_cover",
        "printed_poster",
        "monitor_photo_viewer",
        "monitor_browser_image",
        "portable_image_viewer",
    }
    cropped_screen = scene_type in {
        "cropped_monitor_photo_viewer",
        "cropped_monitor_browser_image",
    }
    margin = size * (0.012 if scene_type == "near_border_artwork" or closeup_scene else 0.06)
    center = (
        np.asarray(
            [
                size * rng.uniform(0.44, 0.56),
                size * rng.uniform(0.44, 0.56),
            ],
            dtype=np.float32,
        )
        if cropped_screen
        else _sample_center(rng, size, patch_width, patch_height, margin, scene_type)
    )
    half = np.array([patch_width / 2, patch_height / 2], dtype=np.float32)
    base_quad = np.array(
        [center - half, center + [-half[0], half[1]], center + half, center + [half[0], -half[1]]],
        dtype=np.float32,
    )[[0, 3, 2, 1]]
    jitter_scale = (
        0.06
        if scene_type in {"perspective", "reflection", "glass_glare"}
        else 0.014
        if scene_type in {"desk_magazine_far", "far_wall_poster"}
        else 0.032
    )
    jitter = rng.normal(0.0, size * jitter_scale, size=(4, 2)).astype(np.float32)
    outer_quad = base_quad + jitter
    if not cropped_screen:
        outer_quad[:, 0] = np.clip(outer_quad[:, 0], margin, size - 1 - margin)
        outer_quad[:, 1] = np.clip(outer_quad[:, 1], margin, size - 1 - margin)
    if scene_type in {"mounted_wall_poster", "wall_poster_cluster", "far_wall_poster"}:
        _draw_wall_poster_context(
            background,
            rng,
            clustered=scene_type == "wall_poster_cluster",
            content_paths=content_paths or [],
            reserved_quad=outer_quad,
        )
    elif scene_type in {"magazine_cover", "desk_magazine_far", "desk_postcard"}:
        _draw_wood_desk_context(background, rng, reserved_quad=outer_quad)
    source_outer = np.array(
        [[0, 0], [patch_width - 1, 0], [patch_width - 1, patch_height - 1], [0, patch_height - 1]],
        np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_outer, outer_quad)
    _draw_planar_shadow(background, outer_quad, rng)
    if target_class == "screen" and scene_type.startswith("monitor_"):
        # 支架位于矩形屏幕外框之后，仅作为设备类别证据，不进入 outer_quad 语义。
        _draw_monitor_support(background, outer_quad, rng)
    elif target_class == "artwork" and scene_type == "easel_artwork":
        # 画架是重要反事实：出现外部支撑结构并不等于显示器。
        _draw_artwork_easel(background, outer_quad, rng)
    warped = cv2.warpPerspective(patch, homography, (size, size), borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full((patch_height, patch_width), 255, np.uint8), homography, (size, size))
    background[valid > 0] = warped[valid > 0]
    if scene_type in {"mounted_wall_poster", "wall_poster_cluster", "far_wall_poster"}:
        _draw_poster_tape(background, rng, outer_quad)
    content_quad = cv2.perspectiveTransform(source_content[None], homography)[0]
    glare = "none"
    if scene_type in {"glass_glare", "reflection"} or rng.random() < 0.30:
        glare = "light" if rng.random() < 0.75 else "medium"
        overlay = background.copy()
        axes = (int(rng.integers(size // 8, size // 3)), int(rng.integers(size // 20, size // 8)))
        center_glare = tuple(int(value) for value in rng.integers(size // 5, size * 4 // 5, size=2))
        cv2.ellipse(overlay, center_glare, axes, float(rng.integers(0, 180)), 0, 360, (245, 245, 245), -1)
        background = cv2.addWeighted(overlay, float(rng.uniform(0.08, 0.22)), background, 1.0, 0)
    background = _apply_camera_degradation(background, rng)
    return background, {
        **base_record,
        "present": True,
        "target_class": target_class,
        "content_quad": _normalized(content_quad, size),
        # 近拍屏幕的物理外框并不完整可见，因此不能伪造 outer 标注；内层图片只要
        # 完整落在画面内，仍是首版产品允许自动处理的 content 目标。
        "outer_quad": _normalized(outer_quad, size) if has_outer and not cropped_screen else None,
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


def _draw_magazine_cover_layout(content: np.ndarray, rng: np.random.Generator) -> None:
    """在公开图像上叠加程序排版，构造画面与印刷边界一致的杂志封面。"""

    height, width = content.shape[:2]
    if min(height, width) < 20:
        return
    overlay = content.copy()
    header_height = max(5, round(height * rng.uniform(0.13, 0.23)))
    footer_height = max(3, round(height * rng.uniform(0.045, 0.085)))
    dark = bool(rng.random() < 0.55)
    header_color = (22, 28, 35) if dark else (225, 218, 199)
    ink_color = (245, 244, 236) if dark else (31, 37, 43)
    cv2.rectangle(overlay, (0, 0), (width - 1, header_height), header_color, -1)
    cv2.rectangle(
        overlay, (0, height - footer_height), (width - 1, height - 1), header_color, -1
    )
    cv2.addWeighted(overlay, 0.88, content, 0.12, 0.0, dst=content)
    font_scale = max(0.20, min(1.65, width / 235.0))
    cv2.putText(
        content,
        "VISUAL",
        (max(2, width // 22), max(5, round(header_height * 0.76))),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        ink_color,
        max(1, round(font_scale * 2)),
        cv2.LINE_AA,
    )
    # 短线提供次级标题密度，不引入真实出版物文字或标识。
    for line in range(2):
        y = min(height - footer_height - 2, header_height + max(3, height // 18) * (line + 1))
        if y > header_height:
            cv2.line(
                content,
                (max(2, width // 20), y),
                (max(3, round(width * rng.uniform(0.38, 0.68))), y),
                ink_color,
                max(1, width // 95),
            )


def _draw_printed_poster_layout(content: np.ndarray, rng: np.random.Generator) -> None:
    """用程序标题和短行模拟墙面海报的印刷信息层，不借用外部海报设计。"""

    height, width = content.shape[:2]
    if min(height, width) < 24:
        return
    top = max(6, round(height * rng.uniform(0.15, 0.24)))
    bottom = max(5, round(height * rng.uniform(0.10, 0.18)))
    overlay = content.copy()
    paper = tuple(int(value) for value in rng.integers(224, 251, size=3))
    cv2.rectangle(overlay, (0, 0), (width - 1, top), paper, -1)
    cv2.rectangle(overlay, (0, height - bottom), (width - 1, height - 1), paper, -1)
    cv2.addWeighted(overlay, 0.88, content, 0.12, 0.0, dst=content)
    ink = (37, 45, 54)
    cv2.putText(
        content,
        "ART / CITY",
        (max(2, width // 20), max(6, round(top * 0.72))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.20, min(1.20, width / 300.0)),
        ink,
        max(1, width // 160),
        cv2.LINE_AA,
    )
    for line in range(3):
        y = height - bottom + max(2, round((line + 1) * bottom / 4))
        if y < height:
            cv2.line(
                content,
                (max(2, width // 20), y),
                (max(3, round(width * rng.uniform(0.40, 0.86))), y),
                ink,
                max(1, width // 180),
            )


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
    if target_class == "screen":
        # 屏幕的 content 四边形由暗色物理 bezel 包围。旧分布复用了画作的米白卡纸，
        # 会把 screen 与 artwork 的监督定义混在一起，真实显示器也几乎不会呈现该结构。
        bezel_ratio = (
            rng.uniform(0.085, 0.14)
            if scene_type == "portable_image_viewer"
            else rng.uniform(0.010, 0.022)
            if scene_type in {"thin_bezel", "monitor_thin_bezel"}
            else rng.uniform(0.055, 0.095)
            if scene_type in {"dark_bezel", "monitor_dark_bezel"}
            else rng.uniform(0.022, 0.050)
        )
        bezel = max(2, round(minimum_edge * bezel_ratio))
        bottom_bezel = min(height // 4, max(bezel, round(bezel * rng.uniform(1.0, 1.45))))
        x0, x1 = bezel, width - bezel
        y0, y1 = bezel, height - bottom_bezel
        bezel_rgb = rng.integers(8, 43, size=3, dtype=np.uint8)
        patch[:] = bezel_rgb
        # 很弱的高光让黑边仍有成像层次，避免模型只记住纯黑合成色块。
        highlight = tuple(int(min(70, int(value) + rng.integers(8, 22))) for value in bezel_rgb)
        cv2.line(patch, (1, 1), (width - 2, 1), highlight, 1)
        frame = bezel
        mat = bezel
        frameless = False
        has_outer = True
    elif target_class == "postcard":
        # 明信片是薄卡片，不复用画作的“深色画框 + 大面积卡纸”。一部分正面为
        # full-bleed，另一部分保留窄纸边，从物体比例与纸张边缘提供真实类别证据。
        full_bleed = scene_type == "postcard_document"
        card_rgb = rng.integers(205, 246, size=3, dtype=np.uint8)
        patch[:] = card_rgb
        border = 0 if full_bleed else max(2, round(minimum_edge * rng.uniform(0.018, 0.055)))
        x0, x1 = border, width - border
        y0, y1 = border, height - border
        frame = border
        mat = border
        frameless = full_bleed
        has_outer = not full_bleed
    else:
        mounted_poster = scene_type in {
            "mounted_wall_poster",
            "wall_poster_cluster",
            "far_wall_poster",
        }
        frameless = scene_type in {
            "frameless_artwork",
            "magazine_cover",
            "desk_magazine_far",
            "printed_poster",
            "closeup_artwork",
        }
        thin = scene_type == "thin_frame"
        thick = scene_type == "thick_frame"
        has_mat = scene_type in {"mat_artwork", "mat_frame_artwork"}
        dark_frame = scene_type in {"dark_frame_artwork", "easel_artwork"}
        frame_ratio = (
            0.0
            if frameless or mounted_poster
            else rng.uniform(0.035, 0.095)
            if dark_frame
            else rng.uniform(0.012, 0.028)
            if thin
            else rng.uniform(0.06, 0.12)
            if thick
            else rng.uniform(0.025, 0.065)
        )
        frame = max(0, round(minimum_edge * frame_ratio))
        mat_ratio = (
            rng.uniform(0.025, 0.075)
            if mounted_poster
            else rng.uniform(0.11, 0.20)
            if has_mat
            else rng.uniform(0.035, 0.085)
        )
        mat = (
            0
            if frameless
            else frame
            if dark_frame
            else max(frame + 2, round(minimum_edge * mat_ratio))
        )
        if mounted_poster:
            # 纸张外缘与印刷画芯必须是两个独立层级；胶带稍后画在背景上，
            # 不属于 outer_quad。
            patch[:] = rng.integers(218, 250, size=3, dtype=np.uint8)
        if not dark_frame:
            cv2.rectangle(
                patch,
                (frame, frame),
                (width - frame - 1, height - frame - 1),
                (220, 216, 205),
                -1,
            )
        y0, y1 = mat, height - mat
        x0, x1 = mat, width - mat
        has_outer = not frameless
    yy, xx = np.indices((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.float32)
    must_use_content = scene_type in {
        "magazine_cover",
        "desk_magazine_far",
        "mounted_wall_poster",
        "wall_poster_cluster",
        "far_wall_poster",
    }
    if content_paths and (must_use_content or rng.random() < 0.88):
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
    content = np.clip(content, 0, 255).astype(np.uint8)
    if scene_type in {"magazine_cover", "desk_magazine_far"}:
        _draw_magazine_cover_layout(content, rng)
    elif scene_type in {"mounted_wall_poster", "wall_poster_cluster", "far_wall_poster"}:
        _draw_printed_poster_layout(content, rng)
    screen_target: tuple[int, int, int, int] | None = None
    if target_class == "screen":
        if scene_type in {
            "monitor_photo_viewer",
            "monitor_browser_image",
            "cropped_monitor_photo_viewer",
            "cropped_monitor_browser_image",
            "portable_image_viewer",
        }:
            content, screen_target = _screen_viewer_layout(content, rng, scene_type)
        else:
            _draw_screen_interface(content, rng, scene_type)
    elif target_class == "postcard" and min(content.shape[:2]) >= 24:
        # 少量留白与邮戳式圆形让 postcard 具有域线索，同时保留公开纹理主体。
        cv2.rectangle(content, (0, 0), (content.shape[1] - 1, content.shape[0] - 1), (238, 234, 220), 1)
        if rng.random() < 0.45:
            radius = max(3, min(content.shape[:2]) // 10)
            cv2.circle(content, (content.shape[1] - radius - 3, radius + 3), radius, (90, 80, 75), 1)
    patch[y0:y1, x0:x1] = content
    if target_class == "screen":
        cv2.rectangle(patch, (x0, y0), (x1 - 1, y1 - 1), (12, 12, 12), 1)
        if scene_type == "portable_image_viewer":
            radius = max(2, bezel // 4)
            cv2.circle(patch, (max(radius + 1, bezel // 2), height // 2), radius, (70, 92, 105), -1)
            cv2.circle(
                patch,
                (min(width - radius - 2, width - bezel // 2), height // 2),
                radius,
                (145, 72, 55),
                -1,
            )
    elif not frameless:
        cv2.rectangle(patch, (x0, y0), (x1 - 1, y1 - 1), (52, 50, 47), 1)
    if screen_target is None:
        target_x0, target_y0, target_x1, target_y1 = x0, y0, x1 - 1, y1 - 1
    else:
        local_x0, local_y0, local_x1, local_y1 = screen_target
        target_x0, target_y0 = x0 + local_x0, y0 + local_y0
        target_x1, target_y1 = x0 + local_x1, y0 + local_y1
    content_quad = np.array(
        [
            [target_x0, target_y0],
            [target_x1, target_y0],
            [target_x1, target_y1],
            [target_x0, target_y1],
        ],
        dtype=np.float32,
    )
    return patch, content_quad, has_outer


def _scene_type(rng: np.random.Generator, target_class: str) -> str:
    values = {
        "artwork": (
            "artwork",
            "frameless_artwork",
            "magazine_cover",
            "magazine_cover",
            "desk_magazine_far",
            "desk_magazine_far",
            "printed_poster",
            "printed_poster",
            "mounted_wall_poster",
            "mounted_wall_poster",
            "wall_poster_cluster",
            "wall_poster_cluster",
            "far_wall_poster",
            "closeup_artwork",
            "thin_frame",
            "thick_frame",
            "mat_artwork",
            "mat_frame_artwork",
            "dark_frame_artwork",
            "easel_artwork",
            "glass_glare",
            "multiple_artworks",
            "far_artwork",
            "offcenter_artwork",
            "wall_poster",
            "near_border_artwork",
            "partial_artwork",
            "perspective",
        ),
        "screen": (
            "monitor_photo_viewer",
            "monitor_photo_viewer",
            "monitor_photo_viewer",
            "monitor_browser_image",
            "monitor_browser_image",
            "cropped_monitor_photo_viewer",
            "cropped_monitor_photo_viewer",
            "cropped_monitor_browser_image",
            "portable_image_viewer",
            "screen_content",
            "dark_bezel",
            "thin_bezel",
            "monitor_screen",
            "monitor_screen",
            "monitor_browser",
            "monitor_desktop",
            "monitor_thin_bezel",
            "monitor_dark_bezel",
            "bright_screen",
            "dark_screen",
            "reflection",
            "perspective",
            "partial_screen",
            "multiple_rectangular_distractors",
            "far_screen",
            "offcenter_screen",
            "desktop_screen",
            "multi_screen_ui",
        ),
        "postcard": (
            "postcard_document",
            "closeup_postcard",
            "closeup_postcard",
            "perspective",
            "near_border_document",
            "far_postcard",
            "offcenter_postcard",
            "desk_postcard",
        ),
    }[target_class]
    return str(rng.choice(values))


def _target_scale_range(scene_type: str) -> tuple[float, float]:
    """返回外框相对画布的边长范围，远拍样本保持可见但显著小于旧分布。"""

    if scene_type in {"far_wall_poster", "desk_magazine_far"}:
        return 0.18, 0.38
    if scene_type.startswith("far_"):
        return 0.14, 0.34
    if scene_type in {"offcenter_artwork", "offcenter_screen", "offcenter_postcard"}:
        return 0.24, 0.52
    if scene_type.startswith("closeup_") or scene_type in {
        "magazine_cover",
        "printed_poster",
        "monitor_photo_viewer",
        "monitor_browser_image",
        "portable_image_viewer",
    }:
        return 0.68, 0.90
    if scene_type in {"cropped_monitor_photo_viewer", "cropped_monitor_browser_image"}:
        return 1.04, 1.28
    return 0.40, 0.78


def _sample_patch_size(
    rng: np.random.Generator,
    size: int,
    minimum: float,
    maximum: float,
    target_class: str,
) -> tuple[int, int]:
    """按真实物体常见画幅采样尺寸，同时保留少量竖屏和竖版卡片。"""

    long_edge = float(rng.uniform(size * minimum, size * maximum))
    if target_class == "screen":
        mode = float(rng.random())
        aspect = (
            float(rng.uniform(1.45, 1.90))
            if mode < 0.78
            else float(rng.uniform(0.55, 0.78))
            if mode < 0.90
            else float(rng.uniform(0.85, 1.25))
        )
    elif target_class == "postcard":
        landscape = bool(rng.random() < 0.78)
        card_aspect = float(rng.uniform(1.35, 1.72))
        aspect = card_aspect if landscape else 1.0 / card_aspect
    else:
        aspect = float(rng.uniform(0.62, 1.62))
    if aspect >= 1.0:
        width, height = long_edge, long_edge / aspect
    else:
        width, height = long_edge * aspect, long_edge
    return max(32, round(width)), max(32, round(height))


def _sample_center(
    rng: np.random.Generator,
    size: int,
    patch_width: int,
    patch_height: int,
    margin: float,
    scene_type: str,
) -> np.ndarray:
    """在完整可见约束内采样中心；偏心场景主动避开中央舒适区。"""

    half_width = patch_width / 2
    half_height = patch_height / 2
    x_min, x_max = margin + half_width, size - 1 - margin - half_width
    y_min, y_max = margin + half_height, size - 1 - margin - half_height
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)
    offcenter = scene_type.startswith("offcenter_") or scene_type.startswith("far_")
    if offcenter:
        candidates = np.asarray(
            [
                [x_min, rng.uniform(y_min, y_max)],
                [x_max, rng.uniform(y_min, y_max)],
                [rng.uniform(x_min, x_max), y_min],
                [rng.uniform(x_min, x_max), y_max],
            ],
            dtype=np.float32,
        )
        center = candidates[int(rng.integers(0, len(candidates)))].copy()
        center += rng.normal(0.0, size * 0.025, size=2).astype(np.float32)
        center[0] = np.clip(center[0], x_min, x_max)
        center[1] = np.clip(center[1], y_min, y_max)
        return center
    return np.asarray(
        [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32
    )


def _draw_screen_interface(
    content: np.ndarray,
    rng: np.random.Generator,
    scene_type: str,
) -> None:
    """在屏幕显示区内加入浏览器/桌面层次，不改变 content_quad 的语义边界。"""

    height, width = content.shape[:2]
    if min(height, width) < 18:
        return
    desktop_scene = scene_type in {
        "desktop_screen",
        "multi_screen_ui",
        "monitor_browser",
        "monitor_desktop",
    }
    if desktop_scene:
        bar_height = max(3, height // 12)
        cv2.rectangle(content, (0, 0), (width - 1, bar_height), (32, 36, 44), -1)
        for index in range(3):
            cv2.circle(
                content,
                (
                    max(2, bar_height // 2 + index * max(3, bar_height // 2)),
                    bar_height // 2,
                ),
                max(1, bar_height // 6),
                (210, 105 + index * 35, 80),
                -1,
            )
        sidebar_width = max(3, width // 13)
        cv2.rectangle(
            content,
            (0, bar_height + 1),
            (sidebar_width, height - 1),
            (43, 48, 58),
            -1,
        )
        panel_count = int(rng.integers(2, 5))
        for index in range(panel_count):
            x0 = int((index + 0.25) * width / panel_count)
            x1 = int((index + 0.85) * width / panel_count)
            y0 = bar_height + max(2, height // 12)
            y1 = height - max(2, height // 12)
            cv2.rectangle(content, (x0, y0), (min(width - 1, x1), y1), (225, 227, 231), 1)
            # 用短线模拟文本与工具区，不绘制可识别内容。
            for row in range(2, 6):
                line_y = y0 + row * max(2, (y1 - y0) // 7)
                if line_y < y1:
                    cv2.line(
                        content,
                        (x0 + 2, line_y),
                        (min(width - 1, x1 - 2), line_y),
                        (205, 208, 214),
                        1,
                    )
    # 屏摄常见的轻微扫描线只作为弱证据，避免覆盖原有公开内容纹理。
    if rng.random() < 0.55:
        alpha = float(rng.uniform(0.025, 0.075))
        scanline = content.copy()
        step = int(rng.integers(3, 7))
        scanline[::step] = np.clip(scanline[::step].astype(np.float32) * 0.72, 0, 255)
        cv2.addWeighted(scanline, alpha, content, 1.0 - alpha, 0.0, dst=content)


def _screen_viewer_layout(
    source: np.ndarray,
    rng: np.random.Generator,
    scene_type: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """生成屏幕 UI，并返回其中真正需要电子化的内层图片矩形。

    工具栏、留白和物理 bezel 只提供 screen 类别上下文，不能进入 ``content_quad``。
    """

    height, width = source.shape[:2]
    cropped_scene = scene_type.startswith("cropped_monitor_")
    browser_scene = "browser" in scene_type
    dark_ui = scene_type == "portable_image_viewer" or bool(rng.random() < 0.18)
    base = int(rng.integers(24, 52)) if dark_ui else int(rng.integers(224, 250))
    canvas = np.full((height, width, 3), base, np.uint8)
    top_bar = max(3, round(height * rng.uniform(0.065, 0.12)))
    bottom_bar = max(2, round(height * rng.uniform(0.025, 0.07)))
    bar_color = (
        tuple(int(value) for value in rng.integers(30, 62, size=3))
        if dark_ui
        else tuple(int(value) for value in rng.integers(238, 253, size=3))
    )
    cv2.rectangle(canvas, (0, 0), (width - 1, top_bar), bar_color, -1)
    cv2.line(
        canvas,
        (0, top_bar),
        (width - 1, top_bar),
        (75, 78, 82) if dark_ui else (190, 193, 198),
        1,
    )
    icon_color = (185, 188, 194) if dark_ui else (92, 96, 104)
    for index in range(4):
        center = (max(2, width - (index + 1) * max(5, width // 18)), top_bar // 2)
        cv2.circle(canvas, center, max(1, top_bar // 9), icon_color, 1)

    if browser_scene:
        # 浏览器 tab、地址栏和滚动条提供“图片位于屏幕 UI 内”的整图证据。
        tab_height = max(2, top_bar // 3)
        cv2.rectangle(canvas, (0, 0), (max(8, width // 3), tab_height), icon_color, 1)
        address_y = min(top_bar - 1, tab_height + max(1, top_bar // 4))
        cv2.line(
            canvas,
            (max(3, width // 12), address_y),
            (max(4, width - width // 8), address_y),
            icon_color,
            1,
        )
        cv2.line(canvas, (width - 2, top_bar + 1), (width - 2, height - 2), icon_color, 1)

    portrait = bool(rng.random() < 0.52)
    target_aspect = float(rng.uniform(0.58, 0.86) if portrait else rng.uniform(1.18, 1.82))
    # 私人开发集中的图片查看器/网页通常让目标图占据可用高度的主要部分；
    # 过小的内层图会把任务错误地推向“网页缩略图检测”。保留少量留白和工具栏，
    # 但让真正需要电子化的图片成为屏幕内的显著主体。
    width_fraction = (
        0.76
        if cropped_scene
        else 0.86
        if browser_scene
        else 0.92
    )
    available_width = max(8, round(width * width_fraction))
    available_height = max(8, height - top_bar - bottom_bar - max(4, height // 18))
    scale = float(
        rng.uniform(0.60, 0.76)
        if cropped_scene
        else rng.uniform(0.74, 0.96)
        if browser_scene
        else rng.uniform(0.80, 0.98)
    )
    if available_width / available_height >= target_aspect:
        target_height = max(6, round(available_height * scale))
        target_width = max(6, round(target_height * target_aspect))
    else:
        target_width = max(6, round(available_width * scale))
        target_height = max(6, round(target_width / target_aspect))
    target_width = min(target_width, width - 4)
    target_height = min(target_height, available_height)
    x0 = max(2, (width - target_width) // 2 + round(rng.uniform(-0.04, 0.04) * width))
    y_min = top_bar + 2
    y0 = max(y_min, y_min + (available_height - target_height) // 2)
    x0 = min(x0, width - target_width - 2)
    y0 = min(y0, height - bottom_bar - target_height - 1)
    x1, y1 = x0 + target_width, y0 + target_height
    target_source = _center_crop_aspect(source, target_aspect)
    target = cv2.resize(target_source, (target_width, target_height), interpolation=cv2.INTER_AREA)
    canvas[y0:y1, x0:x1] = target
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (12, 12, 12), 1)

    # 图片查看器常见的底部任务栏/操作区和缩略图不进入 content_quad。
    taskbar_y = max(y1 + 1, height - bottom_bar)
    if taskbar_y < height:
        cv2.rectangle(canvas, (0, taskbar_y), (width - 1, height - 1), bar_color, -1)
        icon_radius = max(1, bottom_bar // 6)
        for index in range(min(6, max(2, width // max(8, bottom_bar * 3)))):
            icon_x = width // 2 + (index - 2) * max(3, bottom_bar // 2)
            if 0 <= icon_x < width:
                cv2.circle(canvas, (icon_x, min(height - 1, taskbar_y + bottom_bar // 2)), icon_radius, icon_color, 1)

    if rng.random() < 0.82:
        step = int(rng.integers(3, 7))
        canvas[::step] = np.clip(
            canvas[::step].astype(np.float32) * rng.uniform(0.91, 0.98), 0, 255
        )
    yy, xx = np.indices((height, width), dtype=np.float32)
    phase = float(rng.uniform(0.0, np.pi * 2.0))
    pattern = np.sin(xx * rng.uniform(0.12, 0.32) + yy * rng.uniform(0.04, 0.16) + phase)
    canvas = np.clip(
        canvas.astype(np.float32) + pattern[:, :, None] * rng.uniform(0.6, 2.2),
        0,
        255,
    )
    return canvas.astype(np.uint8), (x0, y0, x1 - 1, y1 - 1)


def _center_crop_aspect(image: np.ndarray, target_aspect: float) -> np.ndarray:
    """中心裁剪到目标宽高比，不改变输入数组。"""

    height, width = image.shape[:2]
    source_aspect = width / max(1, height)
    if source_aspect > target_aspect:
        crop_width = max(1, round(height * target_aspect))
        x0 = (width - crop_width) // 2
        return image[:, x0 : x0 + crop_width].copy()
    crop_height = max(1, round(width / max(target_aspect, 1e-6)))
    y0 = (height - crop_height) // 2
    return image[y0 : y0 + crop_height].copy()


def _draw_planar_shadow(
    image: np.ndarray,
    outer_quad: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """在粘贴平面对象之前绘制柔和投影，减少合成剪贴感。"""

    if rng.random() >= 0.82:
        return
    height, width = image.shape[:2]
    offset = rng.normal(
        loc=[width * 0.012, height * 0.016],
        scale=[width * 0.008, height * 0.008],
    ).astype(np.float32)
    shadow = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(shadow, np.rint(outer_quad + offset).astype(np.int32), 255)
    sigma = float(rng.uniform(3.0, 11.0))
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigma).astype(np.float32) / 255.0
    strength = float(rng.uniform(0.08, 0.24))
    image[:] = np.clip(
        image.astype(np.float32) * (1.0 - shadow[:, :, None] * strength),
        0,
        255,
    ).astype(np.uint8)


def _apply_camera_degradation(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """施加受限白平衡、暗角、轻模糊与传感器噪声。"""

    working = image.astype(np.float32)
    gains = rng.uniform(0.88, 1.12, size=3).astype(np.float32)
    working *= gains[None, None] * float(rng.uniform(0.86, 1.12))
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    radius = np.sqrt(
        ((xx - (width - 1) / 2) / max(1, width / 2)) ** 2
        + ((yy - (height - 1) / 2) / max(1, height / 2)) ** 2
    )
    vignette = 1.0 - np.clip(radius, 0.0, 1.0) ** 2 * float(rng.uniform(0.0, 0.16))
    working *= vignette[:, :, None]
    if rng.random() < 0.62:
        sigma = float(rng.uniform(0.25, 0.85))
        working = cv2.GaussianBlur(working, (0, 0), sigma)
    noise = rng.normal(0.0, rng.uniform(1.0, 5.5), working.shape)
    return np.clip(working + noise, 0, 255).astype(np.uint8)


def _draw_monitor_support(
    image: np.ndarray,
    outer_quad: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """在屏幕矩形之后绘制简化支架；支架不属于 outer/content 标注。"""

    top_mid = (outer_quad[0] + outer_quad[1]) * 0.5
    bottom_mid = (outer_quad[2] + outer_quad[3]) * 0.5
    down = bottom_mid - top_mid
    length = max(float(np.linalg.norm(down)), 1.0)
    down /= length
    bottom_width = float(np.linalg.norm(outer_quad[2] - outer_quad[3]))
    across = outer_quad[2] - outer_quad[3]
    across /= max(float(np.linalg.norm(across)), 1.0)
    stem_length = max(4.0, length * float(rng.uniform(0.10, 0.19)))
    stem_width = max(2, round(bottom_width * 0.035))
    stem_end = bottom_mid + down * stem_length
    color = tuple(int(value) for value in rng.integers(18, 58, size=3))
    cv2.line(
        image,
        tuple(np.rint(bottom_mid).astype(int)),
        tuple(np.rint(stem_end).astype(int)),
        color,
        stem_width,
        cv2.LINE_AA,
    )
    half_base = across * bottom_width * float(rng.uniform(0.15, 0.28))
    cv2.line(
        image,
        tuple(np.rint(stem_end - half_base).astype(int)),
        tuple(np.rint(stem_end + half_base).astype(int)),
        color,
        max(2, stem_width // 2),
        cv2.LINE_AA,
    )


def _draw_screen_environment(image: np.ndarray, rng: np.random.Generator) -> None:
    """加入桌面、墙面和次要显示器轮廓，提供真实屏摄上下文。

    次要显示器刻意保持较小、低对比度且部分截断，避免把正样本改造成“多个同等候选”
    的拒绝场景。正式目标随后绘制在最上层。
    """

    height, width = image.shape[:2]
    room_tint = rng.integers(24, 105, size=3).astype(np.float32)
    image[:] = np.clip(image.astype(np.float32) * 0.38 + room_tint * 0.62, 0, 255).astype(
        np.uint8
    )
    desk_y = int(rng.uniform(height * 0.68, height * 0.88))
    desk_color = tuple(int(value) for value in rng.integers(55, 145, size=3))
    cv2.rectangle(image, (0, desk_y), (width - 1, height - 1), desk_color, -1)
    if rng.random() < 0.78:
        distractor_width = int(rng.uniform(width * 0.18, width * 0.34))
        distractor_height = int(distractor_width / rng.uniform(1.45, 1.85))
        x0 = int(rng.choice((-distractor_width // 3, width - distractor_width * 2 // 3)))
        y0 = int(rng.uniform(height * 0.04, height * 0.30))
        x1, y1 = x0 + distractor_width, y0 + distractor_height
        bezel = max(2, distractor_height // 16)
        cv2.rectangle(image, (x0, y0), (x1, y1), (18, 21, 25), -1)
        cv2.rectangle(
            image,
            (x0 + bezel, y0 + bezel),
            (x1 - bezel, y1 - bezel),
            tuple(int(value) for value in rng.integers(70, 155, size=3)),
            -1,
        )
        for row in range(3):
            line_y = y0 + bezel * 2 + row * max(3, distractor_height // 6)
            cv2.line(
                image,
                (x0 + bezel * 2, line_y),
                (x1 - bezel * 2, line_y),
                    (165, 169, 176),
                    1,
                )


def _draw_wood_desk_context(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    reserved_quad: np.ndarray,
) -> None:
    """生成有细木纹和相邻课桌边缘的平面背景，纸面稍后覆盖其上。"""

    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    # 纵向木纹来自一维相关噪声与轻微倾斜纹线；光照沿桌面平滑变化。
    red = float(rng.uniform(171, 205))
    green = red - float(rng.uniform(21, 39))
    blue = green - float(rng.uniform(29, 48))
    base_rgb = np.asarray([red, green, blue], dtype=np.float32)
    columns = rng.normal(0.0, 13.0, width).astype(np.float32)
    correlated = cv2.GaussianBlur(columns[None, :], (0, 0), 0.7)[0]
    fiber = rng.normal(0.0, 34.0, (height, width)).astype(np.float32)
    fiber = cv2.GaussianBlur(fiber, (0, 0), sigmaX=0.8, sigmaY=10.0)
    phase = xx * float(rng.uniform(0.10, 0.24)) + yy * float(rng.uniform(-0.03, 0.03))
    grain = correlated[None, :] + fiber + np.sin(phase) * 1.5
    shade = (xx / max(1, width) - 0.5) * rng.uniform(-16, 16)
    shade += (yy / max(1, height) - 0.5) * rng.uniform(-22, 22)
    sensor = rng.normal(0.0, 3.2, image.shape).astype(np.float32)
    image[:] = np.clip(
        base_rgb[None, None, :] + grain[:, :, None] + shade[:, :, None] + sensor,
        0,
        255,
    ).astype(np.uint8)
    # 侧边的另一张课桌只提供环境线索，不参与主目标四角或内容层监督。
    if rng.random() < 0.58:
        place_left = float(np.mean(reserved_quad[:, 0])) > width * 0.5
        edge = int(width * rng.uniform(0.08, 0.24))
        if place_left:
            polygon = np.asarray(
                [[0, 0], [edge, 0], [max(0, edge - width // 12), height - 1], [0, height - 1]],
                np.int32,
            )
        else:
            polygon = np.asarray(
                [[width - edge, 0], [width - 1, 0], [width - 1, height - 1],
                 [min(width - 1, width - edge + width // 12), height - 1]],
                np.int32,
            )
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        shifted = np.clip(image.astype(np.int16) + int(rng.integers(-35, 26)), 0, 255)
        image[mask > 0] = shifted[mask > 0].astype(np.uint8)
        cv2.polylines(image, [polygon], True, (90, 73, 54), max(2, width // 220))
    if rng.random() < 0.38:
        seam_y = int(rng.uniform(height * 0.72, height * 0.95))
        cv2.line(image, (0, seam_y), (width - 1, seam_y), (121, 94, 66), 2)


def _draw_poster_tape(
    image: np.ndarray,
    rng: np.random.Generator,
    outer_quad: np.ndarray,
) -> None:
    """在纸张四角附近叠加半透明胶带；画芯标注仍指向印刷内容。"""

    top = outer_quad[1] - outer_quad[0]
    bottom = outer_quad[2] - outer_quad[3]
    side_length = float(np.linalg.norm(outer_quad[3] - outer_quad[0]))
    overlay = image.copy()
    drawn = False
    for corner, edge, direction, top_edge in (
        (outer_quad[0], top, 1.0, True),
        (outer_quad[1], top, -1.0, True),
        (outer_quad[3], bottom, 1.0, False),
        (outer_quad[2], bottom, -1.0, False),
    ):
        if rng.random() < 0.18:
            continue
        length = max(float(np.linalg.norm(edge)), 1.0)
        along = edge / length
        outward = np.asarray([along[1], -along[0]], np.float32)
        if not top_edge:
            outward *= -1.0
        center = corner + along * direction * length * 0.07 + outward * side_length * 0.025
        half_length = length * float(rng.uniform(0.09, 0.15))
        half_width = side_length * float(rng.uniform(0.023, 0.042))
        polygon = np.rint(
            np.asarray(
                [
                    center - along * half_length - outward * half_width,
                    center + along * half_length - outward * half_width,
                    center + along * half_length + outward * half_width,
                    center - along * half_length + outward * half_width,
                ]
            )
        ).astype(np.int32)
        cv2.fillConvexPoly(overlay, polygon, (218, 207, 168))
        drawn = True
    if drawn:
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0, dst=image)


def _draw_wall_poster_context(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    clustered: bool,
    content_paths: list[Path],
    reserved_quad: np.ndarray,
) -> None:
    """生成带墙面结构及多张图文海报的环境；邻近海报始终是无标注干扰层。"""

    height, width = image.shape[:2]
    wall_base = float(rng.uniform(194, 240))
    wall_color = np.clip(
        wall_base + rng.normal(0.0, 5.0, size=3), 185, 247
    ).astype(np.float32)
    noise = rng.normal(0.0, 3.0, image.shape).astype(np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    plaster = np.sin(xx * 0.06 + yy * 0.014) * rng.uniform(1.5, 3.5)
    illumination = (xx / max(1, width) - 0.5) * rng.uniform(-24, 24)
    image[:] = np.clip(
        wall_color[None, None] + noise + plaster[:, :, None] + illumination[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    if rng.random() < 0.42:
        # 门洞或墙角放在主海报中心相反的一侧，增加远拍场景结构。
        reserved_center = float(np.mean(reserved_quad[:, 0]))
        doorway_width = int(width * rng.uniform(0.09, 0.23))
        x0 = 0 if reserved_center > width * 0.5 else width - doorway_width
        cv2.rectangle(image, (x0, 0), (x0 + doorway_width, height - 1),
                      tuple(int(value) for value in rng.integers(43, 105, size=3)), -1)
        cv2.line(image, (x0, 0), (x0, height - 1), (139, 139, 135), 2)
    if rng.random() < 0.72:
        seam_x = int(rng.uniform(width * 0.12, width * 0.88))
        cv2.line(image, (seam_x, 0), (seam_x, height - 1), (176, 178, 174), 1)
    if rng.random() < 0.68:
        baseboard_y = int(rng.uniform(height * 0.78, height * 0.94))
        cv2.rectangle(
            image,
            (0, baseboard_y),
            (width - 1, height - 1),
            tuple(int(value) for value in rng.integers(45, 115, size=3)),
            -1,
        )
    reserved_x0, reserved_y0 = np.min(reserved_quad, axis=0)
    reserved_x1, reserved_y1 = np.max(reserved_quad, axis=0)
    distractor_count = int(rng.integers(3, 7)) if clustered else int(rng.integers(0, 3))
    for _ in range(distractor_count):
        poster_width = int(rng.uniform(width * 0.12, width * (0.35 if clustered else 0.25)))
        poster_height = int(poster_width / rng.uniform(0.48, 0.95))
        for _attempt in range(16):
            if clustered and rng.random() < 0.35:
                # 截断的邻海报也是有效干扰，但不应被主海报大面积遮盖。
                x0 = int(rng.choice((-poster_width // 3, width - poster_width * 2 // 3)))
            else:
                x0 = int(rng.uniform(0, max(1, width - poster_width)))
            y0 = int(rng.uniform(-poster_height * 0.12, max(1, height * 0.78 - poster_height)))
            overlap_width = max(0.0, min(x0 + poster_width, reserved_x1) - max(x0, reserved_x0))
            overlap_height = max(0.0, min(y0 + poster_height, reserved_y1) - max(y0, reserved_y0))
            if overlap_width * overlap_height <= poster_width * poster_height * 0.08:
                break
        else:
            continue
        _draw_textured_neighbor_poster(
            image,
            rng,
            (x0, y0),
            (poster_width, poster_height),
            content_paths,
        )


def _draw_textured_neighbor_poster(
    image: np.ndarray,
    rng: np.random.Generator,
    origin: tuple[int, int],
    shape: tuple[int, int],
    content_paths: list[Path],
) -> None:
    """绘制带独立纸边、图像主体和图文字块的透视邻近海报。"""

    width, height = shape
    if width < 12 or height < 12:
        return
    paper = rng.integers(194, 251, size=3, dtype=np.uint8)
    patch = np.empty((height, width, 3), np.uint8)
    patch[:] = paper
    border = max(2, round(min(width, height) * rng.uniform(0.025, 0.10)))
    inner_width = max(1, width - 2 * border)
    inner_height = max(1, height - 2 * border)
    if content_paths:
        content = _texture_crop(
            content_paths[int(rng.integers(0, len(content_paths)))],
            inner_width,
            inner_height,
            rng,
        )
    else:
        # 无外部纹理时仍生成多色局部图形，不能回到纯色矩形的旧分布。
        content = np.empty((inner_height, inner_width, 3), np.uint8)
        content[:] = rng.integers(28, 210, size=3, dtype=np.uint8)
        for _ in range(5):
            start = tuple(int(value) for value in rng.integers((inner_width, inner_height)))
            end = tuple(int(value) for value in rng.integers((inner_width, inner_height)))
            color = tuple(int(value) for value in rng.integers(20, 245, size=3))
            cv2.line(content, start, end, color, max(2, inner_width // 18))
    patch[border : border + inner_height, border : border + inner_width] = content
    if min(width, height) >= 45:
        # 图文块模仿海报信息层，但不生成可辨识的第三方品牌或人名。
        band_y = int(height * rng.uniform(0.10, 0.72))
        band_height = max(10, int(height * rng.uniform(0.12, 0.27)))
        ink = tuple(int(value) for value in rng.integers(8, 95, size=3))
        cv2.rectangle(
            patch,
            (border, band_y),
            (width - border - 1, min(height - border - 1, band_y + band_height)),
            (232, 228, 218),
            -1,
        )
        for row in range(3):
            line_y = band_y + 3 + row * max(3, band_height // 4)
            if line_y >= height - border:
                break
            line_width = int(inner_width * rng.uniform(0.35, 0.88))
            cv2.line(patch, (border + 3, line_y), (border + line_width, line_y), ink, 1)
    x0, y0 = origin
    quad = np.asarray(
        [[x0, y0], [x0 + width - 1, y0], [x0 + width - 1, y0 + height - 1], [x0, y0 + height - 1]],
        np.float32,
    )
    quad += rng.normal(0.0, min(width, height) * 0.035, size=(4, 2)).astype(np.float32)
    source = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, quad)
    canvas_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(patch, transform, canvas_size)
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), transform, canvas_size)
    image[mask > 0] = warped[mask > 0]


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


def _draw_artwork_easel(
    image: np.ndarray,
    outer_quad: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """在画作后方绘制简化双腿画架，作为 monitor 支架线索的反事实样本。"""

    top_mid = (outer_quad[0] + outer_quad[1]) * 0.5
    bottom_mid = (outer_quad[2] + outer_quad[3]) * 0.5
    down = bottom_mid - top_mid
    height = max(float(np.linalg.norm(down)), 1.0)
    down /= height
    across = outer_quad[2] - outer_quad[3]
    width = max(float(np.linalg.norm(across)), 1.0)
    across /= width
    color = tuple(int(value) for value in rng.integers(65, 135, size=3))
    end = bottom_mid + down * height * float(rng.uniform(0.20, 0.34))
    spread = across * width * float(rng.uniform(0.20, 0.33))
    thickness = max(2, round(width * 0.018))
    for foot in (end - spread, end + spread):
        cv2.line(
            image,
            tuple(np.rint(bottom_mid).astype(int)),
            tuple(np.rint(foot).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


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
    # 代理终端会保留每次回车刷新；把长任务限制在约 100 次更新内。
    interval = max(1, (total + 99) // 100)
    if done < total and done not in {0, 1} and done % interval != 0:
        return
    width = 28
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if done >= total else "\r"
    print(f"[{bar}] {done:>6}/{total:<6} {message}", end=ending, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
