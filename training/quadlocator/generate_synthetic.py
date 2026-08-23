"""生成无版权依赖的 QuadLocator 嵌套矩形合成训练集。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.quadlocator.generate_synthetic --output-directory /tmp/quad-synth --count 200

输出 ``manifest.jsonl`` 和 ``images/``。所有内容纹理均由 NumPy/OpenCV 程序生成；
脚本不会读取测试数据或网络素材，执行期间始终显示文本进度条。
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)
    if args.count < 3:
        raise ValueError("count 至少为 3")
    if args.size < 128 or args.size % 32:
        raise ValueError("size 必须不小于 128 且为 32 的倍数")
    if not 0.0 <= args.negative_ratio <= 0.5:
        raise ValueError("negative-ratio 必须位于 0..0.5")
    output_directory = args.output_directory.expanduser().resolve()
    image_directory = output_directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest_path = output_directory / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(args.count):
            _progress(index, args.count, "生成嵌套矩形与退化")
            negative = bool(rng.random() < args.negative_ratio)
            image, record = _sample(rng, args.size, index, negative)
            relative_path = Path("images") / f"sample_{index:06d}.jpg"
            target_path = output_directory / relative_path
            ok = cv2.imwrite(
                str(target_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(78, 98))],
            )
            if not ok:
                raise OSError(f"无法写入合成图：{target_path}")
            record["image"] = relative_path.as_posix()
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    _progress(args.count, args.count, f"完成：{manifest_path}")
    return 0


def _sample(
    rng: np.random.Generator,
    size: int,
    index: int,
    negative: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    background = _background(rng, size)
    split_value = index % 10
    split = "train" if split_value < 8 else "validation" if split_value == 8 else "test"
    base_record: dict[str, object] = {
        "split": split,
        "group_id": f"synthetic-{index:06d}",
        # 每个会话只能属于一个 split，便于训练入口直接执行泄漏检查。
        "capture_session": f"procedural-{split}-{index // 100:04d}",
        "device": "synthetic-pinhole",
        "visible": True,
        "occlusion": 0.0,
        "glare_level": "none",
    }
    if negative:
        # hard negative 保留墙面接缝和矩形阴影，但没有完整可归档目标。
        for _ in range(int(rng.integers(2, 7))):
            x1, y1 = rng.integers(0, size, size=2)
            x2, y2 = rng.integers(0, size, size=2)
            cv2.line(background, (int(x1), int(y1)), (int(x2), int(y2)), (80, 85, 90), 2)
        return background, {
            **base_record,
            "present": False,
            "target_class": "none",
            "content_quad": None,
            "outer_quad": None,
        }

    target_class = str(rng.choice(TARGET_CLASSES))
    patch_width = int(rng.integers(round(size * 0.48), round(size * 0.78)))
    patch_height = int(rng.integers(round(size * 0.42), round(size * 0.78)))
    patch, source_content = _nested_patch(rng, patch_width, patch_height, target_class)
    margin = size * 0.08
    center = np.array(
        [rng.uniform(size * 0.43, size * 0.57), rng.uniform(size * 0.43, size * 0.57)],
        dtype=np.float32,
    )
    half = np.array([patch_width / 2, patch_height / 2], dtype=np.float32)
    base_quad = np.array(
        [center - half, center + [-half[0], half[1]], center + half, center + [half[0], -half[1]]],
        dtype=np.float32,
    )[[0, 3, 2, 1]]
    jitter = rng.normal(0.0, size * 0.035, size=(4, 2)).astype(np.float32)
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
    if rng.random() < 0.38:
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
        "outer_quad": _normalized(outer_quad, size),
        "glare_level": glare,
    }


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
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
) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((height, width, 3), rng.integers(18, 90, size=3), dtype=np.uint8)
    frame = max(5, round(min(width, height) * rng.uniform(0.025, 0.07)))
    mat = max(frame + 5, round(min(width, height) * rng.uniform(0.07, 0.16)))
    cv2.rectangle(patch, (frame, frame), (width - frame - 1, height - frame - 1), (220, 216, 205), -1)
    y0, y1 = mat, height - mat
    x0, x1 = mat, width - mat
    yy, xx = np.indices((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.float32)
    if target_class == "screen":
        content = np.stack(
            ((xx * 1.7) % 255, (yy * 2.3) % 255, ((xx + yy) * 1.1) % 255), axis=2
        )
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
    return patch, content_quad


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
