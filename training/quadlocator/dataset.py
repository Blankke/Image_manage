"""QuadLocator JSONL 数据读取与监督图生成。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_INDEX = {"artwork": 0, "postcard": 1, "screen": 2, "none": 3}
VALID_SPLITS = {"train", "validation", "test"}


class QuadDataset(Dataset[dict[str, torch.Tensor]]):
    """按 split 读取图片，生成 1/4 分辨率热图、mask 与 boundary。"""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        image_size: int = 640,
        heatmap_sigma: float = 2.5,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = self.manifest_path.parent
        self.image_size = image_size
        self.output_size = image_size // 4
        self.heatmap_sigma = heatmap_sigma
        if image_size < 128 or image_size % 32:
            raise ValueError("训练输入尺寸必须不小于 128 且为 32 的倍数")
        self.records = [
            record for record in _read_manifest(self.manifest_path) if record.get("split") == split
        ]
        if not self.records:
            raise ValueError(f"manifest 中没有 split={split} 的样本")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        path = (self.root / str(record["image"])).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("manifest image 不能越出数据目录")
        with Image.open(path) as source:
            source_rgb = np.asarray(source.convert("RGB"))
        image, transform = _letterbox_image(source_rgb, self.image_size)
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        present = bool(record["present"])
        target_class = str(record["target_class"])
        if target_class not in CLASS_INDEX:
            raise ValueError(f"未知 target_class：{target_class}")
        content_quad = _transform_quad(_quad(record.get("content_quad"), present), transform)
        outer_quad = _transform_quad(_quad(record.get("outer_quad"), False), transform)
        content_heatmaps = _corner_heatmaps(content_quad, self.output_size, self.heatmap_sigma)
        outer_heatmaps = _corner_heatmaps(outer_quad, self.output_size, self.heatmap_sigma)
        content_mask = _polygon_mask(content_quad, self.output_size, filled=True)
        boundary = _polygon_mask(content_quad, self.output_size, filled=False)
        target_corners = (
            content_quad.astype(np.float32)
            if content_quad is not None
            else np.zeros((4, 2), dtype=np.float32)
        )
        return {
            "image": image_tensor,
            "content_corner_heatmaps": torch.from_numpy(content_heatmaps),
            "outer_corner_heatmaps": torch.from_numpy(outer_heatmaps),
            "content_mask": torch.from_numpy(content_mask[None]),
            "boundary": torch.from_numpy(boundary[None]),
            "presence": torch.tensor([float(present)], dtype=torch.float32),
            "target_class": torch.tensor(CLASS_INDEX[target_class], dtype=torch.long),
            "content_corners": torch.from_numpy(target_corners),
            "outer_present": torch.tensor([float(outer_quad is not None)], dtype=torch.float32),
        }


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest 第 {line_number} 行必须是对象")
            for required in ("image", "split", "present", "target_class", "group_id"):
                if required not in record:
                    raise ValueError(f"manifest 第 {line_number} 行缺少 {required}")
            if record["split"] not in VALID_SPLITS:
                raise ValueError(f"manifest 第 {line_number} 行 split 无效")
            if record["target_class"] not in CLASS_INDEX:
                raise ValueError(f"manifest 第 {line_number} 行 target_class 无效")
            if bool(record["present"]) == (record["target_class"] == "none"):
                raise ValueError(f"manifest 第 {line_number} 行 present 与 target_class 矛盾")
            records.append(record)
    _validate_group_isolation(records)
    return records


def _validate_group_isolation(records: list[dict[str, Any]]) -> None:
    """防止同一作品或连拍会话泄漏到多个数据 split。"""

    assignments: dict[tuple[str, str], str] = {}
    for record in records:
        split = str(record["split"])
        identifiers = [("group_id", str(record["group_id"]))]
        capture_session = record.get("capture_session")
        if capture_session:
            identifiers.append(("capture_session", str(capture_session)))
        for kind, value in identifiers:
            key = (kind, value)
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    f"数据泄漏：{kind}={value!r} 同时出现在 {previous} 与 {split}"
                )


def _quad(value: Any, required: bool) -> np.ndarray | None:
    if value is None:
        if required:
            raise ValueError("present 样本必须提供 content_quad")
        return None
    quad = np.asarray(value, dtype=np.float32)
    if quad.shape != (4, 2) or np.any(~np.isfinite(quad)) or np.any((quad < 0) | (quad > 1)):
        raise ValueError("quad 必须是 [0,1] 范围内的 4×2 数组")
    return quad


def _letterbox_image(
    image_rgb: np.ndarray,
    size: int,
) -> tuple[np.ndarray, tuple[int, int, float, float, int, int, int]]:
    """与产品 ONNX 运行时一致地保持比例缩放并居中填充。"""

    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=interpolation)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    scale_x = (resized_width - 1) / max(1, width - 1)
    scale_y = (resized_height - 1) / max(1, height - 1)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas, (width, height, scale_x, scale_y, offset_x, offset_y, size)


def _transform_quad(
    quad: np.ndarray | None,
    transform: tuple[int, int, float, float, int, int, int],
) -> np.ndarray | None:
    """把原图归一化四角转换到 letterbox 训练坐标。"""

    if quad is None:
        return None
    width, height, scale_x, scale_y, offset_x, offset_y, size = transform
    pixels = quad * np.array([max(1, width - 1), max(1, height - 1)], dtype=np.float32)
    model = pixels * np.array([scale_x, scale_y], dtype=np.float32)
    model += np.array([offset_x, offset_y], dtype=np.float32)
    return np.clip(model / max(1, size - 1), 0.0, 1.0).astype(np.float32)


def _corner_heatmaps(
    quad: np.ndarray | None,
    output_size: int,
    sigma: float,
) -> np.ndarray:
    heatmaps = np.zeros((4, output_size, output_size), dtype=np.float32)
    if quad is None:
        return heatmaps
    radius = max(2, round(sigma * 4))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    for index, point in enumerate(quad):
        x = int(round(float(point[0]) * (output_size - 1)))
        y = int(round(float(point[1]) * (output_size - 1)))
        x0, x1 = max(0, x - radius), min(output_size, x + radius + 1)
        y0, y1 = max(0, y - radius), min(output_size, y + radius + 1)
        gx0, gy0 = x0 - (x - radius), y0 - (y - radius)
        heatmaps[index, y0:y1, x0:x1] = gaussian[
            gy0 : gy0 + (y1 - y0), gx0 : gx0 + (x1 - x0)
        ]
    return heatmaps


def _polygon_mask(quad: np.ndarray | None, output_size: int, *, filled: bool) -> np.ndarray:
    mask = np.zeros((output_size, output_size), dtype=np.float32)
    if quad is None:
        return mask
    points = np.rint(quad * (output_size - 1)).astype(np.int32)
    if filled:
        cv2.fillConvexPoly(mask, points, 1.0)
    else:
        cv2.polylines(mask, [points], True, 1.0, thickness=max(1, output_size // 160))
    return mask
