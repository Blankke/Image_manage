"""QuadLocator JSONL 数据读取与监督图生成。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

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
        dataset_root: str | Path | None = None,
        max_samples: int = 0,
        augment: bool | None = None,
        seed: int = 20260823,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        # 标准清单的 image 相对于显式数据根目录，避免 manifests/ 与真实图片目录
        # 分离时出现隐式 ../ 路径，同时继续拒绝任何越出用户指定数据根的引用。
        self.root = (
            Path(dataset_root).expanduser().resolve()
            if dataset_root is not None
            else self.manifest_path.parent
        )
        self.image_size = image_size
        self.output_size = image_size // 4
        self.heatmap_sigma = heatmap_sigma
        self.split = split
        self.augment = split == "train" if augment is None else augment
        if split != "train" and self.augment:
            raise ValueError("validation/test 不允许随机 augmentation")
        self.seed = seed
        self.epoch = 0
        if image_size < 128 or image_size % 32:
            raise ValueError("训练输入尺寸必须不小于 128 且为 32 的倍数")
        self.records = [
            record for record in _read_manifest(self.manifest_path) if record.get("split") == split
        ]
        if not self.records:
            raise ValueError(f"manifest 中没有 split={split} 的样本")
        self.raw_sample_count = len(self.records)
        if max_samples < 0:
            raise ValueError("max_samples 不能为负数")
        if max_samples and max_samples < len(self.records):
            selected = np.random.default_rng(seed).choice(
                len(self.records), size=max_samples, replace=False
            )
            self.records = [self.records[int(index)] for index in sorted(selected)]

    def set_epoch(self, epoch: int) -> None:
        """让同一 seed 的逐 epoch 增强可复现，同时避免每轮固定同一变换。"""

        self.epoch = int(epoch)

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
        ambiguous = bool(record.get("ambiguous", False))
        if target_class not in CLASS_INDEX:
            raise ValueError(f"未知 target_class：{target_class}")
        content_quad = _transform_quad(_quad(record.get("content_quad"), present), transform)
        outer_quad = _transform_quad(_quad(record.get("outer_quad"), False), transform)
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            image, content_quad, outer_quad = _augment_sample(
                image,
                content_quad,
                outer_quad,
                rng,
            )
        content_heatmaps = _corner_heatmaps(content_quad, self.output_size, self.heatmap_sigma)
        outer_heatmaps = _corner_heatmaps(outer_quad, self.output_size, self.heatmap_sigma)
        content_mask = _polygon_mask(content_quad, self.output_size, filled=True)
        boundary = _polygon_mask(content_quad, self.output_size, filled=False)
        target_corners = (
            content_quad.astype(np.float32)
            if content_quad is not None
            else np.zeros((4, 2), dtype=np.float32)
        )
        target_outer_corners = (
            outer_quad.astype(np.float32)
            if outer_quad is not None
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
            "outer_corners": torch.from_numpy(target_outer_corners),
            "outer_present": torch.tensor([float(outer_quad is not None)], dtype=torch.float32),
            # 歧义标签不参与四角损失，专门用于验证阶段检查自动拒绝能力。
            "ambiguous": torch.tensor([float(ambiguous)], dtype=torch.float32),
        }

    def statistics(self) -> dict[str, object]:
        """返回可写入 run.json 的有限分布统计，不记录图片内容。"""

        classes = {name: 0 for name in CLASS_INDEX}
        content_presence = {"present": 0, "absent": 0}
        outer_presence = {"present": 0, "absent": 0}
        sources: dict[str, int] = {}
        ambiguous_distribution = {"yes": 0, "no": 0}
        groups: set[str] = set()
        for record in self.records:
            target_class = str(record["target_class"])
            classes[target_class] += 1
            present_key = "present" if bool(record["present"]) else "absent"
            content_presence[present_key] += 1
            outer_key = "present" if record.get("outer_quad") is not None else "absent"
            outer_presence[outer_key] += 1
            source = _record_source(record)
            sources[source] = sources.get(source, 0) + 1
            ambiguous_distribution["yes" if bool(record.get("ambiguous", False)) else "no"] += 1
            groups.add(str(record["group_id"]))
        return {
            "raw_sample_count": self.raw_sample_count,
            "selected_sample_count": len(self.records),
            "unique_group_count": len(groups),
            "class_distribution": classes,
            "content_presence_distribution": content_presence,
            "outer_presence_distribution": outer_presence,
            "dataset_source_distribution": dict(sorted(sources.items())),
            "ambiguous_distribution": ambiguous_distribution,
        }


class SourceGroupBalancedSampler(Sampler[int]):
    """先均匀选择 source，再均匀选择 group，最后选择该 group 的一帧。"""

    def __init__(self, dataset: QuadDataset, *, seed: int, samples_per_epoch: int = 0) -> None:
        if samples_per_epoch < 0:
            raise ValueError("samples_per_epoch 不能为负数")
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.sample_count = samples_per_epoch or len(dataset)
        grouped: dict[str, dict[str, list[int]]] = {}
        for index, record in enumerate(dataset.records):
            source = _record_source(record)
            group = str(record["group_id"])
            grouped.setdefault(source, {}).setdefault(group, []).append(index)
        self._grouped = grouped
        self._sources = sorted(grouped)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.sample_count

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        for _ in range(self.sample_count):
            source = self._sources[int(rng.integers(0, len(self._sources)))]
            groups = sorted(self._grouped[source])
            group = groups[int(rng.integers(0, len(groups)))]
            indices = self._grouped[source][group]
            yield indices[int(rng.integers(0, len(indices)))]


def _record_source(record: dict[str, Any]) -> str:
    explicit = record.get("source") or record.get("dataset")
    if explicit:
        return str(explicit)
    image_parts = Path(str(record["image"])).parts
    return image_parts[0] if len(image_parts) > 1 else "unknown"


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


def _augment_sample(
    image_rgb: np.ndarray,
    content_quad: np.ndarray | None,
    outer_quad: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """在 letterbox 坐标执行同步几何与摄影增强。

    四角使用同一单应矩阵更新。完整可见的正样本只接受四角仍在画幅内的变换；
    near-border 由平移与尺度联合产生，validation/test 不会进入本函数。
    """

    size = image_rgb.shape[0]
    matrix = _sample_homography(size, content_quad, outer_quad, rng)
    border = tuple(int(value) for value in np.median(image_rgb.reshape(-1, 3), axis=0))
    augmented = cv2.warpPerspective(
        image_rgb,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    content = _warp_normalized_quad(content_quad, matrix, size)
    outer = _warp_normalized_quad(outer_quad, matrix, size)
    augmented = _photometric_augmentation(augmented, rng)
    return augmented, content, outer


def _sample_homography(
    size: int,
    content_quad: np.ndarray | None,
    outer_quad: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    center = (size - 1) / 2.0
    source_corners = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    reference = outer_quad if outer_quad is not None else content_quad
    for _attempt in range(10):
        # target size variation 同时覆盖小目标与接近满幅目标；平移提高 near-border 占比。
        scale = float(rng.uniform(0.72, 1.18))
        angle = float(rng.uniform(-12.0, 12.0))
        affine = np.vstack(
            [cv2.getRotationMatrix2D((center, center), angle, scale), [0.0, 0.0, 1.0]]
        ).astype(np.float32)
        translation_limit = size * (0.18 if rng.random() < 0.35 else 0.10)
        affine[0, 2] += float(rng.uniform(-translation_limit, translation_limit))
        affine[1, 2] += float(rng.uniform(-translation_limit, translation_limit))
        perspective = source_corners + rng.normal(0.0, size * 0.025, (4, 2)).astype(np.float32)
        projective = cv2.getPerspectiveTransform(source_corners, perspective)
        matrix = projective @ affine
        if reference is None:
            return matrix.astype(np.float32)
        warped = _warp_normalized_quad(reference, matrix, size)
        if warped is None:
            continue
        # 正样本保持完整标注；1 像素安全带也避免 Gaussian 峰被裁断。
        if np.all((warped >= 1.0 / size) & (warped <= 1.0 - 1.0 / size)):
            area = abs(float(cv2.contourArea(warped.astype(np.float32))))
            if area >= 0.018:
                return matrix.astype(np.float32)
    return np.eye(3, dtype=np.float32)


def _warp_normalized_quad(
    quad: np.ndarray | None,
    matrix: np.ndarray,
    size: int,
) -> np.ndarray | None:
    if quad is None:
        return None
    pixels = quad.astype(np.float32) * max(1, size - 1)
    warped = cv2.perspectiveTransform(pixels[None], matrix.astype(np.float32))[0]
    return np.clip(warped / max(1, size - 1), 0.0, 1.0).astype(np.float32)


def _photometric_augmentation(
    image_rgb: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """组合轻量摄影退化，不引入外部模型或大型依赖。"""

    image = image_rgb.astype(np.float32) / 255.0
    # brightness / contrast / gamma / white balance / saturation
    image = (image - 0.5) * float(rng.uniform(0.75, 1.30)) + 0.5
    image *= float(rng.uniform(0.72, 1.30))
    gamma = float(rng.uniform(0.72, 1.38))
    image = np.power(np.clip(image, 0.0, 1.0), gamma)
    gains = rng.uniform(0.86, 1.14, size=3).astype(np.float32)
    gains[1] = float(rng.uniform(0.96, 1.04))
    image *= gains
    gray = np.sum(image * np.array([0.299, 0.587, 0.114], np.float32), axis=2, keepdims=True)
    image = gray + (image - gray) * float(rng.uniform(0.70, 1.30))

    height, width = image.shape[:2]
    if rng.random() < 0.35:
        # 平滑多边形阴影模拟画框、手部或室内方向光，不改变几何标签。
        shadow = np.ones((height, width), np.float32)
        points = rng.integers(0, max(height, width), size=(4, 2)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillConvexPoly(shadow, cv2.convexHull(points), float(rng.uniform(0.42, 0.82)))
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(3.0, width * 0.035))
        image *= shadow[:, :, None]
    if rng.random() < 0.30:
        # 局部高光保留为观测退化；椭圆边缘经模糊后近似玻璃反射。
        glare = np.zeros((height, width), np.float32)
        center = tuple(int(value) for value in rng.integers([0, 0], [width, height]))
        axes = (
            int(rng.integers(max(2, width // 16), max(3, width // 3))),
            int(rng.integers(max(2, height // 30), max(3, height // 8))),
        )
        cv2.ellipse(glare, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=max(2.0, width * 0.018))
        image = image * (1.0 - glare[:, :, None] * 0.18) + glare[:, :, None] * float(
            rng.uniform(0.25, 0.65)
        )

    image = np.clip(image, 0.0, 1.0)
    if rng.random() < 0.28:
        sigma = float(rng.uniform(0.25, 1.6))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    if rng.random() < 0.18:
        length = int(rng.integers(3, 12))
        angle = float(rng.uniform(0.0, math.pi))
        kernel = np.zeros((length, length), np.float32)
        center = (length - 1) / 2.0
        dx, dy = math.cos(angle) * center, math.sin(angle) * center
        cv2.line(
            kernel,
            (round(center - dx), round(center - dy)),
            (round(center + dx), round(center + dy)),
            1.0,
            1,
        )
        kernel /= max(float(kernel.sum()), 1.0)
        image = cv2.filter2D(image, -1, kernel)
    if rng.random() < 0.45:
        image += rng.normal(0.0, rng.uniform(0.002, 0.025), image.shape).astype(np.float32)
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        quality = int(rng.integers(48, 93))
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if ok:
            decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            image_uint8 = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    return image_uint8
