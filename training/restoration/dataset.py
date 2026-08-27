"""DIV2K HR 在线裁剪数据集，不创建预退化图片缓存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradation import CameraDegradationConfig, degrade_camera_image

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class Div2kHrDataset(Dataset[dict[str, torch.Tensor]]):
    """从 HR 原图随机裁剪，并对每次访问即时合成相机退化。"""

    def __init__(
        self,
        hr_directory: str | Path,
        *,
        patch_size: int = 192,
        degradation: CameraDegradationConfig | None = None,
        seed: int = 20260827,
        max_samples: int = 0,
    ) -> None:
        self.hr_directory = Path(hr_directory).expanduser().resolve()
        if not self.hr_directory.is_dir():
            raise ValueError(f"DIV2K HR 目录不存在：{self.hr_directory}")
        if patch_size < 64 or patch_size % 8:
            raise ValueError("patch_size 必须不小于 64 且为 8 的倍数")
        self.patch_size = patch_size
        self.degradation = degradation or CameraDegradationConfig()
        self.seed = seed
        self.epoch = 0
        paths = tuple(
            path for path in sorted(self.hr_directory.rglob("*")) if path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"DIV2K HR 目录没有可读图像：{self.hr_directory}")
        if max_samples < 0:
            raise ValueError("max_samples 不能为负数")
        if max_samples and max_samples < len(paths):
            selected = np.random.default_rng(seed).choice(len(paths), size=max_samples, replace=False)
            paths = tuple(paths[int(index)] for index in sorted(selected))
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """改变确定性随机种子，使每个 epoch 使用新的裁剪与退化。"""

        if epoch < 0:
            raise ValueError("epoch 不能为负数")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        with Image.open(path) as opened:
            source_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
        clean_crop = _random_crop(source_rgb, self.patch_size, rng)
        sample = degrade_camera_image(clean_crop, rng, self.degradation)
        return {
            "input": torch.from_numpy(sample.degraded_rgb.copy()).permute(2, 0, 1),
            "target": torch.from_numpy(clean_crop.copy()).permute(2, 0, 1).float() / 255.0,
        }


class UnlabeledIdentityDataset(Dataset[dict[str, torch.Tensor]]):
    """将显式指定的无标签图片仅用于输入恒等保护。

    这些图片没有 clean reference、四角或缺陷标签，因而绝不能作为恢复重建目标。
    数据集只返回原始裁剪，训练端只最小化 ``model(image) - image``，使 Fidelity 模型
    对用户自己的内容保持克制。调用方必须显式提供目录，绝不自动扫描 ``private``。
    """

    def __init__(
        self,
        image_directory: str | Path,
        *,
        patch_size: int = 192,
        seed: int = 20260827,
        max_samples: int = 0,
    ) -> None:
        self.image_directory = Path(image_directory).expanduser().resolve()
        if not self.image_directory.is_dir():
            raise ValueError(f"无标签 identity 目录不存在：{self.image_directory}")
        if patch_size < 64 or patch_size % 8:
            raise ValueError("patch_size 必须不小于 64 且为 8 的倍数")
        if max_samples < 0:
            raise ValueError("max_samples 不能为负数")
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0
        paths = tuple(
            path
            for path in sorted(self.image_directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"无标签 identity 目录没有可读图像：{self.image_directory}")
        if max_samples and max_samples < len(paths):
            selected = np.random.default_rng(seed).choice(len(paths), size=max_samples, replace=False)
            paths = tuple(paths[int(index)] for index in sorted(selected))
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch 不能为负数")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with Image.open(self.paths[index]) as opened:
            source_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
        crop = _random_crop(source_rgb, self.patch_size, rng)
        return {"image": torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0}


def _random_crop(image_rgb: np.ndarray, patch_size: int, rng: np.random.Generator) -> np.ndarray:
    """小于 patch 的图片先高质量放大；DIV2K 正常样本直接随机 HR 裁剪。"""

    height, width = image_rgb.shape[:2]
    if min(height, width) < patch_size:
        scale = patch_size / min(height, width)
        image_rgb = np.asarray(
            Image.fromarray(image_rgb).resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        height, width = image_rgb.shape[:2]
    y0 = int(rng.integers(0, height - patch_size + 1))
    x0 = int(rng.integers(0, width - patch_size + 1))
    return np.ascontiguousarray(image_rgb[y0 : y0 + patch_size, x0 : x0 + patch_size])
