"""读取 DIV2K 清单中的真实 LR/HR 配对，不生成离线副本。

``x2`` 与 ``wild_x4`` 的成像链和放大倍率不同，调用方必须明确选择一种 variant，
从数据层阻止它们在同一训练任务里混合。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Div2kPairedSuperResolutionDataset(Dataset[dict[str, torch.Tensor]]):
    """按 manifest 读取 DIV2K LR/HR，并在严格对应坐标做随机裁剪。"""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        *,
        split: str,
        variant: str,
        patch_size: int,
        seed: int,
        max_samples: int = 0,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("超分训练 split 只能是 train 或 validation")
        if variant not in {"x2", "wild_x4"}:
            raise ValueError("超分 variant 只能是 x2 或 wild_x4")
        self.scale = 2 if variant == "x2" else 4
        if patch_size < 64 or patch_size % self.scale:
            raise ValueError("patch-size 至少为 64 且必须被超分倍率整除")
        if max_samples < 0:
            raise ValueError("max-samples 不能为负数")
        self.data_root = data_root.expanduser().resolve()
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0
        records = _records(manifest, self.data_root, split, variant)
        if max_samples:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(records), size=min(max_samples, len(records)), replace=False))
            records = [records[int(index)] for index in indices]
        if not records:
            raise ValueError(f"DIV2K {variant} {split} 没有可训练配对")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        lr_path, hr_path = self.records[index]
        lr = _read_rgb(lr_path)
        hr = _read_rgb(hr_path)
        expected = (lr.shape[0] * self.scale, lr.shape[1] * self.scale)
        if hr.shape[:2] != expected:
            raise ValueError("DIV2K LR/HR 尺寸与声明超分倍率不一致")
        lr_patch = self.patch_size // self.scale
        if lr.shape[0] < lr_patch or lr.shape[1] < lr_patch:
            raise ValueError("DIV2K 图像小于所需 LR 裁剪尺寸")
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        top = int(rng.integers(0, lr.shape[0] - lr_patch + 1))
        left = int(rng.integers(0, lr.shape[1] - lr_patch + 1))
        lr_crop = lr[top : top + lr_patch, left : left + lr_patch]
        hr_crop = hr[
            top * self.scale : (top + lr_patch) * self.scale,
            left * self.scale : (left + lr_patch) * self.scale,
        ]
        return {
            "input": torch.from_numpy(lr_crop.copy()).permute(2, 0, 1),
            "target": torch.from_numpy(hr_crop.copy()).permute(2, 0, 1),
        }


def _records(manifest: Path, data_root: Path, split: str, variant: str) -> list[tuple[Path, Path]]:
    path = manifest.expanduser().resolve()
    if not path.is_file():
        raise ValueError("DIV2K manifest 不存在")
    records: list[tuple[Path, Path]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("split") != split:
            continue
        hr = _resolve(data_root, value.get("hr_image"), line_number)
        if variant == "x2":
            lr = _resolve(data_root, value.get("lr_x2_image"), line_number)
            records.append((lr, hr))
        else:
            wild = value.get("wild_x4_images")
            if not isinstance(wild, list):
                raise ValueError(f"第 {line_number} 行 wild_x4_images 必须是列表")
            records.extend((_resolve(data_root, item, line_number), hr) for item in wild)
    return records


def _resolve(root: Path, value: object, line_number: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"第 {line_number} 行缺少所需 LR/HR 图像路径")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "private" in relative.parts:
        raise ValueError(f"第 {line_number} 行存在不安全图像路径")
    result = (root / relative).resolve()
    if not result.is_file() or root not in result.parents:
        raise ValueError(f"第 {line_number} 行引用的图像不存在或越出 data-root")
    return result


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("DIV2K 图像无法解码")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
