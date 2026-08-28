"""DIV2K 独立 x2/wild-x4 超分数据与模型回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from training.superres.dataset import Div2kPairedSuperResolutionDataset
from training.superres.model import ConservativeSuperResolutionNet


def test_dataset_expands_wild_variants_and_preserves_scale(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_rgb(root / "superres/div2k/DIV2K_train_HR/0001.png", 128, 128)
    _write_rgb(root / "superres/div2k/DIV2K_valid_HR/0001.png", 128, 128)
    _write_rgb(root / "superres/div2k/DIV2K_train_LR_bicubic/X2/0001x2.png", 64, 64)
    _write_rgb(root / "superres/div2k/DIV2K_valid_LR_bicubic/X2/0001x2.png", 64, 64)
    for split in ("train", "valid"):
        _write_rgb(root / f"superres/div2k/wild_x4/DIV2K_{split}_LR_wild/0001x4w1.png", 32, 32)
    manifest = tmp_path / "div2k.jsonl"
    train = _record("train", "train")
    validation = _record("validation", "valid")
    manifest.write_text("\n".join(json.dumps(value) for value in (train, validation)) + "\n", encoding="utf-8")

    x2 = Div2kPairedSuperResolutionDataset(manifest, root, split="train", variant="x2", patch_size=64, seed=1)
    wild = Div2kPairedSuperResolutionDataset(manifest, root, split="train", variant="wild_x4", patch_size=64, seed=1)

    assert x2[0]["input"].shape == (3, 32, 32)
    assert x2[0]["target"].shape == (3, 64, 64)
    assert wild[0]["input"].shape == (3, 16, 16)
    assert wild[0]["target"].shape == (3, 64, 64)


def test_model_upsamples_by_declared_scale() -> None:
    model = ConservativeSuperResolutionNet(scale=4, channels=8, blocks=2)
    output = model(torch.zeros((2, 3, 16, 20)))

    assert output.shape == (2, 3, 64, 80)
    assert float(output.detach().min()) >= 0.0
    assert float(output.detach().max()) <= 1.0


def _record(split: str, wild_split: str) -> dict[str, object]:
    return {
        "sample_id": f"div2k:{split}",
        "split": split,
        "hr_image": f"superres/div2k/DIV2K_{'train' if split == 'train' else 'valid'}_HR/0001.png",
        "lr_x2_image": f"superres/div2k/DIV2K_{'train' if split == 'train' else 'valid'}_LR_bicubic/X2/0001x2.png",
        "wild_x4_images": [f"superres/div2k/wild_x4/DIV2K_{wild_split}_LR_wild/0001x4w1.png"],
    }


def _write_rgb(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
