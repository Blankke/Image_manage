"""DIV2K HR 在线 Fidelity 训练链路的数值与烟测。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from training.restoration.dataset import Div2kHrDataset, UnlabeledIdentityDataset
from training.restoration.degradation import CameraDegradationConfig, degrade_camera_image
from training.restoration.evaluate import main as evaluate_main
from training.restoration.losses import fidelity_loss
from training.restoration.metrics import fidelity_metrics
from training.restoration.model import BoundedResidualNet
from training.restoration.train import main as train_main


def test_camera_degradation_is_deterministic_bounded_and_read_only() -> None:
    clean = _gradient_image(96, 72)
    source_copy = clean.copy()
    config = CameraDegradationConfig(
        min_resize_scale=0.6,
        max_resize_scale=0.6,
        defocus_probability=1.0,
        motion_probability=1.0,
        jpeg_probability=1.0,
        ringing_probability=1.0,
        clean_probability=0.0,
    )
    first = degrade_camera_image(clean, np.random.default_rng(7), config)
    second = degrade_camera_image(clean, np.random.default_rng(7), config)

    assert first.degraded_rgb.dtype == np.float32
    assert first.degraded_rgb.shape == clean.shape
    assert 0.0 <= float(first.degraded_rgb.min()) <= float(first.degraded_rgb.max()) <= 1.0
    assert np.array_equal(first.degraded_rgb, second.degraded_rgb)
    assert np.array_equal(clean, source_copy)
    assert first.parameters["identity"] is False
    assert first.parameters["motion_length"] >= 3
    assert first.parameters["jpeg_quality"] < 100


def test_clean_degradation_branch_keeps_pixels_exact() -> None:
    clean = _gradient_image(48, 40).astype(np.float32) / 255.0
    result = degrade_camera_image(
        clean,
        np.random.default_rng(8),
        CameraDegradationConfig(clean_probability=1.0),
    )
    assert result.parameters["identity"] is True
    assert np.array_equal(result.degraded_rgb, clean)
    assert not np.shares_memory(result.degraded_rgb, clean)


def test_dataset_uses_deterministic_online_crops_without_cached_images(tmp_path: Path) -> None:
    _write_hr_images(tmp_path, count=2, size=96)
    dataset = Div2kHrDataset(
        tmp_path,
        patch_size=64,
        degradation=CameraDegradationConfig(clean_probability=1.0),
        seed=3,
    )
    first = dataset[0]
    repeated = dataset[0]
    dataset.set_epoch(1)
    next_epoch = dataset[0]

    assert len(dataset) == 2
    assert first["input"].shape == (3, 64, 64)
    assert torch.equal(first["input"], first["target"])
    assert torch.equal(first["input"], repeated["input"])
    assert not torch.equal(first["target"], next_epoch["target"])


def test_unlabeled_images_only_supply_identity_observations(tmp_path: Path) -> None:
    _write_hr_images(tmp_path, count=2, size=96)
    dataset = UnlabeledIdentityDataset(tmp_path, patch_size=64, seed=9)
    first = dataset[0]
    repeated = dataset[0]
    dataset.set_epoch(1)

    assert len(dataset) == 2
    assert set(first) == {"image"}
    assert first["image"].shape == (3, 64, 64)
    assert torch.equal(first["image"], repeated["image"])
    assert not torch.equal(first["image"], dataset[0]["image"])


def test_bounded_network_and_metrics_enforce_fidelity_contract() -> None:
    torch.manual_seed(11)
    image = torch.rand((2, 3, 32, 40), dtype=torch.float32)
    model = BoundedResidualNet(channels=8, blocks=2, max_delta=0.04)
    restored = model(image)
    loss, parts = fidelity_loss(restored, image, model(image), image)
    metrics = fidelity_metrics(image, image, image)

    assert float(torch.max(torch.abs(restored - image)).detach()) <= 0.04 + 1e-6
    assert float(loss.detach()) >= 0.0
    assert parts["identity"] >= 0.0
    assert metrics["psnr"] > 100.0
    assert metrics["ssim"] > 0.9999
    assert metrics["identity_mae"] == 0.0
    assert metrics["edge_correlation"] > 0.9999
    assert metrics["color_error_255"] == 0.0


def test_train_then_evaluate_small_hr_directory(tmp_path: Path) -> None:
    train_directory = tmp_path / "train"
    validation_directory = tmp_path / "validation"
    _write_hr_images(train_directory, count=1, size=72)
    _write_hr_images(validation_directory, count=1, size=72)
    run_directory = tmp_path / "run"
    assert (
        train_main(
            [
                "--train-hr-directory",
                str(train_directory),
                "--validation-hr-directory",
                str(validation_directory),
                "--output-directory",
                str(run_directory),
                "--train-samples",
                "1",
                "--validation-samples",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--patch-size",
                "64",
                "--channels",
                "8",
                "--blocks",
                "2",
                "--device",
                "cpu",
                "--private-identity-directory",
                str(train_directory),
                "--private-identity-samples",
                "1",
            ]
        )
        == 0
    )
    result_path = run_directory / "evaluation.json"
    assert (
        evaluate_main(
            [
                "--checkpoint",
                str(run_directory / "best.pt"),
                "--hr-directory",
                str(validation_directory),
                "--output",
                str(result_path),
                "--samples",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads(result_path.read_text(encoding="utf-8"))
    assert (run_directory / "best.pt").is_file()
    assert run["architecture"] == "BoundedResidualNet"
    assert run["private_identity"] == {"enabled": True, "samples": 1, "weight": 0.2}
    assert evaluation["metrics"]["psnr"] > 0.0
    assert set(evaluation["metrics"]) >= {
        "psnr",
        "ssim",
        "identity_mae",
        "edge_correlation",
        "color_error_255",
    }


def _gradient_image(width: int, height: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.clip(
        np.stack((xx * 255 / max(1, width - 1), yy * 255 / max(1, height - 1), (xx + yy) * 127 / max(1, width + height - 2)), axis=2),
        0,
        255,
    ).astype(np.uint8)


def _write_hr_images(directory: Path, *, count: int, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = _gradient_image(size, size)
        image = np.roll(image, index * 5, axis=1)
        Image.fromarray(image, "RGB").save(directory / f"{index + 1:04d}.png")
