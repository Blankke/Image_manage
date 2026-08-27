"""按固定退化切片评估 Fidelity checkpoint，为调参提供可比较证据。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.restoration.evaluate_slices \
        --checkpoint "$SCREENRESTORE_RUN_ROOT/p1/restoration/best.pt" \
        --hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
        --output "$SCREENRESTORE_RUN_ROOT/p1/restoration/evaluation-slices.json" \
        --samples 100 --device auto

每个切片使用固定种子、固定验证图片与明确的在线退化配置。它只评价同尺寸、观测支持的
Fidelity 模型；反光、去摩尔纹、光度参数预测和超分必须使用各自的配对基准，不能借本报告
宣称能力。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import Div2kHrDataset
from .degradation import CameraDegradationConfig
from .losses import fidelity_loss
from .metrics import fidelity_metrics
from .model import BoundedResidualNet
from .train import _accumulate, _batch_to_device, _device, _mean, _progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hr-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.samples < 1:
        raise ValueError("batch-size 与 samples 必须大于 0")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "BoundedResidualNet":
        raise ValueError("checkpoint 不是 BoundedResidualNet")
    model = BoundedResidualNet(
        int(checkpoint["channels"]),
        int(checkpoint["blocks"]),
        float(checkpoint["max_delta"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = _device(args.device)
    model.to(device).eval()
    slices: dict[str, dict[str, object]] = {}
    configs = evaluation_slices()
    for index, (name, degradation) in enumerate(configs.items(), start=1):
        _progress(index - 1, len(configs), f"切片 {name}")
        metrics = evaluate_slice(
            model,
            args.hr_directory,
            degradation,
            samples=args.samples,
            batch_size=args.batch_size,
            patch_size=int(checkpoint["patch_size"]),
            seed=args.seed + index * 1009,
            device=device,
            identity_weight=float(checkpoint["identity_weight"]),
            edge_weight=float(checkpoint["edge_weight"]),
        )
        slices[name] = {"degradation": asdict(degradation), "metrics": metrics}
    _progress(len(configs), len(configs), "切片评估完成")
    result = {
        "format_version": 1,
        "kind": "fidelity_restoration_slice_evaluation",
        "checkpoint": str(checkpoint_path),
        "hr_directory": str(args.hr_directory.expanduser().resolve()),
        "samples_per_slice": args.samples,
        "seed": args.seed,
        "device": str(device),
        "slices": slices,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def evaluation_slices() -> dict[str, CameraDegradationConfig]:
    """返回固定切片，避免用同一混合退化总分掩盖具体能力退化。"""

    isolated = {
        "min_resize_scale": 1.0,
        "max_resize_scale": 1.0,
        "defocus_probability": 0.0,
        "motion_probability": 0.0,
        "jpeg_probability": 0.0,
        "ringing_probability": 0.0,
        "clean_probability": 0.0,
        "max_defocus_sigma": 2.2,
        "max_motion_length": 13,
        "max_noise_std": 0.0,
        "min_exposure_stops": 0.0,
        "max_exposure_stops": 0.0,
        "max_white_balance_shift": 0.0,
        "max_illumination_gradient": 0.0,
        "min_jpeg_quality": 52,
        "max_jpeg_quality": 94,
    }
    return {
        "clean_identity": CameraDegradationConfig(**{**isolated, "clean_probability": 1.0}),
        "noise_light": CameraDegradationConfig(**{**isolated, "max_noise_std": 0.008}),
        "noise_heavy": CameraDegradationConfig(**{**isolated, "max_noise_std": 0.03}),
        "defocus": CameraDegradationConfig(
            **{**isolated, "defocus_probability": 1.0, "max_defocus_sigma": 1.6}
        ),
        "motion": CameraDegradationConfig(
            **{**isolated, "motion_probability": 1.0, "max_motion_length": 11}
        ),
        "jpeg": CameraDegradationConfig(
            **{**isolated, "jpeg_probability": 1.0, "min_jpeg_quality": 45, "max_jpeg_quality": 65}
        ),
        "exposure": CameraDegradationConfig(
            **{**isolated, "min_exposure_stops": -0.8, "max_exposure_stops": 0.5}
        ),
        "white_balance": CameraDegradationConfig(
            **{**isolated, "max_white_balance_shift": 0.22}
        ),
        "illumination": CameraDegradationConfig(
            **{**isolated, "max_illumination_gradient": 0.30}
        ),
        "compound_camera": CameraDegradationConfig(),
    }


def evaluate_slice(
    model: BoundedResidualNet,
    hr_directory: Path,
    degradation: CameraDegradationConfig,
    *,
    samples: int,
    batch_size: int,
    patch_size: int,
    seed: int,
    device: torch.device,
    identity_weight: float,
    edge_weight: float,
) -> dict[str, float]:
    """使用固定 epoch 与 seed 生成可复现的单个退化切片。"""

    dataset = Div2kHrDataset(
        hr_directory,
        patch_size=patch_size,
        degradation=degradation,
        seed=seed,
        max_samples=samples,
    )
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            degraded, target = _batch_to_device(batch, device)
            restored = model(degraded)
            clean_restored = model(target)
            loss, parts = fidelity_loss(
                restored,
                target,
                clean_restored,
                target,
                identity_weight=identity_weight,
                edge_weight=edge_weight,
            )
            parts["loss"] = float(loss)
            _accumulate(totals, parts)
            _accumulate(totals, fidelity_metrics(restored, target, clean_restored))
    return _mean(totals, len(loader))


if __name__ == "__main__":
    raise SystemExit(main())
