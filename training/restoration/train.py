"""训练紧凑的有界残差 Fidelity 恢复模型。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.restoration.train \\
        --train-hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR" \\
        --validation-hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \\
        --output-directory "$SCREENRESTORE_RUN_ROOT/restoration/fidelity-smoke" \\
        --train-samples 300 --validation-samples 50 --epochs 2 --device auto

每一个 batch 从 clean HR 随机裁剪，并即时模拟有限分辨率、光学模糊、噪声、JPEG 与光度
偏差；不生成退化缓存。checkpoint、历史和可复核的 run 配置只会写入 output-directory。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import Div2kHrDataset, UnlabeledIdentityDataset
from .degradation import CameraDegradationConfig
from .losses import fidelity_loss, identity_loss
from .metrics import fidelity_metrics
from .model import BoundedResidualNet, FidelityNetV2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-hr-directory", type=Path, required=True)
    parser.add_argument("--validation-hr-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--max-delta", type=float, default=0.06)
    parser.add_argument(
        "--architecture",
        choices=("bounded_residual_v1", "fidelity_v2"),
        default="bounded_residual_v1",
    )
    parser.add_argument(
        "--preserve-photometric-nuisance",
        action="store_true",
        help="P3：input/target 共享摄影 nuisance，Fidelity 只学习低层退化",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--identity-weight", type=float, default=0.35)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument(
        "--private-identity-directory",
        type=Path,
        help="显式指定无 GT 私有图片目录；仅加入 identity 保护，不作为重建真值",
    )
    parser.add_argument(
        "--private-identity-weight",
        type=float,
        default=0.20,
        help="无 GT identity 损失权重；仅在指定 private-identity-directory 时生效",
    )
    parser.add_argument(
        "--private-identity-samples",
        type=int,
        default=0,
        help="无 GT identity 样本上限，0 表示使用显式目录中的全部图片",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    _validate_args(args)
    _seed_everything(args.seed)
    device = _device(args.device)
    degradation = CameraDegradationConfig(apply_photometric=not args.preserve_photometric_nuisance)
    train_data = Div2kHrDataset(
        args.train_hr_directory,
        patch_size=args.patch_size,
        degradation=degradation,
        seed=args.seed,
        max_samples=args.train_samples,
        preserve_photometric_nuisance=args.preserve_photometric_nuisance,
    )
    validation_data = Div2kHrDataset(
        args.validation_hr_directory,
        patch_size=args.patch_size,
        degradation=degradation,
        seed=args.seed + 1,
        max_samples=args.validation_samples,
        preserve_photometric_nuisance=args.preserve_photometric_nuisance,
    )
    validation_data.set_epoch(0)
    private_identity_data = (
        UnlabeledIdentityDataset(
            args.private_identity_directory,
            patch_size=args.patch_size,
            seed=args.seed + 2,
            max_samples=args.private_identity_samples,
        )
        if args.private_identity_directory is not None
        else None
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    private_identity_loader = (
        DataLoader(
            private_identity_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if private_identity_data is not None
        else None
    )
    model: torch.nn.Module
    if args.architecture == "fidelity_v2":
        model = FidelityNetV2(args.channels, max_delta=args.max_delta).to(device)
    else:
        model = BoundedResidualNet(args.channels, args.blocks, args.max_delta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    run = {
        "format_version": 2,
        "kind": "fidelity_restoration_training",
        "train_hr_directory": str(args.train_hr_directory.expanduser().resolve()),
        "validation_hr_directory": str(args.validation_hr_directory.expanduser().resolve()),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "architecture": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "channels": args.channels,
        "blocks": args.blocks,
        "max_delta": args.max_delta,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "identity_weight": args.identity_weight,
        "edge_weight": args.edge_weight,
        "private_identity": {
            "enabled": private_identity_data is not None,
            "samples": len(private_identity_data) if private_identity_data is not None else 0,
            "weight": args.private_identity_weight if private_identity_data is not None else 0.0,
        },
        "device": str(device),
        "degradation": asdict(degradation),
        "preserve_photometric_nuisance": args.preserve_photometric_nuisance,
    }
    (output_directory / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        if private_identity_data is not None:
            private_identity_data.set_epoch(epoch)
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            args.identity_weight,
            args.edge_weight,
            private_identity_loader,
            args.private_identity_weight,
        )
        validation_metrics = _validate_epoch(
            model,
            validation_loader,
            device,
            epoch,
            args.epochs,
            args.identity_weight,
            args.edge_weight,
        )
        scheduler.step()
        record: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
        checkpoint = _checkpoint(model, args, degradation, epoch, validation_metrics)
        torch.save(checkpoint, output_directory / "last.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(checkpoint, output_directory / "best.pt")
    run["wall_time_seconds"] = round(time.monotonic() - started, 3)
    run["best_validation_loss"] = best_validation
    (output_directory / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_directory / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    identity_weight: float,
    edge_weight: float,
    private_identity_loader: DataLoader[dict[str, torch.Tensor]] | None,
    private_identity_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    private_iterator = iter(private_identity_loader) if private_identity_loader is not None else None
    for batch_index, batch in enumerate(loader, start=1):
        _progress(batch_index - 1, len(loader), f"epoch {epoch}/{epochs} train")
        degraded, target = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, FidelityNetV2):
            restored, _alpha, _budget, artifact_logits, artifact_severity = model.forward_training(
                degraded
            )
        else:
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
        if isinstance(model, FidelityNetV2):
            artifact_loss = F.binary_cross_entropy_with_logits(
                artifact_logits,
                batch["artifact_labels"].to(device),
            )
            severity_loss = F.smooth_l1_loss(
                artifact_severity,
                batch["artifact_severity"].to(device),
            )
            loss = loss + 0.08 * artifact_loss + 0.04 * severity_loss
            parts["artifact_aux"] = float(artifact_loss.detach())
            parts["severity_aux"] = float(severity_loss.detach())
        if private_iterator is not None:
            try:
                private_batch = next(private_iterator)
            except StopIteration:
                private_iterator = iter(private_identity_loader)
                private_batch = next(private_iterator)
            private_image = private_batch["image"].to(device)
            private_identity = identity_loss(model(private_image), private_image)
            loss = loss + private_identity_weight * private_identity
            parts["private_identity"] = float(private_identity.detach())
        parts["loss"] = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _accumulate(totals, parts)
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} train")
    return _mean(totals, len(loader))


def _validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    epoch: int,
    epochs: int,
    identity_weight: float,
    edge_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            _progress(batch_index - 1, len(loader), f"epoch {epoch}/{epochs} validation")
            degraded, target = _batch_to_device(batch, device)
            if isinstance(model, FidelityNetV2):
                restored, _alpha, _budget, artifact_logits, artifact_severity = model.forward_training(
                    degraded
                )
            else:
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
            if isinstance(model, FidelityNetV2):
                artifact_loss = F.binary_cross_entropy_with_logits(
                    artifact_logits,
                    batch["artifact_labels"].to(device),
                )
                severity_loss = F.smooth_l1_loss(
                    artifact_severity,
                    batch["artifact_severity"].to(device),
                )
                loss = loss + 0.08 * artifact_loss + 0.04 * severity_loss
                parts["artifact_aux"] = float(artifact_loss)
                parts["severity_aux"] = float(severity_loss)
            parts["loss"] = float(loss)
            _accumulate(totals, parts)
            _accumulate(totals, fidelity_metrics(restored, target, clean_restored))
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} validation")
    return _mean(totals, len(loader))


def _batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["input"].to(device), batch["target"].to(device)


def _accumulate(totals: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + value


def _mean(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(1, count) for key, value in totals.items()}


def _checkpoint(
    model: torch.nn.Module,
    args: argparse.Namespace,
    degradation: CameraDegradationConfig,
    epoch: int,
    validation_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "format_version": 2,
        "model": type(model).__name__,
        "architecture": args.architecture,
        "channels": args.channels,
        "blocks": args.blocks,
        "max_delta": args.max_delta,
        "patch_size": args.patch_size,
        "identity_weight": args.identity_weight,
        "edge_weight": args.edge_weight,
        "preserve_photometric_nuisance": args.preserve_photometric_nuisance,
        "degradation": asdict(degradation),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须大于 0")
    if args.train_samples < 0 or args.validation_samples < 0:
        raise ValueError("样本数不能为负数")
    if args.workers < 0:
        raise ValueError("workers 不能为负数")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate 必须大于 0")
    if not 0.0 <= args.private_identity_weight <= 2.0:
        raise ValueError("private-identity-weight 必须位于 0..2")
    if args.private_identity_samples < 0:
        raise ValueError("private-identity-samples 不能为负数")
    if args.architecture == "fidelity_v2" and (args.channels < 24 or args.channels % 8):
        raise ValueError("fidelity_v2 channels 必须是不小于 24 的 8 倍数")


def _device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 不支持 CUDA")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前 PyTorch 不支持 MPS")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>4}/{total:<4} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
