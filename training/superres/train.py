"""训练 DIV2K 的独立 x2 或 wild-x4 保守超分模型。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.superres.train \
        --manifest "$SCREENRESTORE_DATA_ROOT/manifests/div2k.restoration.jsonl" \
        --data-root "$SCREENRESTORE_DATA_ROOT" --variant x2 \
        --output-directory "$SCREENRESTORE_RUN_ROOT/sr-x2" --epochs 20 --device auto

x2 bicubic 和 wild-x4 分别训练。wild-x4 的每个真实 LR 变体都保留为独立观测，绝不覆盖或
混入 x2 数据。模型只在 bicubic 基线上输出有界残差，结果属于保守超分而非生成式重绘。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from training.restoration.metrics import fidelity_metrics
from training.restoration.train import _accumulate, _device, _mean, _progress

from .dataset import Div2kPairedSuperResolutionDataset
from .model import ConservativeSuperResolutionNet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("x2", "wild_x4"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--max-delta", type=float, default=0.08)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size 和 learning-rate 必须为正数")
    _seed(args.seed)
    device = _device(args.device)
    train_data = Div2kPairedSuperResolutionDataset(
        args.manifest, args.data_root, split="train", variant=args.variant, patch_size=args.patch_size,
        seed=args.seed, max_samples=args.train_samples,
    )
    validation_data = Div2kPairedSuperResolutionDataset(
        args.manifest, args.data_root, split="validation", variant=args.variant, patch_size=args.patch_size,
        seed=args.seed + 1, max_samples=args.validation_samples,
    )
    validation_data.set_epoch(0)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    scale = 2 if args.variant == "x2" else 4
    model = ConservativeSuperResolutionNet(scale, args.channels, args.blocks, args.max_delta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "format_version": 1,
        "kind": "conservative_super_resolution_training",
        "variant": args.variant,
        "scale": scale,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "patch_size": args.patch_size,
        "channels": args.channels,
        "blocks": args.blocks,
        "max_delta": args.max_delta,
        "epochs": args.epochs,
        "device": str(device),
    }
    history: list[dict[str, float | int]] = []
    best = float("inf")
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        train_metrics = _run_epoch(model, train_loader, device, optimizer, epoch, args.epochs)
        validation_metrics = _validate(model, validation_loader, device, epoch, args.epochs)
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
        checkpoint = _checkpoint(model, args, epoch, validation_metrics)
        torch.save(checkpoint, output / "last.pt")
        if validation_metrics["loss"] < best:
            best = validation_metrics["loss"]
            torch.save(checkpoint, output / "best.pt")
    run.update({"wall_time_seconds": round(time.monotonic() - started, 3), "best_validation_loss": best})
    (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _run_epoch(model: ConservativeSuperResolutionNet, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device, optimizer: torch.optim.Optimizer, epoch: int, epochs: int) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    for index, batch in enumerate(loader, start=1):
        _progress(index - 1, len(loader), f"epoch {epoch}/{epochs} train")
        source, target = _batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        restored = model(source)
        loss, parts = _loss(restored, target, source)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _accumulate(totals, parts)
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} train")
    return _mean(totals, len(loader))


def _validate(model: ConservativeSuperResolutionNet, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device, epoch: int, epochs: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            _progress(index - 1, len(loader), f"epoch {epoch}/{epochs} validation")
            source, target = _batch(batch, device)
            restored = model(source)
            _loss_value, parts = _loss(restored, target, source)
            _accumulate(totals, parts)
            baseline = functional.interpolate(source, scale_factor=model.scale, mode="bicubic", align_corners=False)
            _accumulate(totals, fidelity_metrics(restored, target, baseline))
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} validation")
    return _mean(totals, len(loader))


def _loss(restored: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    baseline = functional.interpolate(source, size=target.shape[-2:], mode="bicubic", align_corners=False)
    reconstruction = torch.sqrt((restored - target).square() + 1e-6).mean()
    edge = torch.sqrt((_gradient(restored) - _gradient(target)).square() + 1e-6).mean()
    color = torch.sqrt((functional.avg_pool2d(restored, 9, 1, 4) - functional.avg_pool2d(baseline, 9, 1, 4)).square() + 1e-6).mean()
    loss = reconstruction + 0.15 * edge + 0.10 * color
    return loss, {"loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()), "edge": float(edge.detach()), "color_drift": float(color.detach())}


def _gradient(image: torch.Tensor) -> torch.Tensor:
    return functional.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 1, 0, 0)) + functional.pad(image[:, :, 1:, :] - image[:, :, :-1, :], (0, 0, 0, 1))


def _batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["input"].to(device), batch["target"].to(device)


def _checkpoint(model: ConservativeSuperResolutionNet, args: argparse.Namespace, epoch: int, metrics: dict[str, float]) -> dict[str, object]:
    return {"format_version": 1, "model": "ConservativeSuperResolutionNet", "epoch": epoch, "variant": args.variant, "scale": model.scale, "channels": model.channels, "blocks": model.blocks, "max_delta": model.max_delta, "patch_size": args.patch_size, "metrics": metrics, "state_dict": model.state_dict()}


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
