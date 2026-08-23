"""训练 QuadLocator-S 多任务模型。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.quadlocator.train --manifest /data/geometry/manifest.jsonl \
        --output-directory /data/geometry/runs/quadlocator-s

可通过 ``--device auto`` 在 CUDA、MPS 与 CPU 间自动选择。每个 epoch 和 batch 都会
显示文本进度条；checkpoint 只写入显式输出目录，不进入核心运行时包。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.quadlocator.dataset import QuadDataset
from training.quadlocator.losses import quadlocator_loss
from training.quadlocator.model import QuadLocatorS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须大于 0")
    _seed_everything(args.seed)
    device = _device(args.device)
    train_data = QuadDataset(args.manifest, split="train", image_size=args.image_size)
    validation_data = QuadDataset(args.manifest, split="validation", image_size=args.image_size)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = QuadLocatorS(args.width_multiplier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | str]] = []
    best_validation = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer, epoch, args.epochs)
        validation_loss = _run_epoch(model, validation_loader, device, None, epoch, args.epochs)
        scheduler.step()
        record: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "validation_loss": round(validation_loss, 8),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "device": str(device),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
        checkpoint = {
            "format_version": 1,
            "model": "QuadLocatorS",
            "width_multiplier": args.width_multiplier,
            "image_size": args.image_size,
            "class_order": ["artwork", "postcard", "screen", "none"],
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
        }
        torch.save(checkpoint, output_directory / "last.pt")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, output_directory / "best.pt")
    (output_directory / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def _run_epoch(
    model: QuadLocatorS,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    epochs: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        _progress(batch_index - 1, len(loader), f"epoch {epoch}/{epochs} {'train' if training else 'val'}")
        values = {key: tensor.to(device) for key, tensor in batch.items()}
        with torch.set_grad_enabled(training):
            outputs = model(values["image"])
            loss, _metrics = quadlocator_loss(outputs, values)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += float(loss.detach())
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} {'train' if training else 'val'}")
    return total_loss / max(1, len(loader))


def _device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 不支持 CUDA")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前 PyTorch 不支持 MPS")
        return device
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
