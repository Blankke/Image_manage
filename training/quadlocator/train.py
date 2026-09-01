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
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from screenrestore.geometry.decoder import CornerDecoderSpec
from training.quadlocator.dataset import QuadDataset, SourceGroupBalancedSampler
from training.quadlocator.losses import quadlocator_loss
from training.quadlocator.metrics import ValidationMetrics
from training.quadlocator.model import QuadLocatorS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="manifest 中 image 的相对根目录；标准外部数据清单应显式提供",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--loss-profile",
        choices=("p2", "boundary", "tail", "full"),
        default="p2",
        help="P3 消融：B1=boundary，B3=tail，B5=full",
    )
    parser.add_argument(
        "--hard-sampling",
        action="store_true",
        help="按 difficulty/hard_taxonomy 加权；仅 B3/B5 启用",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="从旧 QuadLocator checkpoint 加载名称和 shape 均兼容的参数",
    )
    parser.add_argument("--train-samples", type=int, default=0, help="训练样本上限，0 表示全部")
    parser.add_argument("--validation-samples", type=int, default=0, help="验证样本上限，0 表示全部")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="selection_score 连续多少轮未改善后停止；0 表示关闭",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须大于 0")
    if args.train_samples < 0 or args.validation_samples < 0:
        raise ValueError("样本上限不能为负数")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience 不能为负数")
    _seed_everything(args.seed)
    device = _device(args.device)
    train_data = QuadDataset(
        args.manifest,
        split="train",
        image_size=args.image_size,
        dataset_root=args.dataset_root,
        max_samples=args.train_samples,
        augment=True,
        seed=args.seed,
    )
    validation_data = QuadDataset(
        args.manifest,
        split="validation",
        image_size=args.image_size,
        dataset_root=args.dataset_root,
        max_samples=args.validation_samples,
        augment=False,
        seed=args.seed,
    )
    sampler = SourceGroupBalancedSampler(
        train_data,
        seed=args.seed,
        difficulty_weighting=args.hard_sampling,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = QuadLocatorS(args.width_multiplier).to(device)
    warm_start = None
    if args.init_checkpoint is not None:
        warm_start = _load_compatible_checkpoint(model, args.init_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    # 每次训练将可比较的实验元数据独立落盘；不记录图片内容，也不把运行产物放入仓库。
    run_metadata = {
        "format_version": 3,
        "dataset_manifest": str(args.manifest.expanduser().resolve()),
        "dataset_root": str(train_data.root),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "train_distribution": train_data.statistics(),
        "validation_distribution": validation_data.statistics(),
        "architecture": "QuadLocatorS",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "init_checkpoint": str(args.init_checkpoint.expanduser().resolve())
        if args.init_checkpoint is not None
        else None,
        "warm_start": warm_start,
        "early_stopping_patience": args.early_stopping_patience,
        "decoder": CornerDecoderSpec().to_dict(),
        "seed": args.seed,
        "loss_profile": args.loss_profile,
        "hard_sampling": args.hard_sampling,
    }
    _warn_missing_training_domains(run_metadata["train_distribution"])
    (output_directory / "run.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    history: list[dict[str, float | int | str]] = []
    best_validation = float("inf")
    best_selection_score = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        sampler.set_epoch(epoch)
        train_loss, _ = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            epoch,
            args.epochs,
            loss_profile=args.loss_profile,
        )
        validation_loss, validation_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            None,
            epoch,
            args.epochs,
            loss_profile=args.loss_profile,
            collect_validation_metrics=True,
        )
        scheduler.step()
        record: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "validation_loss": round(validation_loss, 8),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "device": str(device),
            "validation_metrics": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
        checkpoint = {
            "format_version": 3,
            "model": "QuadLocatorS",
            "width_multiplier": args.width_multiplier,
            "image_size": args.image_size,
            "class_order": ["artwork", "postcard", "screen", "none"],
            "parameter_count": run_metadata["parameter_count"],
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "decoder": CornerDecoderSpec().to_dict(),
            "seed": args.seed,
            "loss_profile": args.loss_profile,
            "hard_sampling": args.hard_sampling,
        }
        torch.save(checkpoint, output_directory / "last.pt")
        selection_score = float(validation_metrics["selection_score"])
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_validation = validation_loss
            torch.save(checkpoint, output_directory / "best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping：selection_score 连续 {epochs_without_improvement} 轮未改善",
                file=sys.stderr,
            )
            break
    (output_directory / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_metadata["wall_time_seconds"] = round(time.monotonic() - started_at, 4)
    run_metadata["best_validation_loss"] = best_validation
    run_metadata["best_selection_score"] = best_selection_score
    run_metadata["completed_epochs"] = len(history)
    (output_directory / "run.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
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
    *,
    loss_profile: str = "full",
    collect_validation_metrics: bool = False,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    validation_metrics = ValidationMetrics() if collect_validation_metrics else None
    for batch_index, batch in enumerate(loader, start=1):
        _progress(batch_index - 1, len(loader), f"epoch {epoch}/{epochs} {'train' if training else 'val'}")
        values = {key: tensor.to(device) for key, tensor in batch.items()}
        with torch.set_grad_enabled(training):
            outputs = model(values["image"])
            loss, _metrics = quadlocator_loss(outputs, values, profile=loss_profile)
            if validation_metrics is not None:
                validation_metrics.update(outputs, values)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += float(loss.detach())
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} {'train' if training else 'val'}")
    return (
        total_loss / max(1, len(loader)),
        validation_metrics.compute() if validation_metrics is not None else {},
    )


def _load_compatible_checkpoint(model: QuadLocatorS, checkpoint_path: Path) -> dict[str, object]:
    """兼容加载 P1 参数，新 head 和 shape 不一致参数保持当前初始化。"""

    path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("init-checkpoint 缺少 state_dict")
    current = model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current and getattr(tensor, "shape", None) == current[name].shape
    }
    skipped = sorted(name for name in state_dict if name not in compatible)
    missing = sorted(name for name in current if name not in compatible)
    parameter_names = set(dict(model.named_parameters()))
    loaded_parameter_count = sum(
        int(tensor.numel()) for name, tensor in compatible.items() if name in parameter_names
    )
    loaded_buffer_count = sum(
        int(tensor.numel()) for name, tensor in compatible.items() if name not in parameter_names
    )
    model.load_state_dict(compatible, strict=False)
    print(
        f"warm-start loaded {len(compatible)}/{len(current)} tensors "
        f"({loaded_parameter_count} parameters, {loaded_buffer_count} buffer values): "
        + ", ".join(sorted(compatible)),
        file=sys.stderr,
    )
    print("warm-start skipped parameters: " + (", ".join(skipped) or "(none)"), file=sys.stderr)
    print("warm-start missing new parameters: " + (", ".join(missing) or "(none)"), file=sys.stderr)
    return {
        "source": str(path),
        "loaded_tensor_count": len(compatible),
        "loaded_parameter_count": loaded_parameter_count,
        "loaded_buffer_value_count": loaded_buffer_count,
        "loaded_parameter_names": sorted(compatible),
        "skipped_parameter_names": skipped,
        "missing_new_parameters": missing,
    }


def _warn_missing_training_domains(statistics: object) -> None:
    if not isinstance(statistics, dict):
        return
    classes = statistics.get("class_distribution", {})
    outer = statistics.get("outer_presence_distribution", {})
    if not isinstance(classes, dict) or not isinstance(outer, dict):
        return
    missing = [name for name in ("artwork", "screen", "none") if int(classes.get(name, 0)) == 0]
    if int(outer.get("absent", 0)) == 0:
        missing.append("outer_present=0")
    if missing:
        print(
            "WARNING: 训练 split 缺少关键分布：" + ", ".join(missing),
            file=sys.stderr,
        )


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
