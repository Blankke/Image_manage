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
from training.quadlocator.dataset import AUGMENTATION_MODES, QuadDataset, SourceGroupBalancedSampler
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
        choices=(
            "content_only",
            "content_heatmap_only",
            "content_coordinate_only",
            "content_mask",
            "content_boundary",
            "p2",
            "boundary",
            "tail",
            "full",
        ),
        default="p2",
        help=(
            "P4 G1=content_only、G2=content_mask、G3=content_boundary；"
            "G3.5 提供 content_heatmap_only/content_coordinate_only 语义诊断；"
            "P3 保留 p2/boundary/tail/full"
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=(
            "all",
            "content_head",
            "content_backbone",
            "content_mask_backbone",
            "content_boundary_backbone",
        ),
        default="all",
        help=(
            "G1 任务隔离：content_head 只训 content head；"
            "content_backbone 训 backbone/FPN + content head；"
            "content_mask_backbone 供 G2 训练 backbone/FPN + content/mask heads；"
            "content_boundary_backbone 供 G3 训练 backbone/FPN + content/boundary heads"
        ),
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
    parser.add_argument(
        "--validation-samples", type=int, default=0, help="验证样本上限，0 表示全部"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--train-augmentation",
        choices=tuple(sorted(AUGMENTATION_MODES)),
        default="full",
        help="训练增强模式；G3.5 以 none/photometric/geometric/full 分离根因",
    )
    parser.add_argument(
        "--validation-split",
        choices=("train", "validation"),
        default="validation",
        help="验证 split；overfit 诊断可显式使用 train，常规训练必须保持默认 validation",
    )
    parser.add_argument(
        "--evaluate-init",
        action="store_true",
        help="训练前在同一 validation subset 冻结 warm-start 指标，供短程消融直接比较",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="selection_score 连续多少轮未改善后停止；0 表示关闭",
    )
    parser.add_argument(
        "--early-stopping-criterion",
        choices=("product", "geometry"),
        default="product",
        help="early stop 依据；P4 geometry recovery 应使用 geometry",
    )
    parser.add_argument(
        "--geometry-collapse-patience",
        type=int,
        default=0,
        help="连续触发 geometry tail collapse 后停止；0 表示关闭",
    )
    parser.add_argument(
        "--geometry-collapse-nce-p95-ratio",
        type=float,
        default=0.0,
        help="相对 epoch 0 的 NCE P95 上限倍率；watchdog 启用时必须大于 1",
    )
    parser.add_argument(
        "--geometry-collapse-iou-p05-ratio",
        type=float,
        default=0.0,
        help="相对 epoch 0 的 IoU P05 下限倍率；watchdog 启用时必须位于 (0,1)",
    )
    parser.add_argument(
        "--best-geometry-nce-p95-ratio",
        type=float,
        default=0.0,
        help="best_geometry 相对 epoch 0 的 NCE P95 上限倍率；0 表示关闭 eligibility gate",
    )
    parser.add_argument(
        "--best-geometry-iou-p05-ratio",
        type=float,
        default=0.0,
        help="best_geometry 相对 epoch 0 的 IoU P05 下限倍率；0 表示关闭 eligibility gate",
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
    if args.geometry_collapse_patience < 0:
        raise ValueError("geometry-collapse-patience 不能为负数")
    if args.geometry_collapse_patience:
        if not args.evaluate_init:
            raise ValueError("geometry collapse watchdog 必须同时设置 --evaluate-init")
        if args.geometry_collapse_nce_p95_ratio <= 1.0:
            raise ValueError("geometry-collapse-nce-p95-ratio 必须大于 1")
        if not 0.0 < args.geometry_collapse_iou_p05_ratio < 1.0:
            raise ValueError("geometry-collapse-iou-p05-ratio 必须位于 (0,1)")
    eligibility_values = (
        args.best_geometry_nce_p95_ratio,
        args.best_geometry_iou_p05_ratio,
    )
    if any(eligibility_values):
        if not args.evaluate_init:
            raise ValueError("best_geometry eligibility gate 必须同时设置 --evaluate-init")
        if args.best_geometry_nce_p95_ratio < 1.0:
            raise ValueError("best-geometry-nce-p95-ratio 必须不小于 1")
        if not 0.0 < args.best_geometry_iou_p05_ratio <= 1.0:
            raise ValueError("best-geometry-iou-p05-ratio 必须位于 (0,1]")
    _seed_everything(args.seed)
    device = _device(args.device)
    train_data = QuadDataset(
        args.manifest,
        split="train",
        image_size=args.image_size,
        dataset_root=args.dataset_root,
        max_samples=args.train_samples,
        augmentation_mode=args.train_augmentation,
        seed=args.seed,
    )
    validation_data = QuadDataset(
        args.manifest,
        split=args.validation_split,
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
    frozen_modules = _configure_trainable_scope(model, args.trainable_scope)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("trainable-scope 没有留下可训参数")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    # 每次训练将可比较的实验元数据独立落盘；不记录图片内容，也不把运行产物放入仓库。
    run_metadata = {
        "format_version": 4,
        "dataset_manifest": str(args.manifest.expanduser().resolve()),
        "dataset_root": str(train_data.root),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "validation_split": args.validation_split,
        "train_augmentation": args.train_augmentation,
        "train_distribution": train_data.statistics(),
        "validation_distribution": validation_data.statistics(),
        "architecture": "QuadLocatorS",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_scope": args.trainable_scope,
        "frozen_modules": list(frozen_modules),
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
        "early_stopping_criterion": args.early_stopping_criterion,
        "geometry_collapse_watchdog": {
            "patience": args.geometry_collapse_patience,
            "nce_p95_ratio": args.geometry_collapse_nce_p95_ratio,
            "iou_p05_ratio": args.geometry_collapse_iou_p05_ratio,
        },
        "best_geometry_eligibility": {
            "nce_p95_ratio": args.best_geometry_nce_p95_ratio,
            "iou_p05_ratio": args.best_geometry_iou_p05_ratio,
        },
        "decoder": CornerDecoderSpec().to_dict(),
        "seed": args.seed,
        "loss_profile": args.loss_profile,
        "participating_losses": _participating_losses(args.loss_profile),
        "hard_sampling": args.hard_sampling,
        "evaluate_init": args.evaluate_init,
    }
    _warn_missing_training_domains(run_metadata["train_distribution"])
    (output_directory / "run.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    history: list[dict[str, float | int | str]] = []
    best_validation = float("inf")
    best_selection_score = float("-inf")
    best_geometry_key: tuple[float, float, float, float] | None = None
    epochs_without_improvement = 0
    collapse_epochs = 0
    watchdog_reference: dict[str, float] | None = None
    if args.evaluate_init:
        initial_loss, initial_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            None,
            0,
            args.epochs,
            loss_profile=args.loss_profile,
            collect_validation_metrics=True,
            frozen_modules=frozen_modules,
        )
        initial_validation = {
            "epoch": 0,
            "validation_loss": round(initial_loss, 8),
            "device": str(device),
            "validation_metrics": initial_metrics,
        }
        (output_directory / "initial_validation.json").write_text(
            json.dumps(initial_validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        watchdog_reference = {
            "content_corner_nce_p95": float(initial_metrics["content_corner_nce_p95"]),
            "content_iou_p05": float(initial_metrics["content_iou_p05"]),
        }
        if any(eligibility_values):
            # 若后续 epoch 全部突破 tail 安全线，best_geometry 必须保留可审计的
            # warm-start，而不是把最后一个退化 checkpoint 误标为 best。
            best_geometry_key = _geometry_selection_key(initial_metrics)
            initial_checkpoint = {
                "format_version": 4,
                "model": "QuadLocatorS",
                "width_multiplier": args.width_multiplier,
                "image_size": args.image_size,
                "class_order": ["artwork", "postcard", "screen", "none"],
                "parameter_count": run_metadata["parameter_count"],
                "state_dict": model.state_dict(),
                "epoch": 0,
                "validation_loss": initial_loss,
                "validation_metrics": initial_metrics,
                "decoder": CornerDecoderSpec().to_dict(),
                "seed": args.seed,
                "loss_profile": args.loss_profile,
                "trainable_scope": args.trainable_scope,
                "hard_sampling": args.hard_sampling,
            }
            torch.save(initial_checkpoint, output_directory / "best_geometry.pt")
        print(json.dumps(initial_validation, ensure_ascii=False), file=sys.stderr)
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
            frozen_modules=frozen_modules,
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
            frozen_modules=frozen_modules,
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
            "format_version": 4,
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
            "trainable_scope": args.trainable_scope,
            "hard_sampling": args.hard_sampling,
        }
        torch.save(checkpoint, output_directory / "last.pt")
        selection_score = float(validation_metrics["selection_score"])
        geometry_key = _geometry_selection_key(validation_metrics)
        geometry_eligible = not any(eligibility_values) or _is_geometry_eligible(
            validation_metrics,
            watchdog_reference,
            nce_p95_ratio=args.best_geometry_nce_p95_ratio,
            iou_p05_ratio=args.best_geometry_iou_p05_ratio,
        )
        geometry_improved = geometry_eligible and (
            best_geometry_key is None or geometry_key > best_geometry_key
        )
        if geometry_improved:
            best_geometry_key = geometry_key
            torch.save(checkpoint, output_directory / "best_geometry.pt")
        product_improved = selection_score > best_selection_score
        if product_improved:
            best_selection_score = selection_score
            best_validation = validation_loss
            torch.save(checkpoint, output_directory / "best_product.pt")
        improved = (
            geometry_improved
            if args.early_stopping_criterion == "geometry"
            else product_improved
        )
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.geometry_collapse_patience:
            assert watchdog_reference is not None
            collapsed = _is_geometry_tail_collapse(
                validation_metrics,
                watchdog_reference,
                nce_p95_ratio=args.geometry_collapse_nce_p95_ratio,
                iou_p05_ratio=args.geometry_collapse_iou_p05_ratio,
            )
            collapse_epochs = collapse_epochs + 1 if collapsed else 0
            if collapse_epochs >= args.geometry_collapse_patience:
                diagnostics = {
                    "status": "WATCHDOG_STOP",
                    "epoch": epoch,
                    "consecutive_epochs": collapse_epochs,
                    "reference": watchdog_reference,
                    "observed": {
                        "content_corner_nce_p95": float(
                            validation_metrics["content_corner_nce_p95"]
                        ),
                        "content_iou_p05": float(validation_metrics["content_iou_p05"]),
                    },
                    "thresholds": {
                        "nce_p95_max": watchdog_reference["content_corner_nce_p95"]
                        * args.geometry_collapse_nce_p95_ratio,
                        "iou_p05_min": watchdog_reference["content_iou_p05"]
                        * args.geometry_collapse_iou_p05_ratio,
                    },
                }
                (output_directory / "geometry-collapse-diagnostics.json").write_text(
                    json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    "geometry tail collapse watchdog 停止训练："
                    f"epoch={epoch} consecutive={collapse_epochs}",
                    file=sys.stderr,
                )
                break
        if (
            args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "early stopping："
                f"{args.early_stopping_criterion} 连续 {epochs_without_improvement} 轮未改善",
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
    run_metadata["best_geometry_key"] = list(best_geometry_key) if best_geometry_key else None
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
    frozen_modules: tuple[str, ...] = (),
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    if training:
        # requires_grad=False 不会阻止 BatchNorm running stats 更新；G1 必须让
        # 被冻结 head 保持 eval，才是真正的任务隔离。
        for name in frozen_modules:
            getattr(model, name).eval()
    total_loss = 0.0
    validation_metrics = ValidationMetrics() if collect_validation_metrics else None
    for batch_index, batch in enumerate(loader, start=1):
        _progress(
            batch_index - 1, len(loader), f"epoch {epoch}/{epochs} {'train' if training else 'val'}"
        )
        values = {key: tensor.to(device) for key, tensor in batch.items()}
        with torch.set_grad_enabled(training):
            outputs = model(values["image"])
            loss, _metrics = quadlocator_loss(outputs, values, profile=loss_profile)
            if validation_metrics is not None:
                validation_metrics.update(outputs, values)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=5.0,
                )
                optimizer.step()
        total_loss += float(loss.detach())
    _progress(len(loader), len(loader), f"epoch {epoch}/{epochs} {'train' if training else 'val'}")
    return (
        total_loss / max(1, len(loader)),
        validation_metrics.compute() if validation_metrics is not None else {},
    )


def _configure_trainable_scope(model: QuadLocatorS, scope: str) -> tuple[str, ...]:
    """配置 G1 的可训参数，返回需固定为 eval 的顶层模块。"""

    if scope == "all":
        return ()
    frozen = (
        "outer_corner_head",
        "content_mask_head",
        "boundary_head",
        "presence_head",
        "outer_presence_head",
        "class_head",
    )
    if scope == "content_head":
        frozen = (
            "stem",
            "stage2",
            "stage3",
            "stage4",
            "stage5",
            "lateral2",
            "lateral3",
            "lateral4",
            "lateral5",
            "fuse2",
            "fuse3",
            "fuse4",
            *frozen,
        )
    elif scope == "content_mask_backbone":
        # G2 的 mask head 必须可训练；其余非 content 任务则完全冻结，包含
        # BatchNorm running stats，避免它们在 shared backbone 上形成隐式干扰。
        frozen = (
            "outer_corner_head",
            "boundary_head",
            "presence_head",
            "outer_presence_head",
            "class_head",
        )
    elif scope == "content_boundary_backbone":
        # G3 与 G2 一样隔离非目标任务，只保留 boundary head 作为唯一辅助监督。
        frozen = (
            "outer_corner_head",
            "content_mask_head",
            "presence_head",
            "outer_presence_head",
            "class_head",
        )
    elif scope != "content_backbone":
        raise ValueError(f"未知 trainable scope：{scope}")
    for name in frozen:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(False)
    return frozen


def _participating_losses(profile: str) -> list[str]:
    """返回实际进入 total 的损失项，作为消融 run 的可审计契约。"""

    profiles = {
        "content_only": ["content_heatmap", "content_corner_geometry"],
        "content_heatmap_only": ["content_heatmap"],
        "content_coordinate_only": ["content_corner_geometry"],
        "content_mask": ["content_heatmap", "content_mask", "content_corner_geometry"],
        "content_boundary": [
            "content_heatmap",
            "balanced_boundary",
            "content_corner_geometry",
        ],
        "p2": [
            "content_heatmap",
            "outer_heatmap",
            "content_mask",
            "boundary",
            "presence",
            "outer_presence",
            "classification",
            "content_corner_geometry",
            "outer_corner_geometry",
        ],
        "boundary": [
            "content_heatmap",
            "outer_heatmap",
            "content_mask",
            "balanced_boundary",
            "presence",
            "outer_presence",
            "classification",
            "content_corner_geometry",
            "outer_corner_geometry",
        ],
        "tail": [
            "content_heatmap",
            "outer_heatmap",
            "content_mask",
            "boundary",
            "presence",
            "outer_presence",
            "classification",
            "content_corner_geometry",
            "outer_corner_geometry",
            "corner_cvar",
            "peak_ambiguity",
            "mask_quad_consistency",
            "corner_boundary_consistency",
        ],
        "full": [
            "content_heatmap",
            "outer_heatmap",
            "content_mask",
            "balanced_boundary",
            "presence",
            "outer_presence",
            "classification",
            "content_corner_geometry",
            "outer_corner_geometry",
            "corner_cvar",
            "peak_ambiguity",
            "mask_quad_consistency",
            "corner_boundary_consistency",
        ],
    }
    try:
        return profiles[profile]
    except KeyError as error:
        raise ValueError(f"未知 loss profile：{profile}") from error


def _geometry_selection_key(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    """以联合 tail 质量优先选几何 checkpoint，不混入 class/outer/rejection。

    NCE 已按图像对角线归一化，与 IoU 同处约 [0,1]。首项联合约束 P95 NCE 与
    P05 IoU，避免只因 P95 改善几个千分点就选择一个 P05 已经崩为 0 的 checkpoint。
    """

    nce_p95 = float(metrics["content_corner_nce_p95"])
    iou_p05 = float(metrics["content_iou_p05"])
    return (
        (1.0 - nce_p95) + iou_p05,
        float(metrics["content_iou_median"]),
        float(metrics["content_strict_correct_rate"]),
        -nce_p95,
    )


def _is_geometry_tail_collapse(
    metrics: dict[str, object],
    reference: dict[str, float],
    *,
    nce_p95_ratio: float,
    iou_p05_ratio: float,
) -> bool:
    """NCE tail 恶化且 IoU tail 同时坍塌时才触发，避免误杀有益更新。"""

    return (
        float(metrics["content_corner_nce_p95"])
        > reference["content_corner_nce_p95"] * nce_p95_ratio
        and float(metrics["content_iou_p05"])
        < reference["content_iou_p05"] * iou_p05_ratio
    )


def _is_geometry_eligible(
    metrics: dict[str, object],
    reference: dict[str, float] | None,
    *,
    nce_p95_ratio: float,
    iou_p05_ratio: float,
) -> bool:
    """只允许未突破 epoch 0 tail 安全线的 checkpoint 竞争 best_geometry。"""

    if reference is None:
        raise RuntimeError("best_geometry eligibility 缺少 epoch 0 reference")
    return (
        float(metrics["content_corner_nce_p95"])
        <= reference["content_corner_nce_p95"] * nce_p95_ratio
        and float(metrics["content_iou_p05"])
        >= reference["content_iou_p05"] * iou_p05_ratio
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
