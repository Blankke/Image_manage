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
import hashlib
import json
import random
import sys
import time
from collections import Counter
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
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=0,
        help=(
            "CosineAnnealingLR 的固定 horizon；0 表示等于 epochs。"
            "短程轨迹重放可设为原实验总轮数，避免改变前段学习率"
        ),
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=_checkpoint_epochs,
        default=(),
        help="仅显式保存这些 epoch，例如 1,2,4,8,12,14,16",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--loss-profile",
        choices=(
            "decision_only",
            "decision_correction",
            "artwork_context_correction",
            "screen_context_correction",
            "content_confidence",
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
            "P5 decision_only 隔离训练目标存在性与类别；decision_correction "
            "在局部残差训练时蒸馏旧头的高置信正确判断；"
            "screen_context_correction 以二分类 margin 安全训练整图 screen 纠错分支；"
            "P4 G1=content_only、G2=content_mask、G3=content_boundary；"
            "G3.5 提供 content_heatmap_only/content_coordinate_only 语义诊断；"
            "P3 保留 p2/boundary/tail/full"
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=(
            "all",
            "decision_heads",
            "decision_residual_heads",
            "class_context_branch",
            "artwork_context_branch",
            "content_residual_head",
            "content_head",
            "content_backbone",
            "content_mask_backbone",
            "content_boundary_backbone",
        ),
        default="all",
        help=(
            "P5 decision_heads 只训 presence/class heads；"
            "decision_residual_heads 只训零初始化的局部证据残差头；"
            "class_context_branch 只训独立整图类别纠错分支；"
            "artwork_context_branch 只训独立 artwork 标量纠错头；"
            "content_residual_head 只训零初始化的 content heatmap 残差头；"
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
        "--class-balanced-sampling",
        action="store_true",
        help="先均衡 target_class，再在类内均衡 source/group；用于决策头训练",
    )
    parser.add_argument(
        "--focus-taxonomy",
        action="append",
        default=[],
        help="将指定 scene_type/hard_taxonomy 纳入定向采样；可重复传入",
    )
    parser.add_argument(
        "--focus-probability",
        type=float,
        default=0.0,
        help="每次抽样进入 focus taxonomy 池的概率；0 表示关闭",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="从旧 QuadLocator checkpoint 加载名称和 shape 均兼容的参数",
    )
    parser.add_argument(
        "--min-warm-start-parameter-ratio",
        type=float,
        default=0.0,
        help="热启动至少加载的参数值比例；同架构微调建议设为 0.95",
    )
    parser.add_argument("--train-samples", type=int, default=0, help="训练样本上限，0 表示全部")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="每轮从训练 split 抽取的样本数；0 表示等于加载后的训练集大小",
    )
    parser.add_argument(
        "--validation-samples", type=int, default=0, help="验证样本上限，0 表示全部"
    )
    parser.add_argument(
        "--require-train-source",
        action="append",
        default=[],
        metavar="SOURCE=MIN_COUNT",
        help="要求训练子集保留指定来源的最少样本数；可重复指定",
    )
    parser.add_argument(
        "--require-validation-source",
        action="append",
        default=[],
        metavar="SOURCE=MIN_COUNT",
        help="要求验证子集保留指定来源的最少样本数；可重复指定",
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
    parser.add_argument(
        "--best-geometry-nce-median-tolerance",
        type=float,
        default=0.002,
        help="best_geometry 相对 epoch 0 允许的 NCE median 绝对退化量",
    )
    parser.add_argument(
        "--best-geometry-iou-median-tolerance",
        type=float,
        default=0.01,
        help="best_geometry 相对 epoch 0 允许的 IoU median 绝对退化量",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须大于 0")
    scheduler_t_max = _resolve_scheduler_t_max(args.epochs, args.scheduler_t_max)
    if any(epoch > args.epochs for epoch in args.checkpoint_epochs):
        raise ValueError("checkpoint-epochs 不能超过实际 epochs")
    if args.train_samples < 0 or args.validation_samples < 0 or args.samples_per_epoch < 0:
        raise ValueError("样本上限不能为负数")
    if not 0.0 <= args.focus_probability <= 1.0:
        raise ValueError("focus-probability 必须位于 0..1")
    if bool(args.focus_taxonomy) != bool(args.focus_probability):
        raise ValueError("focus-taxonomy 与 focus-probability 必须同时启用")
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
        if args.best_geometry_nce_median_tolerance < 0.0:
            raise ValueError("best-geometry-nce-median-tolerance 不能为负数")
        if args.best_geometry_iou_median_tolerance < 0.0:
            raise ValueError("best-geometry-iou-median-tolerance 不能为负数")
    _assert_public_training_manifest(args.manifest)
    _seed_everything(args.seed)
    device = _device(args.device)
    train_data = QuadDataset(
        args.manifest,
        split="train",
        image_size=args.image_size,
        dataset_root=args.dataset_root,
        max_samples=args.train_samples,
        min_source_samples=_parse_source_requirements(
            args.require_train_source, "require-train-source"
        ),
        augmentation_mode=args.train_augmentation,
        seed=args.seed,
    )
    validation_data = QuadDataset(
        args.manifest,
        split=args.validation_split,
        image_size=args.image_size,
        dataset_root=args.dataset_root,
        max_samples=args.validation_samples,
        min_source_samples=_parse_source_requirements(
            args.require_validation_source, "require-validation-source"
        ),
        augment=False,
        seed=args.seed,
    )
    _check_required_sources(train_data.records, args.require_train_source, split="train")
    _check_required_sources(
        validation_data.records, args.require_validation_source, split="validation"
    )
    sampler = SourceGroupBalancedSampler(
        train_data,
        seed=args.seed,
        difficulty_weighting=args.hard_sampling,
        class_balancing=args.class_balanced_sampling,
        samples_per_epoch=args.samples_per_epoch,
        focus_taxonomies=tuple(args.focus_taxonomy),
        focus_probability=args.focus_probability,
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
    if not 0.0 <= args.min_warm_start_parameter_ratio <= 1.0:
        raise ValueError("min-warm-start-parameter-ratio 必须位于 [0,1]")
    if args.min_warm_start_parameter_ratio and args.init_checkpoint is None:
        raise ValueError("设置热启动覆盖率门槛时必须提供 init-checkpoint")
    warm_start = None
    if args.init_checkpoint is not None:
        warm_start = _load_compatible_checkpoint(model, args.init_checkpoint)
        _assert_warm_start_coverage(
            warm_start,
            sum(parameter.numel() for parameter in model.parameters()),
            args.min_warm_start_parameter_ratio,
        )
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    if args.checkpoint_epochs:
        _validate_milestone_targets(checkpoint_directory, args.checkpoint_epochs)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    # 每次训练将可比较的实验元数据独立落盘；不记录图片内容，也不把运行产物放入仓库。
    run_metadata = {
        "format_version": 5,
        "dataset_manifest": str(args.manifest.expanduser().resolve()),
        "dataset_root": str(train_data.root),
        "train_samples": len(train_data),
        "samples_per_epoch": len(sampler),
        "validation_samples": len(validation_data),
        "validation_split": args.validation_split,
        "train_augmentation": args.train_augmentation,
        "train_distribution": train_data.statistics(),
        "validation_distribution": validation_data.statistics(),
        "architecture": "QuadLocatorS",
        "width_multiplier": args.width_multiplier,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_scope": args.trainable_scope,
        "frozen_modules": list(frozen_modules),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "scheduler_t_max": scheduler_t_max,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "init_checkpoint": str(args.init_checkpoint.expanduser().resolve())
        if args.init_checkpoint is not None
        else None,
        "warm_start": warm_start,
        "min_warm_start_parameter_ratio": args.min_warm_start_parameter_ratio,
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
            "nce_median_tolerance": args.best_geometry_nce_median_tolerance,
            "iou_median_tolerance": args.best_geometry_iou_median_tolerance,
        },
        "validation_metric_protocol": {
            "corner_decoder": CornerDecoderSpec().version,
            "metric_version": 2,
            "nce_normalization": "target_quad_bbox_diagonal",
        },
        "decoder": CornerDecoderSpec().to_dict(),
        "seed": args.seed,
        "loss_profile": args.loss_profile,
        "participating_losses": _participating_losses(args.loss_profile),
        "hard_sampling": args.hard_sampling,
        "class_balanced_sampling": args.class_balanced_sampling,
        "focus_sampling": {
            "taxonomies": list(sampler._focus_taxonomies),
            "probability": args.focus_probability,
        },
        "evaluate_init": args.evaluate_init,
        "milestone_checkpoints": {
            "requested_epochs": list(args.checkpoint_epochs),
            "saved_epochs": [],
            "saved": [],
        },
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
            "content_corner_nce_median": float(initial_metrics["content_corner_nce_median"]),
            "content_corner_nce_p95": float(initial_metrics["content_corner_nce_p95"]),
            "content_iou_median": float(initial_metrics["content_iou_median"]),
            "content_iou_p05": float(initial_metrics["content_iou_p05"]),
        }
        initial_checkpoint = {
            "format_version": 5,
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
        # product 选模也必须把 warm-start 纳入候选。否则第一轮即使降低综合选择分数，
        # 仍会因为 best_selection_score 的 -inf 初值被错误保存为 best_product。
        best_selection_score = float(initial_metrics["selection_score"])
        best_validation = initial_loss
        torch.save(initial_checkpoint, output_directory / "best_product.pt")
        if any(eligibility_values):
            # 若后续 epoch 全部突破 tail 安全线，best_geometry 必须保留可审计的
            # warm-start，而不是把最后一个退化 checkpoint 误标为 best。
            best_geometry_key = _geometry_selection_key(initial_metrics)
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
            "format_version": 5,
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
        if epoch in args.checkpoint_epochs:
            milestone = _save_milestone_checkpoint(checkpoint, checkpoint_directory, epoch)
            milestone["validation_metrics"] = validation_metrics
            run_metadata["milestone_checkpoints"]["saved_epochs"].append(epoch)
            run_metadata["milestone_checkpoints"]["saved"].append(milestone)
            # 长训中断时也必须保留已经落盘的里程碑身份，而非只在正常结束时补写。
            _write_run_metadata(output_directory, run_metadata)
        selection_score = float(validation_metrics["selection_score"])
        geometry_key = _geometry_selection_key(validation_metrics)
        geometry_eligible = not any(eligibility_values) or _is_geometry_eligible(
            validation_metrics,
            watchdog_reference,
            nce_p95_ratio=args.best_geometry_nce_p95_ratio,
            iou_p05_ratio=args.best_geometry_iou_p05_ratio,
            nce_median_tolerance=args.best_geometry_nce_median_tolerance,
            iou_median_tolerance=args.best_geometry_iou_median_tolerance,
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
            geometry_improved if args.early_stopping_criterion == "geometry" else product_improved
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
    _write_run_metadata(output_directory, run_metadata)
    return 0


def _checkpoint_epochs(value: str) -> tuple[int, ...]:
    """解析显式里程碑列表；排序和去重让输出路径具有唯一语义。"""

    if not value.strip():
        return ()
    try:
        epochs = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint-epochs 必须是逗号分隔的整数") from exc
    if any(epoch < 1 for epoch in epochs):
        raise argparse.ArgumentTypeError("checkpoint-epochs 必须全部大于 0")
    return epochs


def _check_required_sources(
    records: list[dict[str, object]], requirements: list[str], *, split: str
) -> None:
    """阻止随机抽样漏掉稀有目标域，训练与验证分别检查。"""

    counts = Counter(str(record.get("source", "unknown")) for record in records)
    option = "require-train-source" if split == "train" else "require-validation-source"
    for source, minimum in _parse_source_requirements(requirements, option).items():
        actual = counts.get(source, 0)
        if actual < minimum:
            raise ValueError(
                f"{split} 来源 {source} 仅有 {actual} 张，低于要求的 {minimum} 张；"
                "请扩大样本上限或重新制定分组数据集"
            )


def _parse_source_requirements(requirements: list[str], option: str) -> dict[str, int]:
    """统一解析来源配额，供训练与验证抽样及最终校验共用。"""

    parsed: dict[str, int] = {}
    for item in requirements:
        source, separator, minimum_text = item.rpartition("=")
        if not separator or not source.strip() or not minimum_text.isdecimal():
            raise ValueError(f"{option} 必须是 SOURCE=正整数")
        minimum = int(minimum_text)
        if minimum < 1 or source in parsed:
            raise ValueError(f"{option} 来源须唯一且最少样本数大于 0")
        parsed[source] = minimum
    return parsed


def _resolve_scheduler_t_max(epochs: int, requested: int) -> int:
    """解析独立 scheduler horizon，允许只运行原完整轨迹的前若干轮。"""

    resolved = requested or epochs
    if resolved < epochs:
        raise ValueError("scheduler-t-max 必须不小于实际 epochs")
    return resolved


def _validate_milestone_targets(directory: Path, epochs: tuple[int, ...]) -> None:
    """训练开始前拒绝覆盖任一已有里程碑，避免半程才发现产物冲突。"""

    existing = [directory / f"epoch-{epoch:03d}.pt" for epoch in epochs]
    conflicts = [path for path in existing if path.exists()]
    if conflicts:
        raise FileExistsError(f"拒绝覆盖已有 milestone checkpoint：{conflicts[0]}")


def _save_milestone_checkpoint(
    checkpoint: dict[str, object],
    directory: Path,
    epoch: int,
) -> dict[str, object]:
    """保存当轮 checkpoint 并返回可写入 run.json 的内容身份。"""

    path = directory / f"epoch-{epoch:03d}.pt"
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有 milestone checkpoint：{path}")
    if int(checkpoint.get("epoch", -1)) != epoch:
        raise ValueError("milestone epoch 与 checkpoint state 不一致")
    torch.save(checkpoint, path)
    return {
        "epoch": epoch,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_metadata(output_directory: Path, metadata: dict[str, object]) -> None:
    (output_directory / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    task_heads = (
        "outer_corner_head",
        "content_corner_residual_head",
        "content_mask_head",
        "boundary_head",
        "presence_head",
        "presence_local_head",
        "outer_presence_head",
        "class_head",
        "class_local_head",
        "class_context_encoder",
        "class_context_avg_pool",
        "class_context_max_pool",
        "class_context_head",
        "artwork_context_head",
    )
    frozen = task_heads
    if scope == "decision_heads":
        frozen = tuple(
            name
            for name, _module in model.named_children()
            if name
            not in {
                "presence_head",
                "class_head",
                "presence_local_head",
                "class_local_head",
            }
        )
    elif scope == "decision_residual_heads":
        frozen = tuple(
            name
            for name, _module in model.named_children()
            if name not in {"presence_local_head", "class_local_head"}
        )
    elif scope == "class_context_branch":
        frozen = tuple(
            name
            for name, _module in model.named_children()
            if name
            not in {
                "class_context_encoder",
                "class_context_avg_pool",
                "class_context_max_pool",
                "class_context_head",
            }
        )
    elif scope == "artwork_context_branch":
        frozen = tuple(
            name for name, _module in model.named_children() if name != "artwork_context_head"
        )
    elif scope == "content_residual_head":
        frozen = tuple(
            name
            for name, _module in model.named_children()
            if name != "content_corner_residual_head"
        )
    elif scope == "content_head":
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
            "content_corner_residual_head",
            "boundary_head",
            "presence_head",
            "outer_presence_head",
            "class_head",
        )
    elif scope == "content_boundary_backbone":
        # G3 与 G2 一样隔离非目标任务，只保留 boundary head 作为唯一辅助监督。
        frozen = (
            "outer_corner_head",
            "content_corner_residual_head",
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
        "decision_only": ["presence", "classification"],
        "decision_correction": [
            "presence_correction",
            "classification_correction",
            "baseline_distillation",
        ],
        "artwork_context_correction": [
            "artwork_vs_rest_correction",
            "correct_binary_margin_distillation",
        ],
        "screen_context_correction": [
            "screen_vs_rest_correction",
            "correct_binary_margin_distillation",
        ],
        "content_confidence": [
            "content_heatmap",
            "content_corner_geometry",
            "corner_cvar",
            "peak_ambiguity",
            "baseline_distillation",
        ],
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

    NCE 已按目标包围盒对角线归一化。首项联合约束 P95 NCE 与 P05 IoU，
    次项联合约束 median，避免 tail 改善时静默牺牲典型样本。
    """

    nce_p95 = float(metrics["content_corner_nce_p95"])
    iou_p05 = float(metrics["content_iou_p05"])
    return (
        (1.0 - nce_p95) + iou_p05,
        (1.0 - float(metrics["content_corner_nce_median"])) + float(metrics["content_iou_median"]),
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
        and float(metrics["content_iou_p05"]) < reference["content_iou_p05"] * iou_p05_ratio
    )


def _is_geometry_eligible(
    metrics: dict[str, object],
    reference: dict[str, float] | None,
    *,
    nce_p95_ratio: float,
    iou_p05_ratio: float,
    nce_median_tolerance: float,
    iou_median_tolerance: float,
) -> bool:
    """只允许同时守住 epoch 0 median 与 tail 的 checkpoint 竞争 best_geometry。"""

    if reference is None:
        raise RuntimeError("best_geometry eligibility 缺少 epoch 0 reference")
    return (
        float(metrics["content_corner_nce_median"])
        <= reference["content_corner_nce_median"] + nce_median_tolerance
        and float(metrics["content_iou_median"])
        >= reference["content_iou_median"] - iou_median_tolerance
        and float(metrics["content_corner_nce_p95"])
        <= reference["content_corner_nce_p95"] * nce_p95_ratio
        and float(metrics["content_iou_p05"]) >= reference["content_iou_p05"] * iou_p05_ratio
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


def _assert_warm_start_coverage(
    warm_start: dict[str, object], parameter_count: int, minimum_ratio: float
) -> None:
    """阻止宽度或架构填错后只加载少量参数却继续完整训练。"""

    loaded = int(warm_start["loaded_parameter_count"])
    ratio = loaded / max(1, parameter_count)
    if ratio < minimum_ratio:
        raise ValueError(
            f"热启动仅加载 {loaded}/{parameter_count} 个参数值，比例 {ratio:.3f} "
            f"低于要求的 {minimum_ratio:.3f}；请核对模型宽度与 checkpoint"
        )


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


def _assert_public_training_manifest(manifest: Path) -> None:
    """训练入口拒绝私人照片；开发验收由独立冻结预测流程读取。"""

    path = manifest.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            _assert_public_training_record(row, line_number)


def _assert_public_training_record(row: dict, line_number: int = 0) -> None:
    """训练、校准和模型选择共用同一条私人集身份检查。"""

    image_parts = Path(str(row.get("image", ""))).parts
    # 不同来源清单可能只在拍摄批次或作品字段保留私人集标识。
    private_identifiers = (
        "source", "group_id", "capture_session", "subject_id", "source_capture"
    )
    if (
        any(str(row.get(key, "")).casefold().startswith("private") for key in private_identifiers)
        or any(part.casefold().startswith("private") for part in image_parts)
    ):
        raise ValueError(f"训练清单第 {line_number} 行引用私人数据")


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
    # 终端仍显示连续进度，但日志最多保留约 100 次刷新，避免长训练产生海量输出。
    interval = max(1, total // 100)
    if done < total and done != 0 and done % interval != 0:
        return
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
