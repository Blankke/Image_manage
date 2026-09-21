#!/usr/bin/env python3
"""按模块前缀插值两个 QuadLocator checkpoint，并用公开 validation 执行保护门。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/blend_quadlocator_checkpoints.py \
      --baseline /runs/p2/stage-b/best.pt \
      --challenger /runs/p5/content-tail/best_geometry.pt \
      --alpha 0.25 \
      --manifest /data/manifests/p2-public/stage-b.geometry.jsonl \
      --dataset-root /data \
      --device mps \
      --output-directory /runs/p5/content-blend/geometry

脚本只读取公开训练清单；默认仅混合 ``content_corner_head``。候选未通过相对
baseline 的 median/tail 保护门时仍保留审计报告，但不会生成 ``best_geometry.pt``。
当模块前缀涉及 presence/class 时，还会自动保护分类准确率、none recall 与目标类宏 recall。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# 允许按文件直接调用，并始终使用当前工作树的训练实现。
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from training.quadlocator.dataset import QuadDataset  # noqa: E402
from training.quadlocator.model import (  # noqa: E402
    QuadLocatorS,
    load_quadlocator_state_dict,
)
from training.quadlocator.train import _device, _run_epoch  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blend_state_dict(
    baseline: dict[str, torch.Tensor],
    challenger: dict[str, torch.Tensor],
    *,
    alpha: float,
    prefixes: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """只插值指定模块；允许 challenger 缺少未参与混合的新架构参数。

    baseline 定义完整产品结构。这样旧 checkpoint 可贡献仍然同构的 geometry head，
    而新增的局部决策头继续取 baseline；被选模块缺键或 shape 不同仍会立即拒绝。
    """

    result: dict[str, torch.Tensor] = {}
    blended: list[str] = []
    for name, baseline_tensor in baseline.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            if name not in challenger:
                raise ValueError(f"challenger 缺少被选参数：{name}")
            challenger_tensor = challenger[name]
            if baseline_tensor.shape != challenger_tensor.shape:
                raise ValueError(f"参数形状不一致：{name}")
            blended.append(name)
            if torch.is_floating_point(baseline_tensor):
                result[name] = baseline_tensor * (1.0 - alpha) + challenger_tensor * alpha
            else:
                source = challenger_tensor if alpha >= 0.5 else baseline_tensor
                result[name] = source.clone()
        else:
            result[name] = baseline_tensor.clone()
    challenger_only_selected = [
        name
        for name in challenger.keys() - baseline.keys()
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if challenger_only_selected:
        raise ValueError(f"baseline 缺少被选参数：{challenger_only_selected[0]}")
    if not blended:
        raise ValueError("module-prefix 没有匹配任何参数")
    return result, blended


def _geometry_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    keys = (
        "content_corner_nce_median",
        "content_corner_nce_p95",
        "content_iou_median",
        "content_iou_p05",
        "content_strict_correct_rate",
    )
    return {key: float(metrics[key]) for key in keys}


def _decision_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """提取决策头插值所需的公开保护指标。"""

    confusion = metrics["class_confusion"]
    total = max(1, sum(sum(int(value) for value in row) for row in confusion))
    accuracy = sum(int(confusion[index][index]) for index in range(4)) / total
    recall = metrics["class_recall"]
    target_macro_recall = sum(float(recall[name]) for name in ("artwork", "postcard", "screen")) / 3
    return {
        "class_accuracy": accuracy,
        "none_recall": float(recall["none"]),
        "target_macro_recall": target_macro_recall,
        "no_candidate_rate": float(metrics["no_candidate_rate"]),
    }


def _is_decision_blend(prefixes: tuple[str, ...]) -> bool:
    """类别或 presence 分支的任意插值都必须启用公开决策保护门。"""

    return any(
        prefix.startswith(
            (
                "presence_head.",
                "class_head.",
                "presence_local_head.",
                "class_local_head.",
                "class_context_encoder.",
                "class_context_head.",
                "artwork_context_head.",
            )
        )
        for prefix in prefixes
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--module-prefix", action="append", default=[])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--nce-p95-ratio", type=float, default=1.05)
    parser.add_argument("--iou-p05-ratio", type=float, default=0.95)
    parser.add_argument("--nce-median-tolerance", type=float, default=0.002)
    parser.add_argument("--iou-median-tolerance", type=float, default=0.01)
    parser.add_argument("--class-accuracy-ratio", type=float, default=0.99)
    parser.add_argument("--none-recall-ratio", type=float, default=0.99)
    parser.add_argument("--target-macro-recall-ratio", type=float, default=0.95)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha 必须位于 0..1")
    if args.validation_samples < 1 or args.batch_size < 1:
        raise ValueError("validation-samples 与 batch-size 必须大于 0")
    output = args.output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录：{output}")
    output.mkdir(parents=True)

    baseline_path = args.baseline.expanduser().resolve()
    challenger_path = args.challenger.expanduser().resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    challenger = torch.load(challenger_path, map_location="cpu", weights_only=False)
    for key in ("model", "width_multiplier", "image_size"):
        if baseline.get(key) != challenger.get(key):
            raise ValueError(f"checkpoint 契约不一致：{key}")
    prefixes = tuple(args.module_prefix or ["content_corner_head."])
    decision_blend = _is_decision_blend(prefixes)
    # 架构新增的零残差头先走与推理/导出相同的严格迁移，再参与插值。这样旧基线
    # 可与新分支 checkpoint 比较，同时不会把任意缺失参数默认为零。
    baseline_model = QuadLocatorS(float(baseline["width_multiplier"]))
    load_quadlocator_state_dict(baseline_model, baseline["state_dict"])
    challenger_model = QuadLocatorS(float(challenger["width_multiplier"]))
    load_quadlocator_state_dict(challenger_model, challenger["state_dict"])
    blended_state, blended_names = blend_state_dict(
        baseline_model.state_dict(),
        challenger_model.state_dict(),
        alpha=args.alpha,
        prefixes=prefixes,
    )

    data = QuadDataset(
        args.manifest,
        split="validation",
        image_size=int(baseline["image_size"]),
        dataset_root=args.dataset_root,
        max_samples=args.validation_samples,
        augment=False,
        seed=args.seed,
    )
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    selected_device = _device(args.device)

    def evaluate(state: dict[str, torch.Tensor]) -> dict[str, Any]:
        model = QuadLocatorS(float(baseline["width_multiplier"])).to(selected_device)
        load_quadlocator_state_dict(model, state)
        _, metrics = _run_epoch(
            model,
            loader,
            selected_device,
            None,
            0,
            0,
            loss_profile="p2",
            collect_validation_metrics=True,
        )
        return metrics

    baseline_metrics = evaluate(baseline["state_dict"])
    candidate_metrics = evaluate(blended_state)
    reference = _geometry_metrics(baseline_metrics)
    observed = _geometry_metrics(candidate_metrics)
    reference_decision = _decision_metrics(baseline_metrics)
    observed_decision = _decision_metrics(candidate_metrics)
    gates = {
        "nce_median": observed["content_corner_nce_median"]
        <= reference["content_corner_nce_median"] + args.nce_median_tolerance,
        "nce_p95": observed["content_corner_nce_p95"]
        <= reference["content_corner_nce_p95"] * args.nce_p95_ratio,
        "iou_median": observed["content_iou_median"]
        >= reference["content_iou_median"] - args.iou_median_tolerance,
        "iou_p05": observed["content_iou_p05"]
        >= reference["content_iou_p05"] * args.iou_p05_ratio,
    }
    if decision_blend:
        gates.update(
            {
                "class_accuracy": observed_decision["class_accuracy"]
                >= reference_decision["class_accuracy"] * args.class_accuracy_ratio,
                "none_recall": observed_decision["none_recall"]
                >= reference_decision["none_recall"] * args.none_recall_ratio,
                "target_macro_recall": observed_decision["target_macro_recall"]
                >= reference_decision["target_macro_recall"]
                * args.target_macro_recall_ratio,
            }
        )
    passed = all(gates.values())
    checkpoint = {
        **{key: value for key, value in baseline.items() if key != "state_dict"},
        "format_version": max(int(baseline.get("format_version", 0)), 5),
        "state_dict": blended_state,
        "epoch": 0,
        "validation_metrics": candidate_metrics,
        "blend": {
            "baseline": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "challenger": str(challenger_path),
            "challenger_sha256": _sha256(challenger_path),
            "alpha": args.alpha,
            "module_prefixes": list(prefixes),
        },
    }
    candidate_path = output / "candidate.pt"
    torch.save(checkpoint, candidate_path)
    if passed:
        torch.save(checkpoint, output / "best_geometry.pt")
    report = {
        "kind": "screenrestore_quadlocator_checkpoint_blend",
        "status": "PASS" if passed else "FAIL",
        "baseline": reference,
        "candidate": observed,
        "baseline_decision": reference_decision,
        "candidate_decision": observed_decision,
        "decision_gates_enabled": decision_blend,
        "gates": gates,
        "alpha": args.alpha,
        "module_prefixes": list(prefixes),
        "blended_tensor_count": len(blended_names),
        "candidate_checkpoint": str(candidate_path),
        "selected_checkpoint": str(output / "best_geometry.pt") if passed else None,
    }
    (output / "blend-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
