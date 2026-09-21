#!/usr/bin/env python3
"""用 validation/calibration 选择 P4 coordinate-only geometry checkpoint。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/evaluate_p4_geometry_trajectory.py \
      --baseline /runs/p2/stage-b/best.pt \
      --checkpoints /runs/p4/checkpoints/epoch-*.pt \
      --internal-manifest /data/manifests/p2-public/stage-b.geometry.jsonl \
      --calibration-manifest /data/manifests/p2-public/calibration-public.geometry.jsonl \
      --smartdoc-manifest /data/manifests/smartdoc.geometry.jsonl \
      --dataset-root /data --evaluation-image-size 512 \
      --device mps --output-directory /runs/p4/trajectory

该脚本只接受 validation split，并在全部照片推理冻结后读取完整标注。选择采用预先声明的
B0 eligibility 安全线、Pareto front 与确定性的 tail-first 字典序，不读取 SmartDoc test。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from benchmarks.geometry_e2e.run import (  # noqa: E402
    _project_json_string,
    _resolve_manifest_image,
)
from scripts.audit_p4_geometry_parity import _prediction_from_raw  # noqa: E402
from training.quadlocator.model import (  # noqa: E402
    QuadLocatorS,
    load_quadlocator_state_dict,
)
from training.quadlocator.train import _assert_public_training_manifest  # noqa: E402

from screenrestore.geometry.detector import _letterbox_tensor  # noqa: E402
from screenrestore.io.image_loader import load_image  # noqa: E402
from screenrestore.validation.geometry_benchmark import corner_metrics  # noqa: E402

OUTPUT_NAMES = (
    "content_corner_heatmaps",
    "outer_corner_heatmaps",
    "content_mask_logits",
    "boundary_logits",
    "presence_logits",
    "outer_presence_logits",
    "class_logits",
)
DATASET_ORDER = ("internal_validation", "calibration", "smartdoc_validation")
METRIC_DIRECTIONS = {
    "corner_nce_median": "min",
    "corner_nce_p95": "min",
    "quad_iou_median": "max",
    "quad_iou_p05": "max",
}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    manifest: Path
    paths: tuple[Path, ...]
    selected_line_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    label: str
    path: Path
    epoch: int
    image_size: int
    width_multiplier: float
    identity: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--internal-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--smartdoc-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--internal-max-samples", type=int, default=1000)
    parser.add_argument("--internal-seed", type=int, default=20260902)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--evaluation-image-size",
        type=int,
        default=0,
        help="所有 checkpoint 共用的推理边长；0 表示采用 B0 checkpoint 的原生边长",
    )
    parser.add_argument("--nce-median-tolerance", type=float, default=0.002)
    parser.add_argument("--nce-p95-tolerance", type=float, default=0.005)
    parser.add_argument("--iou-median-tolerance", type=float, default=0.01)
    parser.add_argument("--iou-p05-tolerance", type=float, default=0.01)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.internal_max_samples < 1 or args.batch_size < 1:
        raise ValueError("internal-max-samples 与 batch-size 必须大于 0")
    tolerances = {
        "corner_nce_median": args.nce_median_tolerance,
        "corner_nce_p95": args.nce_p95_tolerance,
        "quad_iou_median": args.iou_median_tolerance,
        "quad_iou_p05": args.iou_p05_tolerance,
    }
    if any(value < 0.0 for value in tolerances.values()):
        raise ValueError("eligibility tolerance 不能为负数")
    # 该入口会选择 checkpoint，所有参与排序的清单均须为公开数据。
    for manifest in (args.internal_manifest, args.calibration_manifest, args.smartdoc_manifest):
        _assert_public_training_manifest(manifest)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f"拒绝覆盖已有 trajectory 目录：{output_directory}")
    output_directory.mkdir(parents=True)
    started = time.monotonic()
    root = args.dataset_root.expanduser().resolve()
    datasets_list = [
        _dataset_spec(
            "internal_validation",
            args.internal_manifest,
            root,
            max_samples=args.internal_max_samples,
            seed=args.internal_seed,
        ),
        _dataset_spec("calibration", args.calibration_manifest, root),
        _dataset_spec("smartdoc_validation", args.smartdoc_manifest, root),
    ]
    datasets = tuple(datasets_list)
    dataset_order = tuple(dataset.name for dataset in datasets)
    all_paths = tuple(sorted({path for dataset in datasets for path in dataset.paths}))
    checkpoints = _checkpoint_specs(args.baseline, args.checkpoints)
    evaluation_image_size = _resolve_evaluation_image_size(
        args.evaluation_image_size,
        baseline_image_size=checkpoints[0].image_size,
    )
    device = _device(args.device)
    frozen_predictions: dict[str, dict[Path, dict[str, Any]]] = {}
    for index, checkpoint in enumerate(checkpoints, start=1):
        frozen_predictions[checkpoint.label] = _infer_checkpoint(
            checkpoint,
            all_paths,
            device=device,
            batch_size=args.batch_size,
            progress_prefix=f"checkpoint {index}/{len(checkpoints)} {checkpoint.label}",
            evaluation_image_size=evaluation_image_size,
        )

    # 所有 checkpoint 的照片推理完成后才物化 validation GT。
    reports: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        truths = _truth_by_path(dataset, root)
        checkpoint_reports = []
        for checkpoint in checkpoints:
            cases = [
                _score_prediction(frozen_predictions[checkpoint.label][path], truths[path])
                for path in dataset.paths
            ]
            checkpoint_reports.append(
                {
                    "label": checkpoint.label,
                    "epoch": checkpoint.epoch,
                    "checkpoint": checkpoint.identity,
                    "metrics": _aggregate_cases(cases),
                }
            )
        report = {
            "format_version": 2,
            "metric_version": 2,
            "nce_normalization": "target_quad_bbox_diagonal",
            "kind": "p4_geometry_trajectory_validation",
            "protocol": "photo_only_then_frozen_validation_gt_common_input",
            "split": "validation",
            "dataset": dataset.name,
            "manifest": _file_identity(dataset.manifest),
            "sample_count": len(dataset.paths),
            "independent_group_count": len({truths[path]["group_id"] for path in dataset.paths}),
            "evaluation_image_size": evaluation_image_size,
            "checkpoints": checkpoint_reports,
        }
        reports[dataset.name] = report
        filename = {
            "internal_validation": "trajectory-validation.json",
            "calibration": "trajectory-calibration.json",
            "smartdoc_validation": "trajectory-smartdoc-validation.json",
        }[dataset.name]
        _write_json(output_directory / filename, report)

    selection = select_checkpoint(reports, tolerances, dataset_order=dataset_order)
    summary = {
        "format_version": 2,
        "metric_version": 2,
        "nce_normalization": "target_quad_bbox_diagonal",
        "kind": "p4_geometry_trajectory_summary",
        "protocol": "validation_only_checkpoint_selection_common_input",
        "test_data_used": False,
        "evaluation_image_size": evaluation_image_size,
        "dataset_order": list(dataset_order),
        "selection_order": [
            "SmartDoc validation: lower NCE P95",
            "SmartDoc validation: higher IoU P05",
            "SmartDoc validation: lower NCE median",
            "SmartDoc validation: higher IoU median",
            "calibration 同顺序",
            "internal validation 同顺序",
        ],
        "eligibility_tolerances_absolute": tolerances,
        "selection": selection,
        "checkpoints": [
            {
                "label": item.label,
                "epoch": item.epoch,
                "declared_image_size": item.image_size,
                "checkpoint": item.identity,
            }
            for item in checkpoints
        ],
        "datasets": {
            name: {
                "sample_count": reports[name]["sample_count"],
                "independent_group_count": reports[name]["independent_group_count"],
                "manifest": reports[name]["manifest"],
            }
            for name in dataset_order
        },
        "dataset_sample_overlap": _dataset_overlap(datasets),
        "git": _git_metadata(),
        "command": [sys.executable, *sys.argv],
        "wall_time_seconds": round(time.monotonic() - started, 4),
    }
    winner = selection["winner"]
    if winner is not None:
        winner_spec = next(item for item in checkpoints if item.label == winner)
        frozen_path = output_directory / "frozen-geometry.pt"
        shutil.copy2(winner_spec.path, frozen_path)
        frozen_identity = _file_identity(frozen_path)
        frozen_identity["source"] = winner_spec.identity
        summary["frozen_checkpoint"] = frozen_identity
    else:
        summary["frozen_checkpoint"] = None
    _write_json(output_directory / "trajectory-summary.json", summary)
    (output_directory / "trajectory-report.md").write_text(
        _markdown_report(reports, summary), encoding="utf-8"
    )
    print(output_directory / "trajectory-summary.json")
    return 0 if winner is not None else 3


def select_checkpoint(
    reports: dict[str, dict[str, Any]],
    tolerances: dict[str, float],
    *,
    dataset_order: tuple[str, ...] = DATASET_ORDER,
) -> dict[str, Any]:
    """以 B0 四项非退化约束筛选，再对 eligible 集合做 Pareto/字典序选择。"""

    by_dataset = {
        name: {item["label"]: item["metrics"] for item in reports[name]["checkpoints"]}
        for name in dataset_order
    }
    labels = [
        item["label"] for item in reports[dataset_order[0]]["checkpoints"] if item["label"] != "B0"
    ]
    eligibility: dict[str, Any] = {}
    eligible: list[str] = []
    for label in labels:
        checks: dict[str, dict[str, bool]] = {}
        for dataset in dataset_order:
            baseline = by_dataset[dataset]["B0"]
            observed = by_dataset[dataset][label]
            checks[dataset] = {
                "corner_nce_median": observed["corner_nce_median"]
                <= baseline["corner_nce_median"] + tolerances["corner_nce_median"],
                "corner_nce_p95": observed["corner_nce_p95"]
                <= baseline["corner_nce_p95"] + tolerances["corner_nce_p95"],
                "quad_iou_median": observed["quad_iou_median"]
                >= baseline["quad_iou_median"] - tolerances["quad_iou_median"],
                "quad_iou_p05": observed["quad_iou_p05"]
                >= baseline["quad_iou_p05"] - tolerances["quad_iou_p05"],
            }
        passed = all(all(values.values()) for values in checks.values())
        eligibility[label] = {"eligible": passed, "checks": checks}
        if passed:
            eligible.append(label)
    pareto = _pareto_front(labels, by_dataset, dataset_order)
    # Pareto 决策只在已经通过 B0 安全约束的集合内进行；不能让一个不合格但极端偏科的
    # checkpoint 把所有合格候选从 Pareto front 上挤掉。
    eligible_pareto = _pareto_front(eligible, by_dataset, dataset_order)
    winner = (
        min(
            eligible_pareto,
            key=lambda label: _selection_key(label, by_dataset, dataset_order),
        )
        if eligible_pareto
        else None
    )
    return {
        "baseline": "B0",
        "eligibility": eligibility,
        "eligible_checkpoints": eligible,
        "pareto_front": pareto,
        "eligible_pareto_front": eligible_pareto,
        "winner": winner,
        "status": "FROZEN" if winner is not None else "NO_ELIGIBLE_CHECKPOINT",
    }


def _selection_key(
    label: str,
    values: dict[str, dict[str, dict[str, Any]]],
    dataset_order: tuple[str, ...],
) -> tuple[float, ...]:
    result: list[float] = []
    priority = [
        name
        for name in ("smartdoc_validation", "calibration", "internal_validation")
        if name in dataset_order
    ]
    priority.extend(name for name in dataset_order if name not in priority)
    for dataset in priority:
        metrics = values[dataset][label]
        result.extend(
            (
                float(metrics["corner_nce_p95"]),
                -float(metrics["quad_iou_p05"]),
                float(metrics["corner_nce_median"]),
                -float(metrics["quad_iou_median"]),
            )
        )
    return tuple(result)


def _pareto_front(
    labels: list[str],
    values: dict[str, dict[str, dict[str, Any]]],
    dataset_order: tuple[str, ...],
) -> list[str]:
    vectors = {label: _quality_vector(label, values, dataset_order) for label in labels}
    front = []
    for label in labels:
        dominated = any(
            other != label
            and all(a <= b for a, b in zip(vectors[other], vectors[label], strict=True))
            and any(a < b for a, b in zip(vectors[other], vectors[label], strict=True))
            for other in labels
        )
        if not dominated:
            front.append(label)
    return front


def _quality_vector(
    label: str,
    values: dict[str, dict[str, dict[str, Any]]],
    dataset_order: tuple[str, ...],
) -> tuple[float, ...]:
    result = []
    for dataset in dataset_order:
        metrics = values[dataset][label]
        for metric, direction in METRIC_DIRECTIONS.items():
            value = float(metrics[metric])
            result.append(value if direction == "min" else -value)
    return tuple(result)


def _dataset_spec(
    name: str,
    manifest: Path,
    root: Path,
    *,
    max_samples: int = 0,
    seed: int = 0,
) -> DatasetSpec:
    path = manifest.expanduser().resolve()
    scheduled: list[tuple[int, Path]] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        split = _project_json_string(line, "split")
        if split != "validation":
            continue
        image = _project_json_string(line, "image")
        if image is None:
            raise ValueError(f"{path} 第 {line_index + 1} 行缺少 image")
        resolved = _resolve_manifest_image(root, image)
        if not resolved.is_file():
            raise ValueError(f"validation 图片不存在：{resolved}")
        scheduled.append((line_index, resolved))
    if max_samples and max_samples < len(scheduled):
        selected = np.random.default_rng(seed).choice(
            len(scheduled), size=max_samples, replace=False
        )
        scheduled = [scheduled[int(index)] for index in sorted(selected)]
    if not scheduled:
        raise ValueError(f"{path} 没有 validation 样本")
    return DatasetSpec(
        name=name,
        manifest=path,
        paths=tuple(item[1] for item in scheduled),
        selected_line_indices=tuple(item[0] for item in scheduled),
    )


def _checkpoint_specs(baseline: Path, paths: list[Path]) -> tuple[CheckpointSpec, ...]:
    expanded: list[Path] = []
    for raw in paths:
        # shell 通常会先展开 glob；这里保留明确路径语义并拒绝隐式目录扫描。
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"checkpoint 不存在：{path}")
        expanded.append(path)
    ordered = [baseline.expanduser().resolve(), *expanded]
    specs = []
    labels = set()
    for index, path in enumerate(ordered):
        if not path.is_file():
            raise ValueError(f"checkpoint 不存在：{path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint.get("epoch", 0))
        label = "B0" if index == 0 else f"epoch-{epoch:03d}"
        if label in labels:
            raise ValueError(f"checkpoint label 重复：{label}")
        labels.add(label)
        specs.append(
            CheckpointSpec(
                label=label,
                path=path,
                epoch=epoch,
                image_size=int(checkpoint["image_size"]),
                width_multiplier=float(checkpoint["width_multiplier"]),
                identity=_file_identity(path),
            )
        )
    return tuple(specs)


def _infer_checkpoint(
    checkpoint: CheckpointSpec,
    paths: tuple[Path, ...],
    *,
    device: torch.device,
    batch_size: int,
    progress_prefix: str,
    evaluation_image_size: int,
) -> dict[Path, dict[str, Any]]:
    payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    model = QuadLocatorS(checkpoint.width_multiplier)
    load_quadlocator_state_dict(model, payload["state_dict"])
    model.to(device).eval()
    predictions: dict[Path, dict[str, Any]] = {}
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        documents = [load_image(path) for path in batch_paths]
        prepared = [
            _letterbox_tensor(document.original_rgb, evaluation_image_size)
            for document in documents
        ]
        tensor = np.concatenate([item[0] for item in prepared], axis=0)
        with torch.inference_mode():
            outputs = model(torch.from_numpy(tensor).to(device))
        raw_batch = {name: outputs[name].detach().cpu().numpy() for name in OUTPUT_NAMES}
        for local_index, (path, document, prepared_item) in enumerate(
            zip(batch_paths, documents, prepared, strict=True)
        ):
            raw = {name: raw_batch[name][local_index : local_index + 1] for name in OUTPUT_NAMES}
            prediction = _prediction_from_raw(
                raw,
                prepared_item[1],
                document.original_rgb.shape,
                "decoder_v2",
            )
            scale = np.array(
                [
                    max(1, document.original_rgb.shape[1] - 1),
                    max(1, document.original_rgb.shape[0] - 1),
                ],
                np.float32,
            )
            corners = prediction.decoder_diagnostics["content"]["corners"]
            predictions[path] = {
                # corner_metrics 以目标包围盒对角线归一化，必须传像素坐标；若提前
                # 压到 [0,1]，其内部最小分母 1 会悄悄改变 NCE 定义。
                "content_quad": prediction.content_quad,
                "image_scale": scale,
                "target_class": prediction.target_class.value,
                "class_confidence": float(prediction.class_confidence),
                "presence": float(prediction.presence_confidence),
                "layer": float(prediction.layer_confidence),
                "corner_confidences": list(prediction.corner_confidences),
                "corner_diagnostics": corners,
                "candidate_margin": (
                    float(prediction.candidates[0].scores["candidate_margin"])
                    if prediction.candidates
                    else 0.0
                ),
            }
        _progress(min(start + len(batch_paths), len(paths)), len(paths), progress_prefix)
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    return predictions


def _resolve_evaluation_image_size(requested: int, *, baseline_image_size: int) -> int:
    """强制公平比较：所有模型必须在同一推理尺寸上接受评估。"""

    resolved = requested or baseline_image_size
    if resolved < 128 or resolved % 32:
        raise ValueError("evaluation-image-size 必须不小于 128 且为 32 的倍数")
    return resolved


def _truth_by_path(dataset: DatasetSpec, root: Path) -> dict[Path, dict[str, Any]]:
    output: dict[Path, dict[str, Any]] = {}
    raw_lines = [line for line in dataset.manifest.read_text(encoding="utf-8").splitlines()]
    for line_index in dataset.selected_line_indices:
        record = json.loads(raw_lines[line_index])
        path = _resolve_manifest_image(root, str(record["image"]))
        output[path] = {
            "present": bool(record["present"]),
            "target_class": str(record["target_class"]),
            "content_quad": (
                np.asarray(record["content_quad"], np.float32)
                if record.get("content_quad") is not None
                else None
            ),
            "group_id": str(record["group_id"]),
        }
    if len(output) != len(dataset.paths):
        raise ValueError(f"{dataset.name} 存在重复 image，无法建立一对一评分")
    return output


def _score_prediction(prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    quad = prediction["content_quad"]
    target_quad = truth["content_quad"]
    nce, iou = 1.0, 0.0
    if quad is not None and target_quad is not None:
        target_pixels = target_quad * prediction["image_scale"]
        nce, iou, _maximum = corner_metrics(quad, target_pixels)
    strict_geometry = bool(truth["present"] and quad is not None and nce <= 0.01 and iou >= 0.93)
    corners = prediction["corner_diagnostics"]
    return {
        "present": truth["present"],
        "has_candidate": quad is not None,
        "strict_correct": strict_geometry,
        "semantic_strict_correct": bool(
            strict_geometry and prediction["target_class"] == truth["target_class"]
        ),
        "corner_nce": float(nce),
        "quad_iou": float(iou),
        "corner_peak1_min": float(min(item["peak1"] for item in corners)),
        "corner_peak1_mean": float(np.mean([item["peak1"] for item in corners])),
        "corner_peak2_max": float(max(item["peak2"] for item in corners)),
        "corner_peak2_mean": float(np.mean([item["peak2"] for item in corners])),
        "corner_peak_difference_min": float(min(item["peak_difference"] for item in corners)),
        "corner_peak_difference_mean": float(
            np.mean([item["peak_difference"] for item in corners])
        ),
        "corner_peak_ratio_min": float(min(item["peak_ratio"] for item in corners)),
        "corner_peak_ratio_mean": float(np.mean([item["peak_ratio"] for item in corners])),
        "corner_entropy_mean": float(np.mean([item["normalized_entropy"] for item in corners])),
        "corner_entropy_max": float(max(item["normalized_entropy"] for item in corners)),
        "corner_sharpness_min": float(min(item["local_sharpness"] for item in corners)),
        "corner_sharpness_mean": float(np.mean([item["local_sharpness"] for item in corners])),
        "corner_confidence_min": float(min(prediction["corner_confidences"])),
        "corner_confidence_mean": float(np.mean(prediction["corner_confidences"])),
        "candidate_margin": float(prediction["candidate_margin"]),
    }


def _aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [case for case in cases if case["present"]]
    candidates = [case for case in positive if case["has_candidate"]]
    nce = np.asarray([case["corner_nce"] for case in candidates], np.float64)
    iou = np.asarray([case["quad_iou"] for case in candidates], np.float64)
    distribution_names = (
        "corner_peak1_min",
        "corner_peak1_mean",
        "corner_peak2_max",
        "corner_peak2_mean",
        "corner_peak_difference_min",
        "corner_peak_difference_mean",
        "corner_peak_ratio_min",
        "corner_peak_ratio_mean",
        "corner_entropy_mean",
        "corner_entropy_max",
        "corner_sharpness_min",
        "corner_sharpness_mean",
        "corner_confidence_min",
        "corner_confidence_mean",
        "candidate_margin",
    )
    return {
        "sample_count": len(cases),
        "present_count": len(positive),
        "candidate_count": len(candidates),
        "corner_nce_median": float(np.median(nce)) if nce.size else 1.0,
        "corner_nce_p95": float(np.percentile(nce, 95)) if nce.size else 1.0,
        "quad_iou_median": float(np.median(iou)) if iou.size else 0.0,
        "quad_iou_p05": float(np.percentile(iou, 5)) if iou.size else 0.0,
        "strict_correct_rate": float(np.mean([case["strict_correct"] for case in positive]))
        if positive
        else 0.0,
        "semantic_strict_correct_rate": float(
            np.mean([case["semantic_strict_correct"] for case in positive])
        )
        if positive
        else 0.0,
        "heatmap_diagnostics": {
            name: _distribution([case[name] for case in cases]) for name in distribution_names
        },
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "minimum": float(array.min()) if array.size else 0.0,
        "p05": float(np.percentile(array, 5)) if array.size else 0.0,
        "median": float(np.median(array)) if array.size else 0.0,
        "mean": float(np.mean(array)) if array.size else 0.0,
        "p95": float(np.percentile(array, 95)) if array.size else 0.0,
        "maximum": float(array.max()) if array.size else 0.0,
    }


def _dataset_overlap(datasets: tuple[DatasetSpec, ...]) -> dict[str, int]:
    """明确报告 validation slices 的样本重叠，避免把嵌套子集当成独立证据。"""

    output = {}
    for index, first in enumerate(datasets):
        for second in datasets[index + 1 :]:
            output[f"{first.name}__{second.name}"] = len(set(first.paths) & set(second.paths))
    return output


def _markdown_report(reports: dict[str, dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# P4-G3.6 Geometry Trajectory Validation",
        "",
        "选择过程仅使用 validation/calibration；SmartDoc test 未被读取。",
        "",
        f"预先冻结的 absolute tolerance：`{summary['eligibility_tolerances_absolute']}`。",
        "",
    ]
    for dataset in summary["dataset_order"]:
        report = reports[dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| checkpoint | NCE median | NCE P95 | IoU median | IoU P05 | strict |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in report["checkpoints"]:
            metrics = item["metrics"]
            lines.append(
                f"| {item['label']} | {metrics['corner_nce_median']:.8f} | "
                f"{metrics['corner_nce_p95']:.8f} | {metrics['quad_iou_median']:.8f} | "
                f"{metrics['quad_iou_p05']:.8f} | {metrics['strict_correct_rate']:.8f} |"
            )
        lines.append("")
    selection = summary["selection"]
    lines.extend(
        [
            "## Selection",
            "",
            f"- status: `{selection['status']}`",
            f"- eligible: `{selection['eligible_checkpoints']}`",
            f"- Pareto front: `{selection['pareto_front']}`",
            f"- eligible Pareto front: `{selection['eligible_pareto_front']}`",
            f"- frozen winner: `{selection['winner']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前环境不支持 MPS")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不支持 CUDA")
    return torch.device(name)


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * done / max(1, total))
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>5}/{total:<5} {message}",
        end="\n" if done >= total else "\r",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
