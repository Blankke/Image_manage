#!/usr/bin/env python3
"""审计冻结 geometry 的 acceptance features、hard gate 与 calibrated evidence。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_p4_acceptance.py --mode validation \
      --checkpoint /runs/p4/trajectory/frozen-geometry.pt \
      --manifest /data/manifests/p2-public/calibration-public.geometry.jsonl \
      --dataset-root /data --device mps --output-directory /runs/p4/acceptance

    python scripts/audit_p4_acceptance.py --mode test \
      --checkpoint /runs/p4/trajectory/frozen-geometry.pt \
      --manifest /data/manifests/smartdoc.geometry.jsonl \
      --dataset-root /data --calibrator /runs/p4/acceptance/correctness-calibrator.json \
      --device mps --output-directory /runs/p4/frozen-smartdoc-test

validation 模式按 group 确定性拆分 fit/selection/evaluation，拟合 development-only logistic
calibrator。test 模式只消费已冻结校准器，禁止重新选择阈值或参数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from benchmarks.geometry_e2e.run import (  # noqa: E402
    _project_json_string,
    _resolve_manifest_image,
)
from scripts.audit_p4_geometry_parity import _prediction_from_raw  # noqa: E402
from training.quadlocator.correctness_calibrator import fit_calibrator  # noqa: E402
from training.quadlocator.model import (  # noqa: E402
    QuadLocatorS,
    load_quadlocator_state_dict,
)
from training.quadlocator.train import _assert_public_training_manifest  # noqa: E402

from screenrestore.geometry import ConfidencePolicy, CorrectnessCalibrator  # noqa: E402
from screenrestore.geometry.confidence import (  # noqa: E402
    CORRECTNESS_FEATURE_NAMES,
)
from screenrestore.geometry.decoder import CornerDecoderSpec, decode_corner_logits  # noqa: E402
from screenrestore.geometry.detector import _letterbox_tensor  # noqa: E402
from screenrestore.geometry.edge_refine import refine_quad_edges  # noqa: E402
from screenrestore.geometry.types import RejectionReason, quadrilateral_is_valid  # noqa: E402
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
STRICT_NCE_MAX = 0.01
STRICT_IOU_MIN = 0.93
MINIMUM_POLICY_PRECISION = 0.99


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--decoder",
        choices=(
            "decoder_v2",
            "coherent_v1",
            "coherent_class_hybrid_v1",
            "coherent_guarded_v1",
            "coherent_repair_v1",
        ),
        default="decoder_v2",
        help="公开验证对照使用的角点实例选择策略",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    if args.mode == "validation" and args.calibrator is not None:
        raise ValueError("validation 模式会自行拟合 calibrator，不能传入 --calibrator")
    if args.mode == "test" and args.calibrator is None:
        raise ValueError("test 模式必须传入 validation 冻结的 --calibrator")
    if args.mode == "validation":
        # acceptance 拟合属于模型选择，必须在创建输出目录前拒绝私人清单。
        _assert_public_training_manifest(args.manifest)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f"拒绝覆盖已有 acceptance audit：{output_directory}")
    output_directory.mkdir(parents=True)
    started = time.monotonic()
    split = "validation" if args.mode == "validation" else "test"
    root = args.dataset_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    paths = _scheduled_paths(manifest, root, split)
    checkpoint = args.checkpoint.expanduser().resolve()
    predictions = _infer_acceptance_features(
        checkpoint,
        paths,
        device=_device(args.device),
        batch_size=args.batch_size,
        decoder_name=args.decoder,
    )

    # 先冻结所有照片侧 features，再读取完整标注生成 correctness target。
    truths = _truth_by_path(manifest, root, split)
    rows = [_score_case(predictions[path], truths[path]) for path in paths]
    common = {
        "format_version": 1,
        "kind": "p4_acceptance_feature_audit",
        "mode": args.mode,
        "split": split,
        "protocol": "photo_only_features_then_frozen_gt",
        "decoder": args.decoder,
        "checkpoint": _file_identity(checkpoint),
        "manifest": _file_identity(manifest),
        "sample_count": len(rows),
        "independent_group_count": len({row["group_id"] for row in rows}),
        "strict_definition": {
            "positive_target": (
                "target class correct、content-layer quad，"
                f"NCE <= {STRICT_NCE_MAX}, IoU >= {STRICT_IOU_MIN}"
            ),
            "negative_target": "false；无目标样本不可作为可接受候选",
        },
        "geometry": _geometry_report(rows),
        "candidate_margin": _candidate_margin_report(rows),
        "nms_radius_sweep": _nms_radius_sweep_report(rows),
        "feature_associations": _feature_associations(rows),
        "boundary_audit": _boundary_report(rows),
        "current_hard_gate": _policy_metrics(rows, "hard_accepted"),
        "current_hard_without_boundary_gate": _policy_metrics(
            rows, "hard_without_boundary_accepted"
        ),
        "current_hard_without_margin_gate": _policy_metrics(rows, "hard_without_margin_accepted"),
        "current_hard_without_margin_and_boundary_gates": _policy_metrics(
            rows, "hard_without_margin_boundary_accepted"
        ),
    }
    if args.mode == "validation":
        partitioned = _partition_rows(rows)
        calibration = _fit_and_evaluate_calibrators(partitioned, _sha256(manifest))
        calibrator = calibration.pop("calibrator")
        calibrator_path = output_directory / "correctness-calibrator.json"
        _write_json(calibrator_path, calibrator.to_dict())
        common["group_partition"] = {
            name: {
                "sample_count": len(items),
                "independent_group_count": len({item["group_id"] for item in items}),
                "strict_correct_count": sum(item["strict_correct"] for item in items),
            }
            for name, items in partitioned.items()
        }
        common["calibrated_evidence"] = calibration
        common["calibrator"] = _file_identity(calibrator_path)
        common["policy_comparison_evaluation"] = {
            "current_hard_gate": calibration["hard_gate_evaluation"],
            "calibrated_evidence": calibration["full"]["evaluation"],
        }
        common["test_data_used"] = False
        summary_path = output_directory / "acceptance-summary.json"
    else:
        assert args.calibrator is not None
        calibrator_path = args.calibrator.expanduser().resolve()
        calibrator = CorrectnessCalibrator.load(calibrator_path)
        for row in rows:
            row["calibrated_probability"] = calibrator.predict_probability(row["features"])
            row["calibrated_accepted"] = (
                row["has_candidate"]
                and row["area_in_scope"]
                and row["calibrated_probability"] >= calibrator.threshold
            )
        common["calibrator"] = _file_identity(calibrator_path)
        common["calibrated_evidence"] = {
            "threshold": calibrator.threshold,
            "policy": _policy_metrics(rows, "calibrated_accepted"),
            "probability_distribution": _distribution(
                [row["calibrated_probability"] for row in rows]
            ),
        }
        common["policy_comparison_evaluation"] = {
            "current_hard_gate": common["current_hard_gate"],
            "calibrated_evidence": common["calibrated_evidence"]["policy"],
        }
        common["selection_or_fitting_performed"] = False
        summary_path = output_directory / "frozen-test-summary.json"
    common["rejection_reason_counts"] = dict(
        sorted(Counter(reason for row in rows for reason in row["hard_reasons"]).items())
    )
    common["git"] = _git_metadata()
    common["command"] = [sys.executable, *sys.argv]
    common["wall_time_seconds"] = round(time.monotonic() - started, 4)
    _write_json(summary_path, common)
    _write_feature_rows(output_directory / "acceptance-features.jsonl", rows, split)
    (output_directory / "acceptance-report.md").write_text(
        _markdown_report(common), encoding="utf-8"
    )
    print(summary_path)
    return 0


def _scheduled_paths(manifest: Path, root: Path, split: str) -> tuple[Path, ...]:
    paths = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record_split = _project_json_string(line, "split")
        if record_split != split:
            continue
        image = _project_json_string(line, "image")
        if image is None:
            raise ValueError(f"manifest 第 {line_number} 行缺少 image")
        path = _resolve_manifest_image(root, image)
        if not path.is_file():
            raise ValueError(f"图片不存在：{path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"manifest 没有 split={split} 的样本")
    if len(set(paths)) != len(paths):
        raise ValueError("同一 split 存在重复 image")
    return tuple(paths)


def _infer_acceptance_features(
    checkpoint_path: Path,
    paths: tuple[Path, ...],
    *,
    device: torch.device,
    batch_size: int,
    decoder_name: str = "decoder_v2",
) -> dict[Path, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    image_size = int(checkpoint["image_size"])
    model = QuadLocatorS(float(checkpoint["width_multiplier"]))
    load_quadlocator_state_dict(model, checkpoint["state_dict"])
    model.to(device).eval()
    policy = ConfidencePolicy()
    output = {}
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        documents = [load_image(path) for path in batch_paths]
        prepared = [_letterbox_tensor(document.original_rgb, image_size) for document in documents]
        tensor = np.concatenate([item[0] for item in prepared], axis=0)
        with torch.inference_mode():
            values = model(torch.from_numpy(tensor).to(device))
        raw_batch = {name: values[name].detach().cpu().numpy() for name in OUTPUT_NAMES}
        for local_index, (path, document, prepared_item) in enumerate(
            zip(batch_paths, documents, prepared, strict=True)
        ):
            raw = {name: raw_batch[name][local_index : local_index + 1] for name in OUTPUT_NAMES}
            peak_radius_profile = _peak_radius_profile(raw["content_corner_heatmaps"])
            prediction = _prediction_from_raw(
                raw, prepared_item[1], document.original_rgb.shape, decoder_name
            )
            shape = document.original_rgb.shape
            scale = np.array([max(1, shape[1] - 1), max(1, shape[0] - 1)], np.float32)
            coarse = prediction.content_quad
            refinement = None
            hard_reasons: list[str]
            features: dict[str, float] | None = None
            hard_confidence = 0.0
            if coarse is None:
                hard_reasons = [RejectionReason.NO_CANDIDATE.value]
            elif not quadrilateral_is_valid(coarse, shape):
                hard_reasons = [RejectionReason.INVALID_QUAD.value]
            else:
                refinement = refine_quad_edges(
                    document.original_rgb, coarse, prediction.boundary_map
                )
                hard_confidence, reasons, diagnostics = policy.assess(prediction, refinement, shape)
                hard_reasons = [reason.value for reason in reasons]
                features = {name: float(diagnostics[name]) for name in CORRECTNESS_FEATURE_NAMES}
            attempted = refinement.attempted_corners if refinement is not None else None
            rollback = refinement.corners if refinement is not None else coarse
            area_ratio = (
                abs(float(cv2.contourArea(rollback))) / max(1.0, float(shape[0] * shape[1]))
                if rollback is not None
                else 0.0
            )
            corner_items = prediction.decoder_diagnostics.get("content", {}).get("corners", [])
            weakest_corner = min(
                corner_items,
                key=lambda item: float(item["peak_difference"]),
                default=None,
            )
            output[path] = {
                # NCE 的权威口径使用目标包围盒对角线，因此保留像素坐标参与评分。
                "coarse": coarse,
                "refined_attempt": attempted,
                "rollback": rollback,
                "image_scale": scale,
                "has_candidate": coarse is not None,
                "target_class": prediction.target_class.value,
                "presence": float(prediction.presence_confidence),
                "class_confidence": float(prediction.class_confidence),
                "layer": float(prediction.layer_confidence),
                "corner_peak_min": float(min(prediction.corner_confidences)),
                "corner_peak_mean": float(np.mean(prediction.corner_confidences)),
                "peak_difference_min": float(
                    min((item["peak_difference"] for item in corner_items), default=0.0)
                ),
                "peak_ratio_min": float(
                    min((item["peak_ratio"] for item in corner_items), default=1.0)
                ),
                "peak_distance_at_weakest": float(
                    weakest_corner["peak_distance"] if weakest_corner is not None else 0.0
                ),
                "entropy_mean": float(
                    np.mean([item["normalized_entropy"] for item in corner_items])
                )
                if corner_items
                else 1.0,
                "entropy_max": float(
                    max((item["normalized_entropy"] for item in corner_items), default=1.0)
                ),
                "sharpness_min": float(
                    min((item["local_sharpness"] for item in corner_items), default=0.0)
                ),
                "candidate_margin": float(
                    prediction.candidates[0].scores.get("candidate_margin", 0.0)
                    if prediction.candidates
                    else 0.0
                ),
                "peak_radius_profile": peak_radius_profile,
                "features": features,
                "hard_confidence": float(hard_confidence),
                "hard_reasons": hard_reasons,
                "hard_accepted": not hard_reasons,
                "hard_without_boundary_accepted": not [
                    reason
                    for reason in hard_reasons
                    if reason != RejectionReason.BOUNDARY_UNCERTAIN.value
                ],
                "hard_without_margin_accepted": not [
                    reason
                    for reason in hard_reasons
                    if reason != RejectionReason.SCORE_AMBIGUOUS.value
                ],
                "hard_without_margin_boundary_accepted": not [
                    reason
                    for reason in hard_reasons
                    if reason
                    not in {
                        RejectionReason.SCORE_AMBIGUOUS.value,
                        RejectionReason.BOUNDARY_UNCERTAIN.value,
                    }
                ],
                "boundary_gate_pass": bool(
                    refinement is not None
                    and refinement.accepted
                    and refinement.mean_support >= policy.min_boundary_support
                ),
                "refinement_accepted": bool(refinement and refinement.accepted),
                "refinement_reason": refinement.reason if refinement else "no_candidate",
                "refinement_max_shift": float(max(refinement.corner_shifts))
                if refinement
                else 0.0,
                "boundary_support": float(refinement.mean_support) if refinement else 0.0,
                "residual_median": float(max(refinement.residual_median)) if refinement else 999.0,
                "residual_p95": float(max(refinement.residual_p95)) if refinement else 999.0,
                "continuous_coverage": float(min(refinement.continuous_coverage))
                if refinement
                else 0.0,
                "normal_alignment": float(min(refinement.gradient_normal_alignment))
                if refinement
                else 0.0,
                "mask_consistency": float(features["mask_consistency"])
                if features is not None
                else 0.0,
                "area_ratio": area_ratio,
                "area_in_scope": policy.min_area_ratio <= area_ratio <= policy.max_area_ratio,
            }
        _progress(min(start + len(batch_paths), len(paths)), len(paths), "acceptance features")
    return output


def _truth_by_path(manifest: Path, root: Path, split: str) -> dict[Path, dict[str, Any]]:
    output = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # 先只读取分区字段；validation audit 不反序列化 test 行中的任何真值。
        if _project_json_string(line, "split") != split:
            continue
        record = json.loads(line)
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
            "source": str(record.get("source", "unknown")),
            "difficulty": str(record.get("difficulty", "unknown")),
            "hard_taxonomy": str(record.get("hard_taxonomy", record.get("scene_type", "unknown"))),
            "in_scope": bool(record.get("in_scope", record.get("visible", True)))
            and bool(record["present"]),
        }
    return output


def _score_case(prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    stages = {
        name: _stage_metrics(prediction[name], truth["content_quad"], prediction["image_scale"])
        for name in ("coarse", "refined_attempt", "rollback")
    }
    class_correct = prediction["target_class"] == truth["target_class"]
    strict = bool(
        truth["present"]
        and class_correct
        and stages["rollback"]["has_quad"]
        and stages["rollback"]["corner_nce"] <= STRICT_NCE_MAX
        and stages["rollback"]["quad_iou"] >= STRICT_IOU_MIN
    )
    if prediction["features"] is None:
        # 无候选不会进入 calibrator；保留固定长度零值只用于 JSONL 诊断。
        features = {name: 0.0 for name in CORRECTNESS_FEATURE_NAMES}
        complete_features = False
    else:
        features = prediction["features"]
        complete_features = True
    row = {
        **prediction,
        **truth,
        "predicted_target_class": prediction["target_class"],
        "target_class": truth["target_class"],
        "features": features,
        "complete_features": complete_features,
        "class_correct": class_correct,
        "strict_correct": strict,
        "stages": stages,
    }
    return row


def _stage_metrics(
    quad: np.ndarray | None,
    truth: np.ndarray | None,
    image_scale: np.ndarray,
) -> dict[str, Any]:
    if quad is None or truth is None:
        return {"has_quad": quad is not None, "corner_nce": 1.0, "quad_iou": 0.0}
    nce, iou, _maximum = corner_metrics(quad, truth * image_scale)
    return {"has_quad": True, "corner_nce": float(nce), "quad_iou": float(iou)}


def _geometry_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["present"]]
    output = {}
    for stage in ("coarse", "refined_attempt", "rollback"):
        values = [row["stages"][stage] for row in positive if row["stages"][stage]["has_quad"]]
        nce = [item["corner_nce"] for item in values]
        iou = [item["quad_iou"] for item in values]
        output[stage] = {
            "candidate_count": len(values),
            "corner_nce_median": _percentile(nce, 50, 1.0),
            "corner_nce_p95": _percentile(nce, 95, 1.0),
            "quad_iou_median": _percentile(iou, 50, 0.0),
            "quad_iou_p05": _percentile(iou, 5, 0.0),
            "strict_geometry_rate": float(
                np.mean(
                    [
                        item["corner_nce"] <= STRICT_NCE_MAX and item["quad_iou"] >= STRICT_IOU_MIN
                        for item in values
                    ]
                )
            )
            if values
            else 0.0,
        }
    output["final"] = {
        **output["rollback"],
        "definition": "refinement 通过时采用 refined，否则保持 coarse；acceptance 只决定是否自动放行",
    }
    return output


def _candidate_margin_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["complete_features"]]
    if not usable:
        raise ValueError("没有具备完整 acceptance features 的候选")
    thresholds = sorted(
        set(
            [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
            + np.quantile(
                np.asarray([row["candidate_margin"] for row in usable], np.float64),
                np.linspace(0.0, 1.0, 41),
            ).tolist()
        )
    )
    curve = [_threshold_metrics(usable, threshold) for threshold in thresholds]
    fixed = next(item for item in curve if abs(item["threshold"] - 0.06) < 1e-12)
    labels = np.asarray([row["strict_correct"] for row in usable], np.float64)
    margin = np.asarray([row["candidate_margin"] for row in usable], np.float64)
    return {
        "definition": "min over four corners of (peak1 - peak2)",
        "hard_threshold": 0.06,
        "threshold_source": "ConfidencePolicy 保守默认值；仓库中无独立 validation calibration 记录",
        "distribution": _distribution(margin.tolist()),
        "point_biserial_pearson": _pearson(margin, labels),
        "spearman": _spearman(margin, labels),
        "average_precision": _average_precision(margin, labels.astype(bool)),
        "weakest_peak_distance_distribution": _distribution(
            [float(row.get("peak_distance_at_weakest", 0.0)) for row in usable]
        ),
        "low_margin_distance_buckets": _low_margin_distance_buckets(usable),
        "at_0_06": fixed,
        "precision_recall_risk_coverage_curve": curve,
    }


def _peak_radius_profile(heatmaps: np.ndarray) -> dict[str, dict[str, float]]:
    """一次前向结果上重算多个 NMS 半径，避免为诊断重复运行模型。"""

    profile: dict[str, dict[str, float]] = {}
    for radius in (3, 4, 5, 6, 8, 12):
        decoded = decode_corner_logits(heatmaps, CornerDecoderSpec(nms_radius=radius))
        weakest = min(decoded.diagnostics, key=lambda item: item.peak_difference)
        profile[str(radius)] = {
            "candidate_margin": float(weakest.peak_difference),
            "peak_distance_at_weakest": float(weakest.peak_distance),
        }
    return profile


def _nms_radius_sweep_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """比较 NMS 半径的公开 correctness 识别力；这里只诊断，不修改正式策略。"""

    output: dict[str, Any] = {
        "definition": "recompute second-mode margin from the same frozen heatmaps",
        "hard_threshold": 0.06,
    }
    for radius in (3, 4, 5, 6, 8, 12):
        key = str(radius)
        usable = [row for row in rows if row["has_candidate"] and key in row["peak_radius_profile"]]
        margins = np.asarray(
            [row["peak_radius_profile"][key]["candidate_margin"] for row in usable],
            dtype=np.float64,
        )
        accepted = margins >= 0.06
        correct = np.asarray([row["strict_correct"] for row in usable], dtype=bool)
        true_positive = int(np.count_nonzero(accepted & correct))
        false_positive = int(np.count_nonzero(accepted & ~correct))
        strict_count = int(np.count_nonzero(correct))
        output[key] = {
            "sample_count": len(usable),
            "margin_distribution": _distribution(margins.tolist()),
            "accepted_count": int(np.count_nonzero(accepted)),
            "precision": true_positive / max(1, true_positive + false_positive),
            "strict_recall": true_positive / max(1, strict_count),
            "true_positive": true_positive,
            "false_positive": false_positive,
        }
    return output


def _low_margin_distance_buckets(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """区分局部肩峰与远处第二候选；这里只审计，不改变产品拒绝策略。"""

    low_margin = [row for row in rows if float(row["candidate_margin"]) < 0.06]
    buckets = {
        "local_le_4": [
            row for row in low_margin if float(row.get("peak_distance_at_weakest", 0.0)) <= 4.0
        ],
        "near_4_to_8": [
            row
            for row in low_margin
            if 4.0 < float(row.get("peak_distance_at_weakest", 0.0)) <= 8.0
        ],
        "distinct_gt_8": [
            row for row in low_margin if float(row.get("peak_distance_at_weakest", 0.0)) > 8.0
        ],
    }
    return {
        name: {
            "sample_count": len(items),
            "strict_correct_count": sum(bool(item["strict_correct"]) for item in items),
            "strict_correct_rate": (
                float(np.mean([bool(item["strict_correct"]) for item in items]))
                if items
                else 0.0
            ),
        }
        for name, items in buckets.items()
    }


def _feature_associations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["complete_features"]]
    labels = np.asarray([row["strict_correct"] for row in usable], bool)
    directions = {
        "candidate_margin": 1.0,
        "peak_ratio_min": 1.0,
        "entropy_mean": -1.0,
        "entropy_max": -1.0,
        "sharpness_min": 1.0,
        "boundary_support": 1.0,
        "continuous_coverage": 1.0,
        "normal_alignment": 1.0,
        "mask_consistency": 1.0,
        "presence": 1.0,
        "class_confidence": 1.0,
        "layer": 1.0,
    }
    output = {}
    for name, direction in directions.items():
        values = np.asarray([row[name] for row in usable], np.float64)
        oriented = values * direction
        output[name] = {
            "direction": "higher_is_safer" if direction > 0 else "lower_is_safer",
            "pearson": _pearson(oriented, labels.astype(np.float64)),
            "spearman": _spearman(oriented, labels.astype(np.float64)),
            "average_precision": _average_precision(oriented, labels),
            "correct_distribution": _distribution(values[labels].tolist()),
            "incorrect_distribution": _distribution(values[~labels].tolist()),
        }
    return output


def _boundary_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["present"]]
    outcomes = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in positive:
        coarse = row["stages"]["coarse"]
        attempt = row["stages"]["refined_attempt"]
        if not attempt["has_quad"]:
            outcome = "not_fitted"
        elif (
            attempt["corner_nce"] < coarse["corner_nce"] - 1e-5
            and attempt["quad_iou"] > coarse["quad_iou"] + 1e-5
        ):
            outcome = "improved"
        elif (
            attempt["corner_nce"] > coarse["corner_nce"] + 1e-5
            or attempt["quad_iou"] < coarse["quad_iou"] - 1e-5
        ):
            outcome = "worsened"
        else:
            outcome = "neutral"
        outcomes[outcome] += 1
        by_source[row["source"]][outcome] += 1
    return {
        "refinement_accepted_count": sum(row["refinement_accepted"] for row in rows),
        "refinement_accepted_rate": float(np.mean([row["refinement_accepted"] for row in rows])),
        "boundary_gate_pass_count": sum(row["boundary_gate_pass"] for row in rows),
        "boundary_gate_pass_rate": float(np.mean([row["boundary_gate_pass"] for row in rows])),
        "attempt_outcomes": dict(sorted(outcomes.items())),
        "attempt_outcomes_by_source": {
            source: dict(sorted(counts.items())) for source, counts in sorted(by_source.items())
        },
        "support_distribution": _distribution([row["boundary_support"] for row in rows]),
        "residual_p95_distribution": _distribution([row["residual_p95"] for row in rows]),
        "continuous_coverage_distribution": _distribution(
            [row["continuous_coverage"] for row in rows]
        ),
        "normal_alignment_distribution": _distribution([row["normal_alignment"] for row in rows]),
    }


def _partition_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {"fit": [], "selection": [], "evaluation": []}
    assignments = {}
    for row in rows:
        group_id = row["group_id"]
        if group_id not in assignments:
            bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 10
            assignments[group_id] = (
                "fit" if bucket < 6 else "selection" if bucket < 8 else "evaluation"
            )
        output[assignments[group_id]].append(row)
    if any(not values for values in output.values()):
        raise ValueError("group partition 产生空子集，无法进行独立 acceptance calibration")
    return output


def _fit_and_evaluate_calibrators(
    partitioned: dict[str, list[dict[str, Any]]], manifest_sha256: str
) -> dict[str, Any]:
    feature_sets = {
        "full": tuple(CORRECTNESS_FEATURE_NAMES),
        "without_candidate_margin": tuple(
            name for name in CORRECTNESS_FEATURE_NAMES if name != "corner_peak_difference_min"
        ),
        "without_peak_ratio": tuple(
            name for name in CORRECTNESS_FEATURE_NAMES if name != "corner_peak_ratio_min"
        ),
        "without_entropy_sharpness": tuple(
            name
            for name in CORRECTNESS_FEATURE_NAMES
            if name not in {"corner_entropy_mean", "corner_entropy_max", "corner_sharpness_min"}
        ),
        "without_boundary_refinement": tuple(
            name
            for name in CORRECTNESS_FEATURE_NAMES
            if name
            not in {
                "boundary_support_min",
                "boundary_support_mean",
                "line_residual_median_max",
                "line_residual_p95_max",
                "continuous_coverage_min",
                "gradient_normal_alignment_min",
                "boundary_consistency_mean",
                "refine_displacement_mean",
                "refine_displacement_max",
                "area_drift",
                "aspect_drift",
            }
        ),
    }
    results = {}
    full_calibrator = None
    for name, feature_names in feature_sets.items():
        fit_rows = [row for row in partitioned["fit"] if row["complete_features"]]
        # policy 比较必须使用完整且相同的 group evaluation partition。无候选行保留零值
        # diagnostics，并由 has_candidate 结构门拒绝；不能从 calibrated 分母中静默删除。
        selection_rows = list(partitioned["selection"])
        evaluation_rows = list(partitioned["evaluation"])
        calibrator = fit_calibrator(
            [_feature_subset(row["features"], feature_names) for row in fit_rows],
            [row["strict_correct"] for row in fit_rows],
            manifest_sha256=manifest_sha256,
            minimum_precision=MINIMUM_POLICY_PRECISION,
        )
        selection_probabilities = np.asarray(
            [calibrator.predict_probability(row["features"]) for row in selection_rows],
            np.float64,
        )
        selection_labels = np.asarray([row["strict_correct"] for row in selection_rows], bool)
        selection_eligible = np.asarray(
            [row["has_candidate"] and row["area_in_scope"] for row in selection_rows], bool
        )
        threshold = _select_threshold(
            selection_probabilities,
            selection_labels,
            MINIMUM_POLICY_PRECISION,
            eligible=selection_eligible,
        )
        calibrator = CorrectnessCalibrator(
            feature_names=calibrator.feature_names,
            means=calibrator.means,
            scales=calibrator.scales,
            coefficients=calibrator.coefficients,
            intercept=calibrator.intercept,
            threshold=threshold,
            manifest_sha256=calibrator.manifest_sha256,
        )
        for row in evaluation_rows:
            row[f"calibrated_{name}_probability"] = calibrator.predict_probability(row["features"])
            row[f"calibrated_{name}_accepted"] = (
                row["has_candidate"]
                and row["area_in_scope"]
                and row[f"calibrated_{name}_probability"] >= threshold
            )
        results[name] = {
            "feature_names": list(feature_names),
            "threshold": threshold,
            "fit_strict_correct_count": sum(row["strict_correct"] for row in fit_rows),
            "selection": _probability_policy_metrics(
                selection_rows, selection_probabilities, threshold, eligible=selection_eligible
            ),
            "evaluation": _policy_metrics(evaluation_rows, f"calibrated_{name}_accepted"),
            "evaluation_average_precision": _average_precision(
                np.asarray([row[f"calibrated_{name}_probability"] for row in evaluation_rows]),
                np.asarray([row["strict_correct"] for row in evaluation_rows], bool),
            ),
        }
        if name == "full":
            full_calibrator = calibrator
    assert full_calibrator is not None
    evaluation = partitioned["evaluation"]
    results["hard_gate_evaluation"] = _policy_metrics(evaluation, "hard_accepted")
    results["hard_without_boundary_evaluation"] = _policy_metrics(
        evaluation, "hard_without_boundary_accepted"
    )
    results["hard_without_margin_evaluation"] = _policy_metrics(
        evaluation, "hard_without_margin_accepted"
    )
    results["hard_without_margin_boundary_evaluation"] = _policy_metrics(
        evaluation, "hard_without_margin_boundary_accepted"
    )
    results["minimum_precision_target"] = MINIMUM_POLICY_PRECISION
    results["calibrator"] = full_calibrator
    return results


def _feature_subset(features: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    return {name: features[name] for name in names}


def _select_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    minimum_precision: float,
    *,
    eligible: np.ndarray | None = None,
) -> float:
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities 与 labels 形状必须一致")
    allowed = np.ones(probabilities.shape, bool) if eligible is None else np.asarray(eligible, bool)
    if allowed.shape != probabilities.shape:
        raise ValueError("eligible 与 probabilities 形状必须一致")
    candidates = np.unique(np.clip(probabilities[allowed], 1e-6, 1.0 - 1e-6))[::-1]
    # 没有任何非零 coverage 阈值满足 precision 下限时，冻结一个高于 sigmoid 最大值的
    # 安全阈值，使 validation 和后续 test 都保持零接受，不能靠未验证的 test 高分放行。
    selected = float(np.nextafter(1.0, 0.0))
    best_count = 0
    for threshold in candidates:
        accepted = allowed & (probabilities >= threshold)
        count = int(accepted.sum())
        precision = float(labels[accepted].mean()) if count else 1.0
        if precision >= minimum_precision and count > best_count:
            selected = float(threshold)
            best_count = count
    return selected


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    accepted = np.asarray([row["candidate_margin"] >= threshold for row in rows], bool)
    return {"threshold": float(threshold), **_binary_policy_metrics(rows, accepted)}


def _probability_policy_metrics(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    *,
    eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    allowed = np.ones(probabilities.shape, bool) if eligible is None else np.asarray(eligible, bool)
    accepted = allowed & (probabilities >= threshold)
    return {
        "threshold": threshold,
        **_binary_policy_metrics(rows, accepted),
        "average_precision": _average_precision(
            probabilities, np.asarray([row["strict_correct"] for row in rows], bool)
        ),
    }


def _policy_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _binary_policy_metrics(rows, np.asarray([bool(row.get(field, False)) for row in rows]))


def _binary_policy_metrics(rows: list[dict[str, Any]], accepted: np.ndarray) -> dict[str, Any]:
    accepted = np.asarray(accepted, dtype=bool)
    if accepted.shape != (len(rows),):
        raise ValueError("accepted 必须与完整 policy 样本一一对应")
    labels = np.asarray([row["strict_correct"] for row in rows], bool)
    in_scope = np.asarray([row["in_scope"] for row in rows], bool)
    true_positive = int(np.sum(accepted & labels))
    false_positive = int(np.sum(accepted & ~labels))
    false_negative = int(np.sum(~accepted & labels))
    accepted_count = int(accepted.sum())
    return {
        "sample_count": len(rows),
        "accepted_count": accepted_count,
        "precision": true_positive / max(1, accepted_count),
        "recall": true_positive / max(1, int(labels.sum())),
        "coverage": int(np.sum(accepted & in_scope)) / max(1, int(in_scope.sum())),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": int(np.sum(~accepted & ~labels)),
        "in_scope_count": int(in_scope.sum()),
        "strict_correct_count": int(labels.sum()),
    }


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def _pearson(values: np.ndarray, labels: np.ndarray) -> float:
    if values.size < 2 or np.std(values) < 1e-12 or np.std(labels) < 1e-12:
        return 0.0
    return float(np.corrcoef(values, labels)[0, 1])


def _spearman(values: np.ndarray, labels: np.ndarray) -> float:
    if values.size < 2 or np.std(values) < 1e-12 or np.std(labels) < 1e-12:
        return 0.0
    result = spearmanr(values, labels)
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    if not array.size:
        return {
            "count": 0,
            "minimum": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _percentile(values: list[float], percentile: float, default: float) -> float:
    return float(np.percentile(values, percentile)) if values else default


def _write_feature_rows(path: Path, rows: list[dict[str, Any]], split: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = {
                "id": index,
                "split": split,
                "group_id_sha256": hashlib.sha256(row["group_id"].encode("utf-8")).hexdigest(),
                "group_partition": _group_partition(row["group_id"]),
                "strict_correct": row["strict_correct"],
                # 冻结每例硬门结果和拒绝原因，才能按场景定位高置信误接受。
                "hard_accepted": row["hard_accepted"],
                "hard_reasons": row["hard_reasons"],
                "nce": row["stages"]["rollback"]["corner_nce"],
                "iou": row["stages"]["rollback"]["quad_iou"],
                "coarse_nce": row["stages"]["coarse"]["corner_nce"],
                "coarse_iou": row["stages"]["coarse"]["quad_iou"],
                "target_class": row["target_class"],
                "predicted_target_class": row["predicted_target_class"],
                "source": row["source"],
                "difficulty": row["difficulty"],
                "hard_taxonomy": row["hard_taxonomy"],
                "presence": row["presence"],
                "class_confidence": row["class_confidence"],
                "layer_confidence": row["layer"],
                "corner_peak_min": row["corner_peak_min"],
                "corner_peak_mean": row["corner_peak_mean"],
                "peak_difference_min": row["peak_difference_min"],
                "peak_ratio_min": row["peak_ratio_min"],
                "peak_distance_at_weakest": row["peak_distance_at_weakest"],
                "entropy_mean": row["entropy_mean"],
                "entropy_max": row["entropy_max"],
                "sharpness_min": row["sharpness_min"],
                "candidate_margin": row["candidate_margin"],
                "peak_radius_profile": row["peak_radius_profile"],
                "boundary_support": row["boundary_support"],
                "refinement_accepted": row["refinement_accepted"],
                "refinement_max_shift": row["refinement_max_shift"],
                "residual_median": row["residual_median"],
                "residual_p95": row["residual_p95"],
                "continuous_coverage": row["continuous_coverage"],
                "normal_alignment": row["normal_alignment"],
                "mask_consistency": row["mask_consistency"],
                "area_ratio": row["area_ratio"],
                "features": row["features"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _group_partition(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "fit" if bucket < 6 else "selection" if bucket < 8 else "evaluation"


def _markdown_report(report: dict[str, Any]) -> str:
    hard = report["policy_comparison_evaluation"]["current_hard_gate"]
    margin = report["candidate_margin"]
    lines = [
        "# P4-G3.6 Acceptance Feature Audit",
        "",
        f"- mode: `{report['mode']}`",
        f"- samples/groups: `{report['sample_count']}/{report['independent_group_count']}`",
        f"- candidate_margin Pearson/Spearman: `{margin['point_biserial_pearson']:.6f}` / `{margin['spearman']:.6f}`",
        f"- candidate_margin AP: `{margin['average_precision']:.6f}`",
        f"- hard gate precision/coverage: `{hard['precision']:.6f}` / `{hard['coverage']:.6f}`",
        "",
        "## Geometry stages",
        "",
        "| stage | NCE median | NCE P95 | IoU median | IoU P05 | strict geometry |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, metrics in report["geometry"].items():
        lines.append(
            f"| {stage} | {metrics['corner_nce_median']:.8f} | {metrics['corner_nce_p95']:.8f} | "
            f"{metrics['quad_iou_median']:.8f} | {metrics['quad_iou_p05']:.8f} | "
            f"{metrics['strict_geometry_rate']:.8f} |"
        )
    calibrated = report["policy_comparison_evaluation"]["calibrated_evidence"]
    lines.extend(
        [
            "",
            "## Policy comparison",
            "",
            "| policy | accepted | precision | recall | coverage | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| current hard gate | {hard['accepted_count']} | {hard['precision']:.8f} | {hard['recall']:.8f} | {hard['coverage']:.8f} | {hard['false_positive']} | {hard['false_negative']} |",
            f"| calibrated evidence | {calibrated['accepted_count']} | {calibrated['precision']:.8f} | {calibrated['recall']:.8f} | {calibrated['coverage']:.8f} | {calibrated['false_positive']} | {calibrated['false_negative']} |",
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
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
