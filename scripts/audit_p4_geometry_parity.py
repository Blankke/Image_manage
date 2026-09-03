#!/usr/bin/env python3
"""审计 B0 在旧/新 decoder、精修、最终策略及 Torch/ONNX 间的评价口径。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_p4_geometry_parity.py \
      --checkpoint /runs/p2/stage-b/best.pt \
      --onnx /runs/p2/stage-b/quadlocator-s.onnx \
      --manifest /data/manifests/smartdoc.geometry.jsonl \
      --dataset-root /data \
      --data-directory /data/geometry/smartdoc/frames \
      --split test --device mps --save-raw-heatmaps \
      --output-directory /runs/p4/b0-parity

脚本先完成 photo-only 推理，再读取完整标注打分。raw heatmap 保存为 NumPy memmap，
报告只记录有限统计与摘要，不写入图片内容。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# 允许按文件直接执行，确保 benchmarks/training 使用当前工作树实现。
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from benchmarks.geometry_e2e.run import (  # noqa: E402
    _inference_photo_paths,
    _read_manifest,
    _resolve_manifest_image,
    _truth_from_manifest_record,
)
from training.quadlocator.model import QuadLocatorS  # noqa: E402

from screenrestore.geometry import AutomaticGeometryService, ConfidencePolicy  # noqa: E402
from screenrestore.geometry.decoder import (  # noqa: E402
    CornerDecoderSpec,
    CornerPeakDiagnostics,
    decode_corner_logits,
)
from screenrestore.geometry.detector import (  # noqa: E402
    OnnxQuadDetector,
    _layer_confidence,
    _letterbox_tensor,
    _sigmoid,
    _softmax,
)
from screenrestore.geometry.edge_refine import refine_quad_edges  # noqa: E402
from screenrestore.geometry.rectify import order_corners  # noqa: E402
from screenrestore.geometry.types import (  # noqa: E402
    QuadPrediction,
    QuadrilateralCandidate,
    TargetClass,
    TargetLayer,
)
from screenrestore.io.image_loader import load_image  # noqa: E402
from screenrestore.validation import (  # noqa: E402
    aggregate_geometry_results,
    evaluate_geometry_decision,
)
from screenrestore.validation.geometry_benchmark import corner_metrics  # noqa: E402

OUTPUT_NAMES = OnnxQuadDetector.DEFAULT_OUTPUTS
CLASS_ORDER = OnnxQuadDetector.CLASS_ORDER


@dataclass(slots=True)
class _ParityAccumulator:
    maximum: float = 0.0
    absolute_sum: float = 0.0
    value_count: int = 0

    def update(self, first: np.ndarray, second: np.ndarray) -> None:
        difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
        self.maximum = max(self.maximum, float(difference.max(initial=0.0)))
        self.absolute_sum += float(difference.sum())
        self.value_count += int(difference.size)

    def report(self) -> dict[str, float | int]:
        return {
            "max_abs": self.maximum,
            "mean_abs": self.absolute_sum / max(1, self.value_count),
            "value_count": self.value_count,
        }


class _StaticDetector:
    """让统一 localizer 消费一次冻结的 raw prediction。"""

    def __init__(self, prediction: QuadPrediction) -> None:
        self.prediction = prediction

    def predict(self, _image: np.ndarray, _hint: TargetClass | None = None) -> QuadPrediction:
        return self.prediction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-raw-heatmaps", action="store_true")
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_samples < 0:
        raise ValueError("batch-size 必须大于 0，max-samples 不能为负数")
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f"拒绝覆盖已有 parity 目录：{output_directory}")
    output_directory.mkdir(parents=True)
    started = time.monotonic()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    image_size = int(checkpoint["image_size"])
    model = QuadLocatorS(float(checkpoint["width_multiplier"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = _torch_device(args.device)
    model.to(device).eval()

    import onnxruntime as ort

    if hasattr(ort, "disable_telemetry_events"):
        ort.disable_telemetry_events()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    available = {output.name for output in session.get_outputs()}
    if set(OUTPUT_NAMES) - available:
        raise RuntimeError("B0 ONNX 不符合 7-output QuadLocator 契约")

    photos = _inference_photo_paths(
        args.manifest,
        args.dataset_root,
        args.data_directory,
        args.split,
    )
    if args.max_samples:
        photos = photos[: args.max_samples]
    if not photos:
        raise ValueError("没有可审计图片")

    parity = {name: _ParityAccumulator() for name in OUTPUT_NAMES}
    predictions: list[dict[str, Any]] = []
    content_store: np.memmap | None = None
    outer_store: np.memmap | None = None
    for start in range(0, len(photos), args.batch_size):
        batch_paths = photos[start : start + args.batch_size]
        documents = [load_image(path) for path in batch_paths]
        prepared = [_letterbox_tensor(document.original_rgb, image_size) for document in documents]
        batch_tensor = np.concatenate([item[0] for item in prepared], axis=0)
        raw_onnx_list = session.run(list(OUTPUT_NAMES), {input_name: batch_tensor})
        raw_onnx = dict(zip(OUTPUT_NAMES, raw_onnx_list, strict=True))
        with torch.inference_mode():
            torch_outputs = model(torch.from_numpy(batch_tensor).to(device))
        raw_torch = {name: value.detach().cpu().numpy() for name, value in torch_outputs.items()}
        for name in OUTPUT_NAMES:
            parity[name].update(raw_onnx[name], raw_torch[name])

        if args.save_raw_heatmaps and content_store is None:
            heatmap_shape = tuple(
                int(value) for value in raw_onnx["content_corner_heatmaps"].shape[1:]
            )
            content_store = np.lib.format.open_memmap(
                output_directory / "raw-content-heatmaps.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(photos), *heatmap_shape),
            )
            outer_store = np.lib.format.open_memmap(
                output_directory / "raw-outer-heatmaps.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(photos), *heatmap_shape),
            )

        for local_index, (path, document, prepared_item) in enumerate(
            zip(batch_paths, documents, prepared, strict=True)
        ):
            sample_index = start + local_index
            raw = {name: raw_onnx[name][local_index : local_index + 1] for name in OUTPUT_NAMES}
            if content_store is not None and outer_store is not None:
                content_store[sample_index] = raw["content_corner_heatmaps"][0]
                outer_store[sample_index] = raw["outer_corner_heatmaps"][0]
            transform = prepared_item[1]
            variants: dict[str, Any] = {}
            for decoder_name in ("legacy_p2", "decoder_v1", "decoder_v2"):
                prediction = _prediction_from_raw(
                    raw,
                    transform,
                    document.original_rgb.shape,
                    decoder_name,
                )
                decision = AutomaticGeometryService(
                    _StaticDetector(prediction),
                    policy=ConfidencePolicy(),
                ).localize(document.original_rgb)
                refinement = (
                    refine_quad_edges(
                        document.original_rgb,
                        prediction.content_quad,
                        prediction.boundary_map,
                    )
                    if prediction.content_quad is not None
                    else None
                )
                variants[decoder_name] = {
                    "prediction": prediction,
                    "decision": decision,
                    "refinement": refinement,
                }
            scale_x, scale_y, offset_x, offset_y, _input_size = transform
            resized_width = round((document.original_rgb.shape[1] - 1) * scale_x) + 1
            resized_height = round((document.original_rgb.shape[0] - 1) * scale_y) + 1
            predictions.append(
                {
                    "path": path.resolve(),
                    "photo": str(
                        path.resolve().relative_to(args.dataset_root.expanduser().resolve())
                    ),
                    "shape": document.original_rgb.shape,
                    "preprocess": {
                        "original_size_metadata": document.metadata.get("original_size"),
                        "oriented_shape": list(document.original_rgb.shape),
                        "exif_orientation": document.metadata.get("exif", {}).get("274"),
                        "input_size": image_size,
                        "resize_size": [resized_width, resized_height],
                        "offset": [offset_x, offset_y],
                        "scale": [scale_x, scale_y],
                        "heatmap_shape": list(raw["content_corner_heatmaps"].shape),
                    },
                    "raw": {
                        name: _array_summary(value)
                        for name, value in raw.items()
                        if name in {"content_corner_heatmaps", "outer_corner_heatmaps"}
                    },
                    "raw_store_index": sample_index if args.save_raw_heatmaps else None,
                    "variants": variants,
                }
            )
        _progress(min(start + len(batch_paths), len(photos)), len(photos), "photo-only replay")

    if content_store is not None:
        content_store.flush()
    if outer_store is not None:
        outer_store.flush()

    # 所有预测冻结后才物化完整 GT 记录。
    records = _read_manifest(args.manifest)
    truth_by_path = {
        _resolve_manifest_image(args.dataset_root, str(record["image"])): record
        for record in records
        if record.get("split") == args.split
    }
    case_reports: list[dict[str, Any]] = []
    group_ids: list[str] = []
    for prediction_record in predictions:
        record = truth_by_path[prediction_record["path"]]
        truth = _truth_from_manifest_record(record, prediction_record["shape"])
        case_report: dict[str, Any] = {
            "case": str(record.get("id", Path(prediction_record["photo"]).name)),
            "photo": prediction_record["photo"],
            "preprocess": prediction_record["preprocess"],
            "raw": prediction_record["raw"],
            "raw_store_index": prediction_record["raw_store_index"],
            "variants": {},
        }
        for name, values in prediction_record["variants"].items():
            model_prediction: QuadPrediction = values["prediction"]
            decision = values["decision"]
            refinement = values["refinement"]
            case_report["variants"][name] = {
                "decoder": model_prediction.decoder_diagnostics.get("content"),
                "coarse_quad": _normalized(
                    model_prediction.content_quad, prediction_record["shape"]
                ),
                "refined_attempt_quad": _normalized(
                    refinement.attempted_corners if refinement is not None else None,
                    prediction_record["shape"],
                ),
                "rollback_quad": _normalized(
                    refinement.corners if refinement is not None else None,
                    prediction_record["shape"],
                ),
                "refinement": _refinement_report(refinement),
                "decision": decision.to_dict(prediction_record["shape"]),
                "metrics": {
                    "coarse": _quad_metrics(model_prediction.content_quad, truth.content_quad),
                    "refined_attempt": _quad_metrics(
                        refinement.attempted_corners if refinement is not None else None,
                        truth.content_quad,
                    ),
                    "rollback": _quad_metrics(
                        refinement.corners if refinement is not None else None,
                        truth.content_quad,
                    ),
                    "final": evaluate_geometry_decision(decision, truth),
                },
            }
        case_reports.append(case_report)
        group_ids.append(str(record["group_id"]))

    summaries = {
        name: _variant_summary(case_reports, name, group_ids)
        for name in ("legacy_p2", "decoder_v1", "decoder_v2")
    }
    report = {
        "format_version": 1,
        "kind": "p4_geometry_parity_audit",
        "protocol": "photo_only_then_frozen_gt",
        "split": args.split,
        "sample_count": len(case_reports),
        "checkpoint": _file_identity(checkpoint_path),
        "onnx": _file_identity(onnx_path),
        "manifest": _file_identity(args.manifest.expanduser().resolve()),
        "device": str(device),
        "onnx_provider": "CPUExecutionProvider",
        "torch_onnx_parity": {name: value.report() for name, value in parity.items()},
        "raw_heatmap_files": (
            {
                "content": str(output_directory / "raw-content-heatmaps.npy"),
                "outer": str(output_directory / "raw-outer-heatmaps.npy"),
            }
            if args.save_raw_heatmaps
            else None
        ),
        "summaries": summaries,
        "reference_parity": _reference_parity(case_reports, args.reference_evaluation),
        "git": _git_metadata(),
        "command": [sys.executable, *sys.argv],
        "wall_time_seconds": round(time.monotonic() - started, 4),
        "cases": case_reports,
    }
    output = output_directory / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_output = output_directory / "summary.json"
    summary_output.write_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_output)
    return 0


def _prediction_from_raw(
    raw: dict[str, np.ndarray],
    transform: tuple[float, float, int, int, int],
    image_shape: tuple[int, ...],
    decoder_name: str,
) -> QuadPrediction:
    content_quad, confidences, content_diagnostics = _decode(
        raw["content_corner_heatmaps"], transform, image_shape, decoder_name
    )
    outer_probability = float(_sigmoid(raw["outer_presence_logits"]).reshape(-1)[0])
    outer_quad = None
    outer_diagnostics = None
    if outer_probability >= 0.5:
        outer_quad, _outer_confidences, outer_diagnostics = _decode(
            raw["outer_corner_heatmaps"], transform, image_shape, decoder_name
        )
    class_probabilities = _softmax(raw["class_logits"].reshape(-1))
    class_index = int(np.argmax(class_probabilities))
    content_mask = _sigmoid(raw["content_mask_logits"].squeeze())
    boundary = _sigmoid(raw["boundary_logits"].squeeze())
    candidates: tuple[QuadrilateralCandidate, ...] = ()
    if content_quad is not None:
        corners = content_diagnostics.get("corners", [])
        margin = (
            1.0
            if decoder_name == "legacy_p2"
            else min(float(item["peak_difference"]) for item in corners)
        )
        candidates = (
            QuadrilateralCandidate(
                content_quad,
                float(np.mean(confidences)),
                {"heatmap_mean": float(np.mean(confidences)), "candidate_margin": margin},
                "quadlocator_onnx",
                TargetLayer.CONTENT,
            ),
        )
    return QuadPrediction(
        content_quad=content_quad,
        outer_quad=outer_quad,
        corner_confidences=confidences,
        presence_confidence=float(_sigmoid(raw["presence_logits"]).reshape(-1)[0]),
        outer_presence_confidence=outer_probability,
        target_class=CLASS_ORDER[class_index],
        class_confidence=float(class_probabilities[class_index]),
        layer_confidence=_layer_confidence(content_quad, outer_quad, content_mask),
        content_mask=content_mask,
        boundary_map=boundary,
        decoder_diagnostics={"content": content_diagnostics, "outer": outer_diagnostics},
        candidates=candidates,
        backend=f"quadlocator_onnx_{decoder_name}",
    )


def _decode(
    logits: np.ndarray,
    transform: tuple[float, float, int, int, int],
    image_shape: tuple[int, ...],
    decoder_name: str,
) -> tuple[np.ndarray | None, tuple[float, float, float, float], dict[str, Any]]:
    if decoder_name == "legacy_p2":
        coordinates, confidences = _legacy_p2_coordinates(logits)
        current = decode_corner_logits(logits)
        diagnostics = {
            "decoder": {"version": "p2-threshold-centroid", "threshold_ratio": 0.55},
            "corners": [item.to_dict() for item in current.diagnostics],
        }
    elif decoder_name == "decoder_v1":
        decoded = _decode_v1(logits)
        coordinates = decoded[0]
        confidences = decoded[1]
        diagnostics = decoded[2]
    elif decoder_name == "decoder_v2":
        decoded = decode_corner_logits(logits)
        coordinates = decoded.coordinates
        confidences = decoded.confidences
        diagnostics = {
            "decoder": decoded.spec.to_dict(),
            "corners": [item.to_dict() for item in decoded.diagnostics],
        }
    else:
        raise ValueError(f"未知 decoder：{decoder_name}")
    return (
        _map_heatmap_coordinates(coordinates, transform, image_shape, logits.shape[2:]),
        confidences,
        diagnostics,
    )


def _legacy_p2_coordinates(
    logits: np.ndarray,
) -> tuple[np.ndarray | None, tuple[float, float, float, float]]:
    probabilities = _sigmoid(np.asarray(logits, np.float32))[0]
    coordinates: list[list[float]] = []
    confidences: list[float] = []
    for heatmap in probabilities:
        confidence = float(heatmap.max())
        confidences.append(confidence)
        threshold = max(0.05, confidence * 0.55)
        weights = np.where(heatmap >= threshold, heatmap, 0.0)
        total = float(weights.sum())
        if confidence < 0.05 or total <= 1e-8:
            continue
        yy, xx = np.indices(heatmap.shape, dtype=np.float32)
        coordinates.append(
            [float((xx * weights).sum() / total), float((yy * weights).sum() / total)]
        )
    return (
        np.asarray(coordinates, np.float32) if len(coordinates) == 4 else None,
        tuple(confidences),  # type: ignore[arg-type]
    )


def _decode_v1(
    logits: np.ndarray,
) -> tuple[np.ndarray | None, tuple[float, float, float, float], dict[str, Any]]:
    spec = CornerDecoderSpec(version="quad-peak-local-softargmax-v1")
    probabilities = _sigmoid(np.asarray(logits, np.float32))[0]
    coordinates: list[list[float]] = []
    confidences: list[float] = []
    diagnostics: list[CornerPeakDiagnostics] = []
    for heatmap in probabilities:
        height, width = heatmap.shape
        peak_y, peak_x = divmod(int(np.argmax(heatmap)), width)
        peak1 = float(heatmap[peak_y, peak_x])
        suppressed = heatmap.copy()
        yy, xx = np.ogrid[:height, :width]
        suppressed[(xx - peak_x) ** 2 + (yy - peak_y) ** 2 <= spec.nms_radius**2] = -1.0
        second_y, second_x = divmod(int(np.argmax(suppressed)), width)
        peak2 = max(0.0, float(suppressed[second_y, second_x]))
        radius = spec.local_window // 2
        y0, y1 = max(0, peak_y - radius), min(height, peak_y + radius + 1)
        x0, x1 = max(0, peak_x - radius), min(width, peak_x + radius + 1)
        local = heatmap[y0:y1, x0:x1]
        total = max(float(local.sum()), 1e-8)
        local_y, local_x = np.indices(local.shape, dtype=np.float32)
        x = float(((local_x + x0) * local).sum() / total)
        y = float(((local_y + y0) * local).sum() / total)
        global_weights = heatmap / max(float(heatmap.sum()), 1e-8)
        entropy = -float(np.sum(global_weights * np.log(np.clip(global_weights, 1e-12, 1.0))))
        normalized_entropy = entropy / max(float(np.log(max(2, heatmap.size))), 1e-8)
        local_mean = (total - peak1) / max(1, local.size - 1)
        item = CornerPeakDiagnostics(
            peak1=peak1,
            peak2=peak2,
            peak_difference=max(0.0, peak1 - peak2),
            peak_ratio=peak1 / max(peak2, 1e-8),
            peak_distance=float(np.hypot(second_x - peak_x, second_y - peak_y)),
            normalized_entropy=float(np.clip(normalized_entropy, 0.0, 1.0)),
            local_sharpness=float(np.clip(peak1 - local_mean, 0.0, 1.0)),
            x=x,
            y=y,
        )
        diagnostics.append(item)
        confidences.append(peak1)
        if peak1 >= spec.minimum_peak:
            coordinates.append([x, y])
    return (
        np.asarray(coordinates, np.float32) if len(coordinates) == 4 else None,
        tuple(confidences),  # type: ignore[arg-type]
        {"decoder": spec.to_dict(), "corners": [item.to_dict() for item in diagnostics]},
    )


def _map_heatmap_coordinates(
    coordinates: np.ndarray | None,
    transform: tuple[float, float, int, int, int],
    image_shape: tuple[int, ...],
    heatmap_shape: tuple[int, int],
) -> np.ndarray | None:
    if coordinates is None:
        return None
    scale_x, scale_y, offset_x, offset_y, input_size = transform
    output_height, output_width = heatmap_shape
    points = []
    for heatmap_x, heatmap_y in coordinates:
        model_x = float(heatmap_x) * (input_size - 1) / max(1, output_width - 1)
        model_y = float(heatmap_y) * (input_size - 1) / max(1, output_height - 1)
        points.append(
            [
                float(np.clip((model_x - offset_x) / max(scale_x, 1e-8), 0, image_shape[1] - 1)),
                float(np.clip((model_y - offset_y) / max(scale_y, 1e-8), 0, image_shape[0] - 1)),
            ]
        )
    try:
        return order_corners(np.asarray(points, np.float32))
    except ValueError:
        return None


def _variant_summary(
    cases: list[dict[str, Any]], name: str, group_ids: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("coarse", "refined_attempt", "rollback"):
        metrics = [case["variants"][name]["metrics"][stage] for case in cases]
        result[stage] = _raw_metric_summary(metrics)
    final_metrics = [case["variants"][name]["metrics"]["final"] for case in cases]
    result["final_decision"] = aggregate_geometry_results(final_metrics, group_ids=group_ids)
    result["refinement_accepted_rate"] = float(
        np.mean([case["variants"][name]["refinement"]["accepted"] for case in cases])
    )
    result["rejection_reason_counts"] = _count_rejection_reasons(final_metrics)
    return result


def _raw_metric_summary(metrics: list[dict[str, float | bool]]) -> dict[str, float | int]:
    selected = [item for item in metrics if bool(item["has_truth"]) and bool(item["has_quad"])]
    nce = np.asarray([float(item["corner_nce"]) for item in selected], np.float64)
    iou = np.asarray([float(item["quad_iou"]) for item in selected], np.float64)
    return {
        "eligible_count": sum(bool(item["has_truth"]) for item in metrics),
        "candidate_count": len(selected),
        "corner_nce_median": float(np.median(nce)) if nce.size else 1.0,
        "corner_nce_p95": float(np.percentile(nce, 95)) if nce.size else 1.0,
        "quad_iou_median": float(np.median(iou)) if iou.size else 0.0,
        "quad_iou_p05": float(np.percentile(iou, 5)) if iou.size else 0.0,
        "strict_correct_rate": float(
            np.mean(
                [
                    (float(item["corner_nce"]) <= 0.01 and float(item["quad_iou"]) >= 0.93)
                    for item in selected
                ]
            )
        )
        if selected
        else 0.0,
    }


def _quad_metrics(quad: np.ndarray | None, truth: np.ndarray | None) -> dict[str, float | bool]:
    if truth is None:
        return {
            "has_truth": False,
            "has_quad": quad is not None,
            "corner_nce": 0.0,
            "quad_iou": 0.0,
        }
    if quad is None:
        return {"has_truth": True, "has_quad": False, "corner_nce": 1.0, "quad_iou": 0.0}
    nce, iou, maximum = corner_metrics(quad, truth)
    return {
        "has_truth": True,
        "has_quad": True,
        "corner_nce": round(nce, 8),
        "quad_iou": round(iou, 8),
        "max_corner_error_px": round(maximum, 4),
    }


def _reference_parity(cases: list[dict[str, Any]], path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    reference = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    by_photo = {str(case["photo"]): case for case in reference["cases"]}
    deltas = []
    matched = 0
    for case in cases:
        old = by_photo.get(case["photo"])
        if old is None:
            continue
        candidates = old["metrics"].get("candidates", [])
        if not candidates:
            continue
        expected = candidates[0]
        observed = case["variants"]["legacy_p2"]["metrics"]["coarse"]
        deltas.append(
            (
                abs(float(expected["corner_nce"]) - float(observed["corner_nce"])),
                abs(float(expected["quad_iou"]) - float(observed["quad_iou"])),
            )
        )
        matched += 1
    delta_array = np.asarray(deltas, np.float64) if deltas else np.zeros((0, 2), np.float64)
    return {
        "reference": _file_identity(path.expanduser().resolve()),
        "matched_candidate_count": matched,
        "legacy_coarse_nce_max_abs_delta": float(delta_array[:, 0].max(initial=0.0)),
        "legacy_coarse_iou_max_abs_delta": float(delta_array[:, 1].max(initial=0.0)),
        "reference_summary": reference.get("summary"),
    }


def _refinement_report(refinement: Any) -> dict[str, Any]:
    if refinement is None:
        return {"accepted": False, "reason": "no_candidate"}
    return {
        "accepted": bool(refinement.accepted),
        "reason": refinement.reason,
        "outcome": refinement.outcome,
        "edge_support": list(refinement.edge_support),
        "corner_shifts_px": list(refinement.corner_shifts),
        "residual_median_px": list(refinement.residual_median),
        "residual_p95_px": list(refinement.residual_p95),
        "continuous_coverage": list(refinement.continuous_coverage),
        "gradient_normal_alignment": list(refinement.gradient_normal_alignment),
        "boundary_consistency": list(refinement.boundary_consistency),
        "area_drift": refinement.area_drift,
        "aspect_drift": refinement.aspect_drift,
    }


def _normalized(quad: np.ndarray | None, shape: tuple[int, ...]) -> list[list[float]] | None:
    if quad is None:
        return None
    scale = np.asarray([max(1, shape[1] - 1), max(1, shape[0] - 1)], np.float32)
    return np.clip(quad / scale, 0.0, 1.0).astype(float).tolist()


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


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


def _torch_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前环境不支持 MPS")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不支持 CUDA")
    return torch.device(name)


def _count_rejection_reasons(metrics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in metrics:
        for reason in item["rejection_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * done / max(1, total))
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>5}/{total:<5} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
