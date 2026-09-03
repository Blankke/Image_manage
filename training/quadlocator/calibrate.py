"""冻结 QuadLocator checkpoint，在 validation split 校准各头置信度起点。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.quadlocator.calibrate \
        --checkpoint "$SCREENRESTORE_RUN_ROOT/p2/stage-c/best.pt" \
        --manifest "$SCREENRESTORE_DATA_ROOT/manifests/p2/calibration.geometry.jsonl" \
        --dataset-root "$SCREENRESTORE_DATA_ROOT" \
        --output "$SCREENRESTORE_RUN_ROOT/p2/calibration.json"

本脚本不创建 optimizer、不更新参数且不读取 test split。阈值只允许从保守默认值向上
调整，以 validation precision 约束下的 true-positive 数量选取。boundary/layer/combined
是模型空间代理校准；最终产品阈值仍需结合 ONNX e2e validation 的原图精修诊断复核。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.quadlocator.dataset import QuadDataset
from training.quadlocator.losses import _softargmax_corners
from training.quadlocator.model import QuadLocatorS

DEFAULT_THRESHOLDS = {
    "content_presence": 0.66,
    "outer_presence": 0.50,
    "class": 0.58,
    "corner": 0.52,
    "boundary": 0.16,
    "layer": 0.58,
    "combined": 0.68,
}

# P3/P4 只扩展训练与 decoder 元数据，模型仍保持与 P2 v2 相同的
# QuadLocatorS 7-output 契约，因此冻结 checkpoint 校准可安全共用。
SUPPORTED_CHECKPOINT_FORMATS = frozenset({2, 3, 4})


@dataclass(slots=True)
class CalibrationValues:
    scores: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in DEFAULT_THRESHOLDS}
    )
    labels: dict[str, list[bool]] = field(
        default_factory=lambda: {name: [] for name in DEFAULT_THRESHOLDS}
    )
    present: list[bool] = field(default_factory=list)
    predicted_non_none: list[bool] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--minimum-precision", type=float, default=0.99)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0.5 <= args.minimum_precision <= 1.0:
        raise ValueError("batch-size 或 minimum-precision 无效")
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_format = checkpoint.get("format_version")
    if checkpoint_format not in SUPPORTED_CHECKPOINT_FORMATS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_CHECKPOINT_FORMATS))
        raise RuntimeError(
            f"confidence calibration 只接受 format_version={supported} checkpoint；"
            f"实际为 {checkpoint_format!r}"
        )
    device = _device(args.device)
    model = QuadLocatorS(float(checkpoint["width_multiplier"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    dataset = QuadDataset(
        args.manifest,
        split="validation",
        image_size=int(checkpoint["image_size"]),
        dataset_root=args.dataset_root,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    values = CalibrationValues()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            _progress(batch_index - 1, len(loader), "冻结模型 calibration")
            device_batch = {key: tensor.to(device) for key, tensor in batch.items()}
            _collect(values, model(device_batch["image"]), device_batch)
    _progress(len(loader), len(loader), "冻结模型 calibration")
    calibrated = {
        name: _calibrate_threshold(
            values.scores[name],
            values.labels[name],
            minimum=DEFAULT_THRESHOLDS[name],
            minimum_precision=args.minimum_precision,
        )
        for name in DEFAULT_THRESHOLDS
    }
    product = _product_proxy(values, calibrated)
    report = {
        "format_version": 1,
        "protocol": "quadlocator_validation_calibration",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "manifest": str(args.manifest.expanduser().resolve()),
        "split": "validation",
        "test_split_read": False,
        "model_frozen": True,
        "minimum_precision": args.minimum_precision,
        "sample_count": len(dataset),
        "independent_group_count": dataset.statistics()["unique_group_count"],
        "thresholds": calibrated,
        "product_proxy": product,
        "notes": [
            "阈值不低于保守默认值",
            "boundary/layer/combined 为模型空间代理，需用 ONNX 原图精修 validation 再复核",
            "test 禁止用于阈值选择",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def _collect(
    values: CalibrationValues,
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> None:
    content = _softargmax_corners(outputs["content_corner_heatmaps"]).cpu().numpy()
    outer = _softargmax_corners(outputs["outer_corner_heatmaps"]).cpu().numpy()
    content_target = targets["content_corners"].cpu().numpy()
    outer_target = targets["outer_corners"].cpu().numpy()
    present = targets["presence"].cpu().numpy().reshape(-1) >= 0.5
    outer_present = targets["outer_present"].cpu().numpy().reshape(-1) >= 0.5
    target_class = targets["target_class"].cpu().numpy()
    content_presence = torch.sigmoid(outputs["presence_logits"]).cpu().numpy().reshape(-1)
    outer_presence = torch.sigmoid(outputs["outer_presence_logits"]).cpu().numpy().reshape(-1)
    class_probability = torch.softmax(outputs["class_logits"], dim=1).cpu().numpy()
    predicted_class = np.argmax(class_probability, axis=1)
    corner_score = (
        torch.sigmoid(outputs["content_corner_heatmaps"])
        .flatten(2)
        .amax(dim=2)
        .amin(dim=1)
        .cpu()
        .numpy()
    )
    boundary_probability = torch.sigmoid(outputs["boundary_logits"]).flatten(1).cpu().numpy()
    mask_probability = torch.sigmoid(outputs["content_mask_logits"]).cpu().numpy()
    for index in range(len(present)):
        content_nce = _corner_nce(content[index], content_target[index]) if present[index] else 1.0
        outer_nce = (
            _corner_nce(outer[index], outer_target[index]) if outer_present[index] else 1.0
        )
        class_correct = bool(predicted_class[index] == target_class[index])
        corner_good = bool(present[index] and content_nce <= 0.01)
        boundary_score = float(np.percentile(boundary_probability[index], 95))
        layer_score = _layer_score(
            content[index],
            outer[index],
            float(outer_presence[index]),
            mask_probability[index, 0],
        )
        layer_good = bool(
            present[index]
            and content_nce <= 0.02
            and (not outer_present[index] or content_nce <= outer_nce)
        )
        components = np.array(
            [
                content_presence[index],
                np.max(class_probability[index]),
                corner_score[index],
                boundary_score,
                layer_score,
            ],
            np.float64,
        )
        combined = float(np.dot(components, [0.22, 0.14, 0.24, 0.24, 0.16]))
        scores = {
            "content_presence": float(content_presence[index]),
            "outer_presence": float(outer_presence[index]),
            "class": float(np.max(class_probability[index])),
            "corner": float(corner_score[index]),
            "boundary": boundary_score,
            "layer": layer_score,
            "combined": combined,
        }
        labels = {
            "content_presence": bool(present[index]),
            "outer_presence": bool(outer_present[index]),
            "class": class_correct,
            "corner": corner_good,
            "boundary": corner_good,
            "layer": layer_good,
            "combined": bool(corner_good and class_correct and layer_good),
        }
        for name in DEFAULT_THRESHOLDS:
            values.scores[name].append(scores[name])
            values.labels[name].append(labels[name])
        values.present.append(bool(present[index]))
        values.predicted_non_none.append(bool(predicted_class[index] != 3))


def _calibrate_threshold(
    scores: list[float],
    labels: list[bool],
    *,
    minimum: float,
    minimum_precision: float,
) -> dict[str, float | int]:
    score = np.asarray(scores, np.float64)
    label = np.asarray(labels, bool)
    candidates = sorted({minimum, *[float(value) for value in score if value >= minimum]})
    best: tuple[int, float, float, int, int] | None = None
    for threshold in candidates:
        accepted = score >= threshold
        accepted_count = int(np.sum(accepted))
        true_positive = int(np.sum(accepted & label))
        precision = true_positive / max(1, accepted_count)
        if precision < minimum_precision:
            continue
        candidate = (true_positive, -threshold, precision, accepted_count, int(np.sum(label)))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        threshold = 1.0
        accepted_count = int(np.sum(score >= threshold))
        true_positive = int(np.sum((score >= threshold) & label))
        precision = true_positive / max(1, accepted_count)
        positive_count = int(np.sum(label))
    else:
        true_positive, negative_threshold, precision, accepted_count, positive_count = best
        threshold = -negative_threshold
    return {
        "threshold": round(float(threshold), 8),
        "precision": round(float(precision), 8),
        "recall": round(true_positive / max(1, positive_count), 8),
        "accepted_count": accepted_count,
        "positive_count": positive_count,
    }


def _product_proxy(
    values: CalibrationValues,
    calibrated: dict[str, dict[str, float | int]],
) -> dict[str, float | int]:
    accepted = np.asarray(values.predicted_non_none, bool)
    for name in ("content_presence", "class", "corner", "boundary", "layer", "combined"):
        accepted &= np.asarray(values.scores[name]) >= float(calibrated[name]["threshold"])
    good = np.asarray(values.labels["combined"], bool)
    present = np.asarray(values.present, bool)
    accepted_count = int(np.sum(accepted))
    return {
        "accepted_count": accepted_count,
        "accepted_precision": round(float(np.sum(accepted & good) / max(1, accepted_count)), 8),
        "in_scope_coverage": round(float(np.sum(accepted & present) / max(1, np.sum(present))), 8),
    }


def _corner_nce(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(predicted - target, axis=1)) / np.sqrt(2.0))


def _layer_score(
    content: np.ndarray,
    outer: np.ndarray,
    outer_presence: float,
    content_mask: np.ndarray,
) -> float:
    mask_evidence = float(np.clip(np.mean(content_mask > 0.5) * 4.0, 0.0, 1.0))
    if outer_presence < 0.5:
        return 0.65 + 0.25 * mask_evidence
    contour = cv2.convexHull(outer.astype(np.float32)).reshape(-1, 1, 2)
    if len(contour) != 4:
        return 0.0
    containment = np.mean(
        [cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0 for point in content]
    )
    return float(np.clip((0.50 + 0.30 * mask_evidence) * containment, 0.0, 1.0))


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


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    filled = round(width * min(1.0, done / max(1, total)))
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
