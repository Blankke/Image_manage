"""QuadLocator 逐 epoch 几何、分类、校准与产品代理指标。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from training.quadlocator.losses import _softargmax_corners


@dataclass(slots=True)
class ValidationMetrics:
    """累积 validation batch，最终只输出可序列化标量与小型混淆矩阵。"""

    content_nce: list[float] = field(default_factory=list)
    content_iou: list[float] = field(default_factory=list)
    corner_confidence: list[float] = field(default_factory=list)
    outer_nce: list[float] = field(default_factory=list)
    outer_iou: list[float] = field(default_factory=list)
    outer_probabilities: list[float] = field(default_factory=list)
    outer_targets: list[int] = field(default_factory=list)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=np.int64))
    mask_intersection: float = 0.0
    mask_union: float = 0.0
    boundary_true_positive: float = 0.0
    boundary_predicted: float = 0.0
    boundary_target: float = 0.0
    boundary_positive_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(101, dtype=np.int64)
    )
    boundary_negative_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(101, dtype=np.int64)
    )
    present_count: int = 0
    no_candidate_count: int = 0
    layer_ambiguous_count: int = 0
    accepted_count: int = 0
    accepted_correct_count: int = 0
    content_strict_correct_count: int = 0
    ambiguous_target_count: int = 0
    ambiguous_rejected_count: int = 0

    def update(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> None:
        content_corners = (
            _softargmax_corners(outputs["content_corner_heatmaps"]).detach().cpu().numpy()
        )
        outer_corners = _softargmax_corners(outputs["outer_corner_heatmaps"]).detach().cpu().numpy()
        content_target = targets["content_corners"].detach().cpu().numpy()
        outer_target = targets["outer_corners"].detach().cpu().numpy()
        present = targets["presence"].detach().cpu().numpy().reshape(-1) >= 0.5
        ambiguous_targets = (
            targets.get("ambiguous", torch.zeros_like(targets["presence"]))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            >= 0.5
        )
        outer_present = targets["outer_present"].detach().cpu().numpy().reshape(-1) >= 0.5
        content_confidences = (
            torch.sigmoid(outputs["content_corner_heatmaps"])
            .flatten(2)
            .amax(dim=2)
            .detach()
            .cpu()
            .numpy()
        )
        presence_probability = (
            torch.sigmoid(outputs["presence_logits"]).detach().cpu().numpy().reshape(-1)
        )
        outer_probability = (
            torch.sigmoid(outputs["outer_presence_logits"]).detach().cpu().numpy().reshape(-1)
        )
        class_probability = torch.softmax(outputs["class_logits"], dim=1).detach().cpu().numpy()
        predicted_class = np.argmax(class_probability, axis=1)
        target_class = targets["target_class"].detach().cpu().numpy()
        for truth, prediction in zip(target_class, predicted_class, strict=True):
            self.confusion[int(truth), int(prediction)] += 1

        mask_prediction = torch.sigmoid(outputs["content_mask_logits"]) >= 0.5
        mask_target = targets["content_mask"] >= 0.5
        self.mask_intersection += float(torch.logical_and(mask_prediction, mask_target).sum())
        self.mask_union += float(torch.logical_or(mask_prediction, mask_target).sum())
        boundary_probability = torch.sigmoid(outputs["boundary_logits"])
        boundary_prediction = boundary_probability >= 0.5
        boundary_target = targets["boundary"] >= 0.5
        self.boundary_true_positive += float(
            torch.logical_and(boundary_prediction, boundary_target).sum()
        )
        self.boundary_predicted += float(boundary_prediction.sum())
        self.boundary_target += float(boundary_target.sum())
        probability_u8 = (
            torch.clamp(torch.round(boundary_probability * 100.0), 0, 100)
            .to(torch.int64)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        target_flat = boundary_target.detach().cpu().numpy().reshape(-1)
        self.boundary_positive_histogram += np.bincount(probability_u8[target_flat], minlength=101)
        self.boundary_negative_histogram += np.bincount(probability_u8[~target_flat], minlength=101)

        for index in range(len(present)):
            self.outer_probabilities.append(float(outer_probability[index]))
            self.outer_targets.append(int(outer_present[index]))
            if present[index]:
                self.present_count += 1
                nce = _corner_nce(content_corners[index], content_target[index])
                iou = _quad_iou(content_corners[index], content_target[index])
                self.content_nce.append(nce)
                self.content_iou.append(iou)
                self.corner_confidence.append(float(np.mean(content_confidences[index])))
                self.content_strict_correct_count += int(nce <= 0.01 and iou >= 0.93)
            if outer_present[index]:
                self.outer_nce.append(_corner_nce(outer_corners[index], outer_target[index]))
                self.outer_iou.append(_quad_iou(outer_corners[index], outer_target[index]))

            no_candidate = bool(
                presence_probability[index] < 0.5 or np.min(content_confidences[index]) < 0.05
            )
            if no_candidate:
                self.no_candidate_count += 1
            layer_ambiguous = bool(
                outer_probability[index] >= 0.5
                and not _contains(outer_corners[index], content_corners[index])
            )
            if layer_ambiguous:
                self.layer_ambiguous_count += 1
            class_confidence = float(np.max(class_probability[index]))
            accepted = bool(
                not no_candidate
                and not layer_ambiguous
                and presence_probability[index] >= 0.66
                and class_confidence >= 0.58
                and np.min(content_confidences[index]) >= 0.52
                and predicted_class[index] != 3
            )
            if accepted:
                self.accepted_count += 1
                correct = bool(
                    present[index]
                    and predicted_class[index] == target_class[index]
                    and self.content_nce[-1] <= 0.01
                    and self.content_iou[-1] >= 0.93
                )
                self.accepted_correct_count += int(correct)
            if ambiguous_targets[index]:
                self.ambiguous_target_count += 1
                self.ambiguous_rejected_count += int(not accepted)

    def compute(self) -> dict[str, object]:
        outer = _binary_calibration(self.outer_probabilities, self.outer_targets)
        confusion = self.confusion.astype(int).tolist()
        class_recall: dict[str, float] = {}
        for index, name in enumerate(("artwork", "postcard", "screen", "none")):
            denominator = int(self.confusion[index].sum())
            class_recall[name] = (
                float(self.confusion[index, index] / denominator) if denominator else 0.0
            )
        sample_count = max(1, int(self.confusion.sum()))
        accepted_precision = (
            self.accepted_correct_count / self.accepted_count if self.accepted_count else 0.0
        )
        coverage = self.accepted_correct_count / self.present_count if self.present_count else 0.0
        ambiguous_rejection_rate = (
            self.ambiguous_rejected_count / self.ambiguous_target_count
            if self.ambiguous_target_count
            else 1.0
        )
        boundary_curve = _binary_curve_from_histograms(
            self.boundary_positive_histogram,
            self.boundary_negative_histogram,
        )
        result: dict[str, object] = {
            "content_corner_nce_median": _percentile(self.content_nce, 50),
            "content_corner_nce_p95": _percentile(self.content_nce, 95, empty=1.0),
            "content_iou_median": _percentile(self.content_iou, 50),
            "content_iou_p05": _percentile(self.content_iou, 5),
            "content_strict_correct_rate": (
                self.content_strict_correct_count / self.present_count
                if self.present_count
                else 0.0
            ),
            "corner_heatmap_confidence": float(np.mean(self.corner_confidence))
            if self.corner_confidence
            else 0.0,
            "content_mask_iou": self.mask_intersection / max(1.0, self.mask_union),
            "boundary_f1": 2.0
            * self.boundary_true_positive
            / max(1.0, self.boundary_predicted + self.boundary_target),
            "boundary_auprc": boundary_curve["auprc"],
            "boundary_best_f1": boundary_curve["best_f1"],
            "boundary_best_threshold": boundary_curve["best_threshold"],
            "boundary_f1_at_0_5": boundary_curve["f1_at_0_5"],
            "outer_presence": outer,
            "outer_corner_nce_median": _percentile(self.outer_nce, 50),
            "outer_corner_nce_p95": _percentile(self.outer_nce, 95, empty=1.0),
            "outer_iou_median": _percentile(self.outer_iou, 50),
            "outer_iou_p05": _percentile(self.outer_iou, 5),
            "class_confusion": confusion,
            "class_recall": class_recall,
            "no_candidate_rate": self.no_candidate_count / sample_count,
            "layer_ambiguous_rate": self.layer_ambiguous_count / sample_count,
            "accepted_precision_proxy": accepted_precision,
            "coverage_proxy": coverage,
            "ambiguous_target_count": self.ambiguous_target_count,
            "ambiguous_rejection_rate": ambiguous_rejection_rate,
        }
        # 选模同时关注几何中位/尾部、拒绝错误、outer 校准和分类。歧义样本的误接受
        # 直接违背无人值守语义，因此给它独立且足够显著的权重。
        result["selection_score"] = float(
            0.28 * result["content_iou_median"]
            + 0.18 * result["content_iou_p05"]
            + 0.14 * (1.0 - min(1.0, result["content_corner_nce_p95"] / 0.05))
            + 0.075 * (1.0 - result["no_candidate_rate"])
            + 0.075 * (1.0 - result["layer_ambiguous_rate"])
            + 0.10 * float(np.trace(self.confusion) / sample_count)
            + 0.05 * (1.0 - outer["brier"])
            + 0.10 * ambiguous_rejection_rate
        )
        return result


def _binary_curve_from_histograms(
    positive: np.ndarray,
    negative: np.ndarray,
) -> dict[str, float]:
    """从 0.01 分箱统计生成可复现的 PR-AUC 与最佳 F1。"""

    true_positive = np.cumsum(positive[::-1]).astype(np.float64)
    false_positive = np.cumsum(negative[::-1]).astype(np.float64)
    total_positive = max(1.0, float(positive.sum()))
    recall = true_positive / total_positive
    precision = true_positive / np.maximum(1.0, true_positive + false_positive)
    f1 = 2.0 * precision * recall / np.maximum(1e-12, precision + recall)
    best_index = int(np.argmax(f1))
    thresholds = np.arange(100, -1, -1, dtype=np.float64) / 100.0
    # recall 随阈值降低而递增，可直接梯形积分。
    auprc = float(np.trapezoid(precision, recall))
    threshold_05_index = int(np.argmin(np.abs(thresholds - 0.5)))
    return {
        "auprc": auprc,
        "best_f1": float(f1[best_index]),
        "best_threshold": float(thresholds[best_index]),
        "f1_at_0_5": float(f1[threshold_05_index]),
    }


def _corner_nce(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(predicted - target, axis=1)) / np.sqrt(2.0))


def _quad_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = cv2.convexHull(predicted.astype(np.float32)).reshape(-1, 2)
    target = cv2.convexHull(target.astype(np.float32)).reshape(-1, 2)
    if len(predicted) != 4 or len(target) != 4:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(predicted, target)
    union = (
        abs(float(cv2.contourArea(predicted))) + abs(float(cv2.contourArea(target))) - intersection
    )
    return float(intersection / union) if union > 1e-8 else 0.0


def _contains(outer: np.ndarray, content: np.ndarray) -> bool:
    contour = cv2.convexHull(outer.astype(np.float32)).reshape(-1, 1, 2)
    if len(contour) != 4:
        return False
    return all(
        cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0 for point in content
    )


def _binary_calibration(probabilities: list[float], targets: list[int]) -> dict[str, float]:
    if not probabilities:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "brier": 0.0,
            "ece": 0.0,
        }
    probability = np.asarray(probabilities, np.float64)
    target = np.asarray(targets, np.int64)
    predicted = probability >= 0.5
    true_positive = int(np.sum(predicted & (target == 1)))
    false_positive = int(np.sum(predicted & (target == 0)))
    false_negative = int(np.sum(~predicted & (target == 1)))
    true_negative = int(np.sum(~predicted & (target == 0)))
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (probability >= lower) & (probability < lower + 0.1)
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(probability[selected])) - float(np.mean(target[selected]))
            )
    return {
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "false_positive_rate": false_positive / max(1, false_positive + true_negative),
        "brier": float(np.mean(np.square(probability - target))),
        "ece": ece,
    }


def _percentile(values: list[float], percentile: int, *, empty: float = 0.0) -> float:
    return float(np.percentile(values, percentile)) if values else empty
