"""严格禁止 clean reference 参与推理的端到端几何评分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.stats import beta

from screenrestore.geometry import LocalizationDecision, TargetClass, TargetLayer, order_corners


@dataclass(frozen=True, slots=True)
class GeometryGroundTruth:
    """人工或离线标注得到的产品层级真值。"""

    content_quad: np.ndarray | None
    target_class: TargetClass
    target_layer: TargetLayer = TargetLayer.CONTENT
    in_scope: bool = True

    def __post_init__(self) -> None:
        if self.target_class == TargetClass.NONE:
            if self.content_quad is not None:
                raise ValueError("none 负样本不能提供 content_quad")
            # 无目标样本用于检验自动拒绝，不参与四角质量与 in-scope 覆盖率。
            object.__setattr__(self, "in_scope", False)
            return
        if self.content_quad is None:
            raise ValueError("存在目标的真值必须提供 content_quad")
        quad = order_corners(self.content_quad)
        object.__setattr__(self, "content_quad", quad.copy())


@dataclass(frozen=True, slots=True)
class GeometryGate:
    """第一版产品验收目标；最低样本数防止把 smoke test 当发布结论。"""

    accepted_precision_min: float = 0.99
    in_scope_coverage_min: float = 0.90
    wrong_layer_rate_max: float = 0.005
    nce_p95_max: float = 0.01
    iou_median_min: float = 0.97
    iou_p05_min: float = 0.93
    minimum_samples: int = 100


def evaluate_geometry_decision(
    decision: LocalizationDecision,
    ground_truth: GeometryGroundTruth,
) -> dict[str, Any]:
    """仅在定位已完成后使用四角 GT 打分。"""

    selected_nce = 1.0
    selected_iou = 0.0
    max_corner_error = float("inf")
    if decision.proposed_corners is not None and ground_truth.content_quad is not None:
        selected_nce, selected_iou, max_corner_error = corner_metrics(
            decision.proposed_corners,
            ground_truth.content_quad,
        )
    class_correct = decision.target_class == ground_truth.target_class
    layer_correct = decision.layer == ground_truth.target_layer
    if ground_truth.target_class == TargetClass.NONE:
        # hard negative 只要求无人值守路径停止。强行要求分类头恰好输出 none 会把
        # “因低置信度而正确拒绝”的安全行为误判为失败。
        strict_correct = not decision.accepted
        layer_correct = True
    else:
        # strict_correct 描述候选四角/语义本身，与当前接受阈值解耦，供 validation
        # 校准器和风险覆盖曲线学习；accepted precision 仍只在 accepted 子集上统计。
        strict_correct = bool(
            decision.proposed_corners is not None
            and class_correct
            and layer_correct
            and selected_nce <= 0.01
            and selected_iou >= 0.93
        )
    candidate_metrics = []
    if ground_truth.content_quad is not None:
        for candidate in decision.candidates:
            nce, iou, maximum = corner_metrics(candidate.corners, ground_truth.content_quad)
            candidate_metrics.append(
                {
                    "source": candidate.source,
                    "layer": candidate.layer.value,
                    "runtime_score": round(candidate.confidence, 6),
                    "corner_nce": round(nce, 8),
                    "quad_iou": round(iou, 8),
                    "max_corner_error_px": round(maximum, 4),
                }
            )
    candidate_metrics.sort(key=lambda item: (item["corner_nce"], -item["quad_iou"]))
    refinement_outcome = "neutral"
    if bool(decision.diagnostics.get("refinement_accepted", False)) is False:
        refinement_outcome = "rolled_back"
    elif decision.coarse_corners is not None and ground_truth.content_quad is not None:
        coarse_nce, coarse_iou, _ = corner_metrics(decision.coarse_corners, ground_truth.content_quad)
        if selected_nce < coarse_nce - 1e-5 and selected_iou > coarse_iou + 1e-5:
            refinement_outcome = "improved"
        elif selected_nce > coarse_nce + 1e-5 or selected_iou < coarse_iou - 1e-5:
            refinement_outcome = "worsened"
    return {
        "accepted": decision.accepted,
        "correct": strict_correct,
        "strict_correct": strict_correct,
        "class_correct": class_correct,
        "layer_correct": layer_correct,
        "in_scope": ground_truth.in_scope,
        "corner_nce": round(selected_nce, 8),
        "quad_iou": round(selected_iou, 8),
        "max_corner_error_px": round(max_corner_error, 4),
        "confidence": round(decision.confidence, 6),
        "refinement_outcome": refinement_outcome,
        "backend": decision.backend,
        "rejection_reasons": [reason.value for reason in decision.rejection_reasons],
        "candidates": candidate_metrics,
        "best_candidate": candidate_metrics[0] if candidate_metrics else None,
    }


def aggregate_geometry_results(
    results: list[dict[str, Any]],
    gate: GeometryGate | None = None,
    group_ids: list[str] | None = None,
) -> dict[str, Any]:
    """聚合选择正确率、覆盖率、层级错误与尾部几何质量。

    ``minimum_samples`` 的发布含义是独立 group 数，而非同一连拍的图片数量。旧调用方
    未提供 ``group_ids`` 时维持一条结果视作一个独立 group 的 smoke 兼容语义。
    """

    gate = gate or GeometryGate()
    sample_count = len(results)
    if group_ids is not None and len(group_ids) != sample_count:
        raise ValueError("group_ids 长度必须与几何结果数量一致")
    independent_group_count = len(set(group_ids)) if group_ids is not None else sample_count
    accepted = [item for item in results if item["accepted"]]
    in_scope = [item for item in results if item["in_scope"]]
    accepted_correct = [item for item in accepted if item["correct"]]
    wrong_layer = [item for item in accepted if not item["layer_correct"]]
    accepted_precision = len(accepted_correct) / max(1, len(accepted))
    coverage = sum(bool(item["accepted"]) for item in in_scope) / max(1, len(in_scope))
    wrong_layer_rate = len(wrong_layer) / max(1, len(accepted))
    accepted_nce = np.asarray([item["corner_nce"] for item in accepted], dtype=np.float64)
    accepted_iou = np.asarray([item["quad_iou"] for item in accepted], dtype=np.float64)
    nce_p95 = float(np.percentile(accepted_nce, 95)) if accepted_nce.size else 1.0
    iou_median = float(np.median(accepted_iou)) if accepted_iou.size else 0.0
    iou_p05 = float(np.percentile(accepted_iou, 5)) if accepted_iou.size else 0.0
    gates = {
        "minimum_samples": independent_group_count >= gate.minimum_samples,
        "accepted_precision": accepted_precision >= gate.accepted_precision_min,
        "in_scope_coverage": coverage >= gate.in_scope_coverage_min,
        "wrong_layer_rate": wrong_layer_rate <= gate.wrong_layer_rate_max,
        "nce_p95": nce_p95 <= gate.nce_p95_max,
        "iou_median": iou_median >= gate.iou_median_min,
        "iou_p05": iou_p05 >= gate.iou_p05_min,
    }
    resolved_groups = group_ids if group_ids is not None else [
        f"sample-{index}" for index in range(sample_count)
    ]
    accepted_group_results: dict[str, list[bool]] = {}
    for group_id, item in zip(resolved_groups, results, strict=True):
        if item["accepted"]:
            accepted_group_results.setdefault(group_id, []).append(bool(item["strict_correct"]))
    accepted_group_correct = [all(values) for values in accepted_group_results.values()]
    group_failures = sum(not value for value in accepted_group_correct)
    error_interval = binomial_interval(group_failures, len(accepted_group_correct))
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "sample_count": sample_count,
        "independent_group_count": independent_group_count,
        "accepted_count": len(accepted),
        "independent_accepted_group_count": len(accepted_group_correct),
        "independent_accepted_group_failures": group_failures,
        "accepted_group_error_rate_95ci": error_interval,
        "accepted_group_error_rate_95_one_sided_upper": error_interval["one_sided_upper"],
        "supports_99_percent_precision_at_95_percent_confidence": bool(
            accepted_group_correct and error_interval["one_sided_upper"] <= 0.01
        ),
        "accepted_precision": round(accepted_precision, 8),
        "in_scope_coverage": round(coverage, 8),
        "wrong_layer_rate": round(wrong_layer_rate, 8),
        "corner_nce_p95": round(nce_p95, 8),
        "quad_iou_median": round(iou_median, 8),
        "quad_iou_p05": round(iou_p05, 8),
        "gates": gates,
        "thresholds": {
            "accepted_precision_min": gate.accepted_precision_min,
            "in_scope_coverage_min": gate.in_scope_coverage_min,
            "wrong_layer_rate_max": gate.wrong_layer_rate_max,
            "nce_p95_max": gate.nce_p95_max,
            "iou_median_min": gate.iou_median_min,
            "iou_p05_min": gate.iou_p05_min,
            "minimum_samples": gate.minimum_samples,
        },
    }


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> dict[str, float]:
    """返回精确双侧区间及单侧上界；successes 可表示错误次数。"""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("二项计数必须满足 0 <= successes <= trials")
    if trials == 0:
        return {
            "estimate": 0.0,
            "lower": 0.0,
            "upper": 1.0,
            "one_sided_upper": 1.0,
            "confidence": confidence,
        }
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    one_sided_upper = (
        1.0
        if successes == trials
        else float(beta.ppf(confidence, successes + 1, trials - successes))
    )
    return {
        "estimate": successes / trials,
        "lower": lower,
        "upper": upper,
        "one_sided_upper": one_sided_upper,
        "confidence": confidence,
    }


def corner_metrics(detected: np.ndarray, oracle: np.ndarray) -> tuple[float, float, float]:
    """在循环位移和顺逆时针排列中选择平均角点误差最低的匹配。"""

    detected_ordered = order_corners(detected)
    oracle_ordered = order_corners(oracle)
    diagonal = float(np.linalg.norm(oracle_ordered.max(axis=0) - oracle_ordered.min(axis=0)))
    best_mean = float("inf")
    best_max = float("inf")
    best_iou = 0.0
    for shift in range(4):
        rolled = np.roll(detected_ordered, shift, axis=0)
        for candidate in (rolled, rolled[::-1]):
            errors = np.linalg.norm(candidate - oracle_ordered, axis=1)
            mean_error = float(np.mean(errors))
            if mean_error < best_mean:
                best_mean = mean_error
                best_max = float(np.max(errors))
                best_iou = polygon_iou(candidate, oracle_ordered)
    return best_mean / max(diagonal, 1.0), best_iou, best_max


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """以 OpenCV 凸多边形交集计算连续坐标 IoU。"""

    first_area = abs(float(cv2.contourArea(np.asarray(first, np.float32))))
    second_area = abs(float(cv2.contourArea(np.asarray(second, np.float32))))
    intersection, _polygon = cv2.intersectConvexConvex(
        np.asarray(first, np.float32),
        np.asarray(second, np.float32),
    )
    union = first_area + second_area - float(intersection)
    return float(intersection) / max(union, 1e-8)
