"""几何严格正确率的风险、覆盖与校准统计。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def risk_coverage_report(results: list[dict[str, Any]], bins: int = 10) -> dict[str, object]:
    """只用冻结预测的 confidence 与 strict_correct 生成机器可读曲线。"""

    if not results:
        return {
            "count": 0,
            "risk_coverage": [],
            "precision_coverage": [],
            "calibration": [],
            "ece": 0.0,
            "brier": 0.0,
        }
    confidence = np.asarray([float(item.get("confidence", 0.0)) for item in results])
    correct = np.asarray([bool(item.get("strict_correct", False)) for item in results], np.float64)
    in_scope = np.asarray([bool(item.get("in_scope", False)) for item in results], bool)
    order = np.argsort(-confidence)
    curve: list[dict[str, float | int]] = []
    counts = sorted(set(np.linspace(1, len(results), min(101, len(results)), dtype=int)))
    for count in counts:
        selected = order[:count]
        precision = float(correct[selected].mean())
        coverage = float(np.sum(in_scope[selected])) / max(1, int(np.sum(in_scope)))
        curve.append(
            {
                "accepted_count": int(count),
                "threshold": float(confidence[selected[-1]]),
                "risk": 1.0 - precision,
                "precision": precision,
                "in_scope_coverage": coverage,
            }
        )
    calibration: list[dict[str, float | int]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        count = int(selected.sum())
        if not count:
            continue
        mean_confidence = float(confidence[selected].mean())
        observed = float(correct[selected].mean())
        ece += count / len(results) * abs(mean_confidence - observed)
        calibration.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "strict_accuracy": observed,
            }
        )
    return {
        "count": len(results),
        "risk_coverage": curve,
        "precision_coverage": curve,
        "calibration": calibration,
        "ece": float(ece),
        "brier": float(np.mean(np.square(confidence - correct))),
    }


def slice_report(cases: list[dict[str, object]]) -> dict[str, object]:
    """按类别、来源、设备、困难类型、难度与 scope 聚合冻结结果。"""

    dimensions = ("target_class", "source", "device", "hard_taxonomy", "difficulty", "in_scope")
    values: dict[str, dict[str, list[dict[str, Any]]]] = {
        dimension: defaultdict(list) for dimension in dimensions
    }
    for case in cases:
        metrics = case.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metadata = case.get("slice_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        for dimension in dimensions:
            value = metrics.get(dimension, metadata.get(dimension, "unknown"))
            values[dimension][str(value)].append(metrics)
    output: dict[str, object] = {}
    for dimension, groups in values.items():
        output[dimension] = {
            name: {
                "count": len(items),
                "accepted": sum(bool(item.get("accepted")) for item in items),
                "strict_correct": sum(bool(item.get("strict_correct")) for item in items),
                "refinement": {
                    outcome: sum(item.get("refinement_outcome") == outcome for item in items)
                    for outcome in ("improved", "neutral", "worsened", "rolled_back")
                },
            }
            for name, items in sorted(groups.items())
        }
    return output


__all__ = ["risk_coverage_report", "slice_report"]
