#!/usr/bin/env python3
"""根据完整 B0 parity 与 G1 同口径结果冻结 geometry FULL gate。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/evaluate_p4_geometry_gate.py \
      --b0-summary /runs/p4/geometry-b0-parity/summary.json \
      --g1-summary /runs/p4/geometry-g1/evaluate-best-geometry/summary.json \
      --g1-initial /runs/p4/geometry-g1/initial_validation.json \
      --g1-history /runs/p4/geometry-g1/history.json \
      --output /runs/p4/geometry-full-gate.json

返回码 0 表示 PASS，3 表示 BLOCKED。输出只依据已存在实验产物，不修改 checkpoint。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-summary", type=Path, required=True)
    parser.add_argument("--g1-summary", type=Path, required=True)
    parser.add_argument("--g1-initial", type=Path, required=True)
    parser.add_argument("--g1-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 FULL gate：{output}")

    b0 = _read(args.b0_summary)
    g1 = _read(args.g1_summary)
    initial = _read(args.g1_initial)["validation_metrics"]
    history = _read(args.g1_history)
    b0_coarse = b0["summaries"]["decoder_v2"]["coarse"]
    g1_variant = g1["summaries"]["decoder_v2"]
    g1_coarse = g1_variant["coarse"]
    final = g1_variant["final_decision"]
    best_validation = max(
        (record["validation_metrics"] for record in history),
        key=lambda metrics: _balanced_geometry_score(metrics),
    )

    parity_max = max(float(values["max_abs"]) for values in b0["torch_onnx_parity"].values())
    reference = b0.get("reference_parity") or {}
    gates = {
        "legacy_evaluator_parity": (
            float(reference.get("legacy_coarse_nce_max_abs_delta", 1.0)) <= 1e-8
            and float(reference.get("legacy_coarse_iou_max_abs_delta", 1.0)) <= 1e-8
        ),
        "torch_onnx_parity": parity_max <= 1e-3,
        "mixed_validation_nce_p95_non_regression": float(best_validation["content_corner_nce_p95"])
        <= float(initial["content_corner_nce_p95"]),
        "mixed_validation_iou_p05_non_regression": float(best_validation["content_iou_p05"])
        >= float(initial["content_iou_p05"]),
        "smartdoc_nce_median_improved": float(g1_coarse["corner_nce_median"])
        < float(b0_coarse["corner_nce_median"]),
        "smartdoc_nce_p95_improved": float(g1_coarse["corner_nce_p95"])
        < float(b0_coarse["corner_nce_p95"]),
        "smartdoc_iou_median_improved": float(g1_coarse["quad_iou_median"])
        > float(b0_coarse["quad_iou_median"]),
        "smartdoc_iou_p05_improved": float(g1_coarse["quad_iou_p05"])
        > float(b0_coarse["quad_iou_p05"]),
        "development_precision": float(final["accepted_precision"]) >= 0.95,
        "development_coverage": float(final["in_scope_coverage"]) >= 0.50,
    }
    report = {
        "status": "PASS" if all(gates.values()) else "BLOCKED",
        "reason": (
            "G1 已证明 content geometry 可稳定恢复，可继续受控后续消融"
            if all(gates.values())
            else "G1 未同时改善 mixed validation 与 SmartDoc raw geometry，禁止 geometry FULL"
        ),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "parity_max_abs": parity_max,
        "b0_decoder_v2_coarse": b0_coarse,
        "g1_decoder_v2_coarse": g1_coarse,
        "g1_final_decision": final,
        "g1_initial_validation": initial,
        "g1_best_balanced_validation": best_validation,
        "inputs": {
            "b0_summary": str(args.b0_summary.expanduser().resolve()),
            "g1_summary": str(args.g1_summary.expanduser().resolve()),
            "g1_initial": str(args.g1_initial.expanduser().resolve()),
            "g1_history": str(args.g1_history.expanduser().resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if report["status"] == "PASS" else 3


def _balanced_geometry_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    nce_p95 = float(metrics["content_corner_nce_p95"])
    iou_p05 = float(metrics["content_iou_p05"])
    return (
        (1.0 - nce_p95) + iou_p05,
        float(metrics["content_iou_median"]),
        float(metrics["content_strict_correct_rate"]),
        -nce_p95,
    )


def _read(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
