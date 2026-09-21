"""P4 独立开发域 acceptance evidence，fit/selection/evaluation 单向流动。

使用范例：bash scripts/run_p4_g37.sh acceptance-dev
由已通过 readiness 的阶段入口调用；选定 policy 落盘后才读取 evaluation GT。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scripts import audit_p4_acceptance as audit
from scripts.audit_p4_dataset_isolation import token
from scripts.prepare_p4_target_domain import group_partition
from training.quadlocator.correctness_calibrator import fit_calibrator
from training.quadlocator.train import _assert_public_training_record

from screenrestore.geometry.confidence import CORRECTNESS_FEATURE_NAMES

BOUNDARY_FEATURES = tuple(
    name
    for name in CORRECTNESS_FEATURE_NAMES
    if any(
        word in name
        for word in ("boundary", "residual", "coverage", "alignment", "refine", "drift")
    )
)
FEATURE_SETS = {
    "combined": tuple(CORRECTNESS_FEATURE_NAMES),
    "boundary_refinement": BOUNDARY_FEATURES,
    "without_margin": tuple(
        n for n in CORRECTNESS_FEATURE_NAMES if n != "corner_peak_difference_min"
    ),
    "without_boundary_refinement": tuple(
        n for n in CORRECTNESS_FEATURE_NAMES if n not in BOUNDARY_FEATURES
    ),
    "without_entropy_sharpness": tuple(
        n for n in CORRECTNESS_FEATURE_NAMES if not any(w in n for w in ("entropy", "sharpness"))
    ),
}
HARD_POLICIES = {
    "current_hard_gate": "hard_accepted",
    "without_absolute_margin_veto": "hard_without_margin_accepted",
    "without_boundary_veto": "hard_without_boundary_accepted",
}


def release_gate(rows: list[dict], accepted: np.ndarray) -> dict:
    metrics = audit._binary_policy_metrics(rows, accepted)
    selected = [r for r, flag in zip(rows, accepted, strict=True) if flag]
    nce = [r["stages"]["rollback"]["corner_nce"] for r in selected]
    iou = [r["stages"]["rollback"]["quad_iou"] for r in selected]
    # outer GT 缺失时不能把 wrong-layer 计为已知零；严格门保持关闭。
    wrong_layer = sum(r.get("wrong_layer", True) for r in selected) / max(1, len(selected))
    geometry = {
        "nce_p95": float(np.percentile(nce, 95)) if nce else 1.0,
        "iou_median": float(np.median(iou)) if iou else 0.0,
        "iou_p05": float(np.percentile(iou, 5)) if iou else 0.0,
    }
    gates = {
        "precision": metrics["precision"] >= 0.99,
        "coverage": metrics["coverage"] >= 0.9,
        "wrong_layer": wrong_layer < 0.005,
        "nce": geometry["nce_p95"] <= 0.01,
        "iou_median": geometry["iou_median"] >= 0.97,
        "iou_p05": geometry["iou_p05"] >= 0.93,
        "release_groups": len({r["group_id"] for r in rows}) >= 100,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "policy": metrics,
        "wrong_layer_rate": wrong_layer,
        "geometry": geometry,
    }


def evaluate_development(records: list[dict], args: object, checkpoint: Path, output: Path) -> dict:
    # 本阶段会拟合/选择 policy，因此只接受公开目标域记录。
    for line_number, record in enumerate(records, 1):
        _assert_public_training_record(record, line_number)
    partitions = {
        p: [r for r in records if group_partition(r["group_id"]) == p]
        for p in ("fit", "selection", "evaluation")
    }
    # 身份级单次锁放在 run root，换输出目录也不能重新使用 evaluation。
    identity = token(json.dumps(sorted({r["digital_source_id"] for r in records})))
    locks = args.run_root / "p4-g37-evaluation-locks"
    locks.mkdir(exist_ok=True)
    evaluation_locks = [
        locks / f"{token(source)}.json"
        for source in sorted({r["digital_source_id"] for r in partitions["evaluation"]})
    ]
    if any(lock.exists() for lock in evaluation_locks):
        raise ValueError("evaluation_already_consumed")

    def rows_for(partition: str) -> list[dict]:
        manifest = output / f"{partition}.geometry.jsonl"
        with manifest.open("x") as handle:
            for r in partitions[partition]:
                handle.write(json.dumps(r) + "\n")
        paths = audit._scheduled_paths(manifest, args.data_root, "validation")
        predictions = audit._infer_acceptance_features(
            checkpoint, paths, device=torch.device(args.device), batch_size=8
        )
        truths = audit._truth_by_path(manifest, args.data_root, "validation")
        rows = [audit._score_case(predictions[p], truths[p]) for p in paths]
        for row, record, path in zip(rows, partitions[partition], paths, strict=True):
            # wrong-layer：更接近 outer 且不满足 content strict 几何。缺少 outer 时只支持
            # content 精确匹配的已知正确样本，其他接受样本保守计错。
            stage = row["stages"]["rollback"]
            content_correct = stage["corner_nce"] <= 0.01 and stage["quad_iou"] >= 0.93
            outer = record.get("outer_quad")
            if outer is not None:
                m = audit._stage_metrics(
                    predictions[path]["rollback"],
                    np.asarray(outer, np.float32),
                    predictions[path]["image_scale"],
                )
                row["wrong_layer"] = not content_correct and m["quad_iou"] > stage["quad_iou"]
            else:
                row["wrong_layer"] = not content_correct
            row["strict_correct"] = row["strict_correct"] and row["in_scope"]
        return rows

    fit, selection = rows_for("fit"), rows_for("selection")
    models, candidates, comparison = {}, {}, {}
    for name, field in HARD_POLICIES.items():
        flags = np.asarray([r[field] for r in selection], bool)
        candidates[name] = flags
        comparison[name] = audit._binary_policy_metrics(selection, flags)
    usable = [r for r in fit if r["complete_features"]]
    if not usable or len({r["strict_correct"] for r in usable}) < 2:
        return {"status": "BLOCKED: insufficient_fit_class_support"}
    for name, features in FEATURE_SETS.items():
        model = fit_calibrator(
            [{k: r["features"][k] for k in features} for r in usable],
            [r["strict_correct"] for r in usable],
            manifest_sha256=identity,
            minimum_precision=0.99,
        )
        probabilities = np.asarray([model.predict_probability(r["features"]) for r in selection])
        eligible = np.asarray(
            [
                r["complete_features"] and r["has_candidate"] and r["area_in_scope"]
                for r in selection
            ]
        )
        threshold = audit._select_threshold(
            probabilities,
            np.asarray([r["strict_correct"] for r in selection]),
            0.99,
            eligible=eligible,
        )
        models[name] = (model, threshold)
        candidates[name] = eligible & (probabilities >= threshold)
        comparison[name] = audit._binary_policy_metrics(selection, candidates[name])
    eligible_names = [
        n for n, m in comparison.items() if m["precision"] >= 0.99 and m["accepted_count"] > 0
    ]
    if not eligible_names:
        (output / "selection.json").write_text(json.dumps(comparison, indent=2) + "\n")
        return {"status": "CALIBRATION_FAILURE: no_selection_policy", "evaluation_consumed": False}
    # 只以 selection coverage 最大化，名称作确定性 tie-breaker；evaluation 不参与消融。
    chosen = min(eligible_names, key=lambda n: (-comparison[n]["coverage"], n))
    frozen = {
        "policy": chosen,
        "selection_comparison": comparison,
        "identity": identity,
        "model": models[chosen][0].to_dict() if chosen in models else None,
        "threshold": models[chosen][1] if chosen in models else None,
    }
    (output / "frozen-policy.json").write_text(json.dumps(frozen, indent=2) + "\n")
    for lock in evaluation_locks:
        with lock.open("x") as handle:
            json.dump({"status": "evaluation_consumed", "policy": chosen}, handle)
    evaluation = rows_for("evaluation")
    if chosen in models:
        model, threshold = models[chosen]
        accepted = np.asarray(
            [
                r["complete_features"]
                and r["has_candidate"]
                and r["area_in_scope"]
                and model.predict_probability(r["features"]) >= threshold
                for r in evaluation
            ]
        )
    else:
        accepted = np.asarray([r[HARD_POLICIES[chosen]] for r in evaluation])
    gate = release_gate(evaluation, accepted)
    return {
        "status": "G4_CANDIDATE_REVIEW_REQUIRED" if gate["passed"] else "CALIBRATION_FAILURE",
        "selected_policy": chosen,
        "selection_comparison": comparison,
        "evaluation": gate,
        "evaluation_consumed": True,
        "runtime_changed": False,
    }
