"""P4-G3.7 隔离、坐标、readiness 与诊断实验契约回归。

运行：source .venv/bin/activate && which python && python -m pytest -q tests/unit/test_p4_g37.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.audit_p4_dataset_isolation import (
    identity_rows,
    load_test_identity_index,
    overlap,
    read_development,
)
from scripts.prepare_p4_target_domain import (
    annotation_sha256,
    group_partition,
    leakage_errors,
    render_overlays,
    validate_target,
)
from scripts.run_p4_g37 import STAGES, comparable_config


def test_development_loader_never_materializes_test_gt(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"split":"test","content_quad": FORBIDDEN\n'
        + json.dumps({"split": "validation", "image": "a"})
        + "\n"
    )
    assert read_development(path) == [{"split": "validation", "image": "a"}]
    with pytest.raises(ValueError, match="no-test-access"):
        read_development(path, "test")
    assert "test" not in STAGES and "all" not in STAGES


def test_overlap_checks_bytes_normalized_path_and_source_family(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"identical image")
    (tmp_path / "b.png").write_bytes(b"identical image")
    first = {
        "image": "./a.png",
        "digital_source_id": "digital-A",
        "subject_id": "s1",
        "group_id": "g1",
        "capture_session": "c1",
        "split": "train",
    }
    second = {
        **first,
        "image": "b.png",
        "subject_id": "s2",
        "group_id": "g2",
        "capture_session": "c2",
        "split": "validation",
    }
    report = overlap(identity_rows([first], tmp_path), identity_rows([second], tmp_path))
    assert report["image_leakage"] and report["source_family_leakage"]
    assert not report["group_leakage"] and not report["capture_session_leakage"]
    assert report["intersection"]["normalized_path"] == 0
    alias = identity_rows([{**first, "image": "a.png"}], tmp_path)
    assert overlap(identity_rows([first], tmp_path), alias)["intersection"]["normalized_path"] == 1


@pytest.mark.parametrize(
    "field", ["digital_source_id", "subject_id", "group_id", "capture_session"]
)
def test_identifier_leakage_cannot_cross_split(field: str) -> None:
    a = {
        "digital_source_id": "a",
        "subject_id": "a",
        "group_id": "a",
        "capture_session": "a",
        "split": "train",
    }
    b = {key: "b" for key in a}
    b.update({field: "a", "split": "validation"})
    assert f"{field}_leakage" in leakage_errors([a, b])


def test_source_family_must_have_one_group_even_inside_validation() -> None:
    a = {
        "digital_source_id": "a",
        "subject_id": "a",
        "group_id": "a",
        "capture_session": "a",
        "split": "validation",
    }
    assert "digital_source_id_leakage" in leakage_errors([a, {**a, "group_id": "b"}])


def test_group_partition_is_order_independent() -> None:
    ids = [f"group-{i}" for i in range(100)]
    a = {g: group_partition(g) for g in ids}
    b = {g: group_partition(g) for g in reversed(ids)}
    assert a == b
    assert set(a.values()) == {"fit", "selection", "evaluation"}
    with pytest.raises(ValueError):
        group_partition("")


def test_readiness_uses_groups_and_requires_isolation(tmp_path: Path) -> None:
    result = validate_target([], tmp_path)
    assert result["status"] == "BLOCKED: insufficient_target_domain_groups"
    assert result["independent_groups_by_domain"] == dict.fromkeys(
        ("postcard", "artwork", "screen", "poster"), 0
    )
    assert "test_isolation_unverified" in result["reasons"]
    # 复制相同记录再多次，也只计一个 group，不能拿图片量代替独立作品。
    records = [
        {
            "image": "missing.png",
            "domain": "artwork",
            "group_id": "one",
            "subject_id": "one",
            "digital_source_id": "one",
            "capture_session": "one",
            "split": "validation",
            "present": False,
        }
    ] * 1000
    result = validate_target(records, tmp_path)
    assert result["independent_groups_by_domain"]["artwork"] == 1
    assert result["status"].startswith("BLOCKED")


def test_test_index_rejects_paths_and_gt(tmp_path: Path) -> None:
    assert load_test_identity_index(None) is None
    p = tmp_path / "index.json"
    p.write_text(
        json.dumps(
            {
                "kind": "geometry_identity_hash_index",
                "split": "test",
                "rows": [{"image": "secret", "content_quad": []}],
            }
        )
    )
    with pytest.raises(ValueError, match="only identity hashes"):
        load_test_identity_index(p)


def test_overlay_pending_and_annotation_change_invalidates_review(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "图.png"
    Image.new("RGB", (64, 48)).save(image)
    record = {
        "image": image.name,
        "split": "validation",
        "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "outer_quad": None,
    }
    output = tmp_path / "overlays"
    render_overlays([record], tmp_path, output)
    review = json.loads((output / "review.jsonl").read_text())
    assert review["status"] == "pending"
    assert review["annotation_sha256"] == annotation_sha256(record)
    assert annotation_sha256({**record, "overlay_review": review}) == annotation_sha256(record)
    assert annotation_sha256({**record, "content_quad": None}) != annotation_sha256(record)
    with pytest.raises(FileExistsError):
        render_overlays([record], tmp_path, output)


def test_normalized_and_pixel_nce_contract() -> None:
    from scripts.audit_p4_acceptance import _stage_metrics
    from training.quadlocator.metrics import _corner_nce

    quad = np.array([[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]], np.float32)
    delta = np.array([0.02, 0], np.float32)
    expected = 0.02 / np.hypot(0.4, 0.4)
    assert _corner_nce(quad + delta, quad) == pytest.approx(expected)
    result = _stage_metrics((quad + delta) * 1000, quad, np.array([1000, 1000]))
    assert result["corner_nce"] == pytest.approx(expected)


def test_policy_denominator_and_shape() -> None:
    from scripts.audit_p4_acceptance import _binary_policy_metrics

    rows = [
        {"strict_correct": v, "in_scope": scope}
        for v, scope in [(True, True), (False, True), (True, True), (False, False)]
    ]
    m = _binary_policy_metrics(rows, np.array([True, True, False, False]))
    assert m["sample_count"] == 4
    assert m["precision"] == m["recall"] == 0.5
    assert m["coverage"] == pytest.approx(2 / 3)
    assert [
        m[k] for k in ["true_positive", "false_positive", "false_negative", "true_negative"]
    ] == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        _binary_policy_metrics(rows, np.ones((4, 1), bool))


def test_native512_config_reproduces_only_resolution_and_duration() -> None:
    control = {
        "image_size": 256,
        "learning_rate": 1e-6,
        "loss_profile": "content_coordinate_only",
        "trainable_scope": "content_head",
        "train_augmentation": "full",
        "seed": 20260902,
        "batch_size": 16,
        "hard_sampling": False,
        "validation_split": "validation",
        "train_samples": 5000,
        "validation_samples": 1000,
        "scheduler_t_max": 4,
    }
    config = comparable_config(control)
    assert config == {**control, "image_size": 512, "epochs": 1}
    for key, value in [
        ("loss_profile", "content_only"),
        ("train_augmentation", "none"),
        ("learning_rate", 2e-6),
    ]:
        with pytest.raises(ValueError, match="missing_comparable_256_arm"):
            comparable_config({**control, key: value})


def test_no_eligible_challenger_preserves_b0() -> None:
    from scripts.evaluate_p4_geometry_trajectory import DATASET_ORDER, select_checkpoint

    base = {
        "corner_nce_median": 0.05,
        "corner_nce_p95": 0.1,
        "quad_iou_median": 0.9,
        "quad_iou_p05": 0.8,
    }
    reports = {
        name: {
            "checkpoints": [
                {"label": "B0", "metrics": base},
                {"label": "epoch-001", "metrics": {**base, "quad_iou_median": 0.7}},
            ]
        }
        for name in DATASET_ORDER
    }
    result = select_checkpoint(reports, dict(zip(base, [0.002, 0.005, 0.01, 0.01], strict=True)))
    assert result["status"] == "NO_ELIGIBLE_CHECKPOINT"
    assert result["winner"] is None and result["baseline"] == "B0"


def test_training_metric_matches_frozen_corner_permutation() -> None:
    from training.quadlocator.metrics import _corner_nce, _quad_iou

    quad = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], np.float32)
    # 四个 heatmap channel 错位也必须按冻结 evaluator 的对应规则报告几何，
    # channel loss 仍保持原监督，不改变训练目标。
    assert _corner_nce(np.roll(quad, 1, axis=0), quad) == pytest.approx(0)
    assert _corner_nce(quad[::-1], quad) == pytest.approx(0)
    assert _quad_iou(quad[[0, 2, 1, 3]], quad) == pytest.approx(1)


def test_training_validation_clips_letterbox_padding() -> None:
    import torch
    from training.quadlocator.losses import _softargmax_corners
    from training.quadlocator.metrics import ValidationMetrics, _corner_nce

    logits = torch.full((1, 4, 32, 32), -10.0)
    for i, (x, y) in enumerate([(6, 2), (25, 2), (25, 29), (6, 29)]):
        logits[0, i, y, x] = 10
    target = torch.tensor([[[0.2, 0.25], [0.8, 0.25], [0.8, 0.75], [0.2, 0.75]]])
    outputs = {
        "content_corner_heatmaps": logits,
        "outer_corner_heatmaps": logits,
        "presence_logits": torch.ones(1, 1),
        "outer_presence_logits": torch.zeros(1, 1),
        "class_logits": torch.zeros(1, 4),
        "content_mask_logits": torch.zeros(1, 1, 32, 32),
        "boundary_logits": torch.zeros(1, 1, 32, 32),
    }
    targets = {
        "content_corners": target,
        "outer_corners": target,
        "presence": torch.ones(1, 1),
        "outer_present": torch.zeros(1, 1),
        "target_class": torch.zeros(1, dtype=torch.long),
        "content_mask": torch.zeros(1, 1, 32, 32),
        "boundary": torch.zeros(1, 1, 32, 32),
        "image_bounds": torch.tensor([[0.0, 0.25, 1.0, 0.75]]),
    }
    accumulator = ValidationMetrics()
    accumulator.update(outputs, targets)
    predicted = _softargmax_corners(logits).numpy()[0]
    expected = np.clip(predicted, np.array([0, 0.25], np.float32), np.array([1, 0.75], np.float32))
    assert accumulator.content_nce[0] == pytest.approx(_corner_nce(expected, target.numpy()[0]))
    assert accumulator.content_nce[0] < _corner_nce(predicted, target.numpy()[0])


def test_review_manifest_keeps_human_pending_state(tmp_path: Path) -> None:
    from scripts.prepare_p4_target_domain import build_reviewed_manifest

    record = {"image": "a", "split": "validation"}
    review = {
        "annotation_sha256": annotation_sha256(record),
        "image_sha256": "a" * 64,
        "reviewer": "",
        "reviewed_at": "",
        "status": "pending",
    }
    output = tmp_path / "reviewed.jsonl"
    build_reviewed_manifest([record], [review], output)
    assert json.loads(output.read_text())["overlay_review"]["status"] == "pending"
    with pytest.raises(FileExistsError):
        build_reviewed_manifest([record], [review], output)


def test_acceptance_release_requires_precision_and_coverage() -> None:
    from scripts.p4_acceptance_development import release_gate

    rows = [
        {
            "group_id": f"g{i}",
            "strict_correct": True,
            "in_scope": True,
            "wrong_layer": False,
            "stages": {"rollback": {"corner_nce": 0.001, "quad_iou": 0.99}},
        }
        for i in range(100)
    ]
    assert release_gate(rows, np.ones(100, bool))["passed"]
    low_coverage = release_gate(rows, np.arange(100) < 1)
    assert low_coverage["policy"]["precision"] == 1
    assert not low_coverage["passed"]
    rows[0]["strict_correct"] = False
    rows[1]["strict_correct"] = False
    assert not release_gate(rows, np.ones(100, bool))["passed"]


def test_acceptance_consumed_source_cannot_be_renamed_and_reused(tmp_path: Path) -> None:
    from argparse import Namespace

    from scripts.audit_p4_dataset_isolation import token
    from scripts.p4_acceptance_development import evaluate_development

    group = next(f"g{i}" for i in range(100) if group_partition(f"g{i}") == "evaluation")
    locks = tmp_path / "p4-g37-evaluation-locks"
    locks.mkdir()
    (locks / f"{token('source-a')}.json").write_text("{}")
    with pytest.raises(ValueError, match="evaluation_already_consumed"):
        evaluate_development(
            [{"group_id": group, "digital_source_id": "source-a", "image": "renamed.png"}],
            Namespace(run_root=tmp_path),
            tmp_path / "unused.pt",
            tmp_path,
        )


def test_target_readiness_accepts_complete_independent_reviewed_groups(tmp_path: Path) -> None:
    from PIL import Image
    from scripts.audit_p4_dataset_isolation import sha256
    from scripts.prepare_p4_target_domain import CONDITIONS

    records = []
    for domain in ("postcard", "artwork", "screen", "poster"):
        for group_index in range(8):
            group = f"{domain}-{group_index}"
            for i in range(15):
                name = f"{group}-{i}.png"
                Image.new("RGB", (8, 8), (len(records) // 256, len(records) % 256, 10)).save(
                    tmp_path / name
                )
                positive = i < 12
                row = {
                    "image": name,
                    "split": "validation",
                    "domain": domain,
                    "digital_source_id": group,
                    "source": "authorized-public-capture",
                    "source_license": "photographer-authorized-training",
                    "subject_id": group,
                    "group_id": group,
                    "capture_session": f"{group}-s{i // 6 if positive else 0}",
                    "present": positive,
                    "target_class": ("artwork" if domain == "poster" else domain)
                    if positive
                    else "none",
                    "content_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
                    if positive
                    else None,
                    "outer_quad": None,
                    "in_scope": positive,
                    "visible": positive,
                    "occlusion": 0.0,
                    "glare_level": "none",
                    "device": "synthetic-test",
                    "scene_type": domain if positive else "partial",
                    "capture_conditions": sorted(CONDITIONS),
                }
                row["overlay_review"] = {
                    "status": "approved",
                    "reviewer": "test-fixture",
                    "reviewed_at": "2026-09-12",
                    "image_sha256": sha256(tmp_path / name),
                    "annotation_sha256": annotation_sha256(row),
                }
                records.append(row)
    isolation = {
        "test_isolation": "verified_index",
        "slices": {"target_development": {}},
        "overlap": {
            f"{name}__target_development": {
                "image_leakage": False,
                "source_family_leakage": False,
                "group_leakage": False,
                "capture_session_leakage": False,
                "source_family_verified": True,
            }
            for name in (
                "stage_b_train",
                "internal_validation",
                "calibration",
                "smartdoc_validation",
                "smartdoc_test",
            )
        },
    }
    result = validate_target(records, tmp_path, isolation)
    assert result["status"] == "READY", result["reasons"]
    # 原图字节改变使审核失效，不能沿用旧 approved。
    (tmp_path / records[0]["image"]).write_bytes(b"changed")
    result = validate_target(records, tmp_path, isolation)
    assert "overlay_not_reviewed_or_stale" in result["reasons"]


def test_runner_fallback_does_not_select_unapproved_checkpoint(tmp_path: Path) -> None:
    from scripts.run_p4_g37 import resolve_incumbent

    b0 = tmp_path / "b0.pt"
    assert resolve_incumbent(tmp_path, b0) == b0
    stage = tmp_path / "sanity512"
    stage.mkdir()
    (stage / "result.json").write_text(json.dumps({"status": "NO_ELIGIBLE_CHECKPOINT"}))
    assert resolve_incumbent(tmp_path, b0) == b0
    (stage / "result.json").write_text(json.dumps({"status": "RUNNING"}))
    with pytest.raises(ValueError, match="unknown geometry freeze state"):
        resolve_incumbent(tmp_path, b0)


def test_acceptance_development_rejects_private_records_before_fit(tmp_path: Path) -> None:
    from scripts.p4_acceptance_development import evaluate_development

    with pytest.raises(ValueError, match="引用私人数据"):
        evaluate_development(
            [{"image": "private-validation/photo.jpg", "group_id": "group-01"}],
            object(),
            tmp_path / "missing.pt",
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_acceptance_freezes_selection_before_single_evaluation(tmp_path: Path, monkeypatch) -> None:
    from argparse import Namespace

    from scripts import p4_acceptance_development as development

    from screenrestore.geometry.confidence import CORRECTNESS_FEATURE_NAMES

    records, by_path = [], {}
    output = tmp_path / "result"
    output.mkdir()
    events = []
    for partition in ("fit", "selection", "evaluation"):
        group = next(f"group-{i}" for i in range(100) if group_partition(f"group-{i}") == partition)
        for i in range(2):
            image = f"{partition}-{i}.png"
            record = {
                "group_id": group,
                "digital_source_id": group,
                "image": image,
                "split": "validation",
            }
            records.append(record)
            by_path[tmp_path / image] = (record, i == 0)

    def scheduled(manifest, root, split):
        return tuple(root / json.loads(line)["image"] for line in manifest.read_text().splitlines())

    def infer(checkpoint, paths, **kwargs):
        partition = paths[0].name.split("-")[0]
        events.append(partition)
        if partition == "evaluation":
            assert (output / "frozen-policy.json").exists()
        result = {}
        for path in paths:
            record, correct = by_path[path]
            result[path] = {
                **record,
                "has_candidate": True,
                "area_in_scope": True,
                "complete_features": True,
                "in_scope": True,
                "strict_correct": correct,
                "features": dict.fromkeys(CORRECTNESS_FEATURE_NAMES, float(correct)),
                "hard_accepted": correct,
                "hard_without_margin_accepted": correct,
                "hard_without_boundary_accepted": correct,
                "stages": {
                    "rollback": {
                        "corner_nce": 0.001 if correct else 0.1,
                        "quad_iou": 0.99 if correct else 0.5,
                    }
                },
            }
        return result

    monkeypatch.setattr(development.audit, "_scheduled_paths", scheduled)
    monkeypatch.setattr(development.audit, "_infer_acceptance_features", infer)
    monkeypatch.setattr(development.audit, "_truth_by_path", lambda *a: dict.fromkeys(by_path, {}))
    monkeypatch.setattr(development.audit, "_score_case", lambda prediction, truth: prediction)
    args = Namespace(run_root=tmp_path, data_root=tmp_path, device="cpu")
    result = development.evaluate_development(records, args, tmp_path / "unused.pt", output)
    assert events == ["fit", "selection", "evaluation"]
    assert result["evaluation"]["policy"]["sample_count"] == 2
    assert result["status"] == "CALIBRATION_FAILURE"
    assert result["runtime_changed"] is False
    with pytest.raises(ValueError, match="evaluation_already_consumed"):
        development.evaluate_development(records, args, tmp_path / "unused.pt", output)


def test_report_generation_uses_only_stage_results(tmp_path: Path) -> None:
    from scripts.report_p4_g37 import render_reports

    render_reports(
        {"sanity512": {"status": "BLOCKED", "reason": "missing_comparable_256_arm"}}, tmp_path
    )
    assert "BLOCKED" in (tmp_path / "P4_G37_RESULTS.md").read_text()
