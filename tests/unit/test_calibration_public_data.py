"""公开几何校准的来源与冻结评估绑定。

运行范例：source .venv/bin/activate && which python && python -m pytest tests/unit/test_calibration_public_data.py -q
"""

from __future__ import annotations

import json

import pytest

from screenrestore.geometry.confidence import CORRECTNESS_FEATURE_NAMES


def _fixture(tmp_path):  # type: ignore[no-untyped-def]
    manifest = tmp_path / "public.geometry.jsonl"
    rows = [
        {"image": f"geometry/public/{index}.jpg", "source": "smartdoc",
         "group_id": f"public:{index}", "split": "validation"}
        for index in range(2)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    cases = [
        {
            "case": f"{index}.jpg",
            "photo": row["image"],
            "slice_metadata": {"source": row["source"]},
            "decision": {"diagnostics": {name: float(index) for name in CORRECTNESS_FEATURE_NAMES}},
            "metrics": {"strict_correct": index == 0},
        }
        for index, row in enumerate(rows)
    ]
    evaluation.write_text(
        json.dumps({"oracle_loaded_after_all_predictions": True, "cases": cases}),
        encoding="utf-8",
    )
    return manifest, evaluation


def test_public_calibration_features_are_bound_to_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.prepare_geometry_calibration import main as prepare
    from training.quadlocator.correctness_calibrator import main as fit

    manifest, evaluation = _fixture(tmp_path)
    features = tmp_path / "features.jsonl"
    assert prepare([
        "--evaluation", str(evaluation), "--manifest", str(manifest),
        "--output", str(features), "--split", "validation",
    ]) == 0
    records = [json.loads(line) for line in features.read_text().splitlines()]
    assert len(records) == 2
    assert all(row["source"] == "smartdoc" and row["manifest_sha256"] for row in records)
    output = tmp_path / "calibrator.json"
    assert fit([
        "--input", str(features), "--manifest", str(manifest), "--output", str(output),
    ]) == 0
    assert output.is_file()


def test_calibration_rejects_mismatched_or_private_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.prepare_geometry_calibration import main as prepare
    from training.quadlocator.correctness_calibrator import main as fit

    manifest, evaluation = _fixture(tmp_path)
    features = tmp_path / "features.jsonl"
    data = json.loads(evaluation.read_text())
    data["cases"][0]["slice_metadata"]["source"] = "private-development"
    evaluation.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="来源与公开校准清单不一致"):
        prepare([
            "--evaluation", str(evaluation), "--manifest", str(manifest),
            "--output", str(features), "--split", "validation",
        ])
    assert not features.exists()
    # 即使传入公开清单，拟合入口也要拒绝与清单来源不一致的特征行。
    public_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    data["cases"][0]["slice_metadata"]["source"] = "smartdoc"
    evaluation.write_text(json.dumps(data), encoding="utf-8")
    assert prepare([
        "--evaluation", str(evaluation), "--manifest", str(manifest),
        "--output", str(features), "--split", "validation",
    ]) == 0
    feature_rows = [json.loads(line) for line in features.read_text().splitlines()]
    feature_rows[0]["source"] = "private-development"
    features.write_text("".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="特征与公开清单"):
        fit([
            "--input", str(features), "--manifest", str(manifest),
            "--output", str(tmp_path / "calibrator.json"),
        ])
    public_rows[0]["source"] = "private-development"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in public_rows), encoding="utf-8")
    with pytest.raises(ValueError, match="引用私人数据"):
        fit([
            "--input", str(features), "--manifest", str(manifest),
            "--output", str(tmp_path / "calibrator.json"),
        ])


def test_acceptance_validation_rejects_private_manifest_before_output(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.audit_p4_acceptance import main as audit

    manifest, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["source"] = "private-development"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "acceptance"
    with pytest.raises(ValueError, match="引用私人数据"):
        audit([
            "--mode", "validation", "--checkpoint", str(tmp_path / "missing.pt"),
            "--manifest", str(manifest), "--dataset-root", str(tmp_path),
            "--output-directory", str(output),
        ])
    assert not output.exists()


def test_checkpoint_selection_rejects_private_calibration_before_output(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from scripts.evaluate_p4_geometry_trajectory import main as select

    manifest, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["source"] = "private-development"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "selection"
    with pytest.raises(ValueError, match="引用私人数据"):
        select([
            "--baseline", str(tmp_path / "missing-baseline.pt"),
            "--checkpoints", str(tmp_path / "missing-candidate.pt"),
            "--internal-manifest", str(manifest),
            "--calibration-manifest", str(manifest),
            "--smartdoc-manifest", str(manifest),
            "--dataset-root", str(tmp_path),
            "--output-directory", str(output),
        ])
    assert not output.exists()
