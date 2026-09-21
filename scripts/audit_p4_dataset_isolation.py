#!/usr/bin/env python3
"""P4 开发数据身份审计；不打开 test 图片或反序列化 test GT。

用法：source .venv/bin/activate && which python
python scripts/audit_p4_dataset_isolation.py --data-root /data --output-directory /runs/isolation
只输出聚合计数；缺失 source 身份保持 unknown，禁止用 group 猜测 source。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.geometry_e2e.run import _project_json_string  # noqa: E402
from training.quadlocator.train import _assert_public_training_manifest  # noqa: E402

IDENTIFIERS = ("digital_source_id", "subject_id", "group_id", "capture_session")
IDENTITY_FIELDS = ("image_path", "normalized_path", "image_sha256", *IDENTIFIERS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token(value: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", value).encode()).hexdigest()


def read_development(path: Path, split: str | None = None) -> list[dict]:
    """只投影 split 后物化开发记录；test 行永远不交给 JSON decoder。"""
    if split not in (None, "train", "validation"):
        raise ValueError("no-test-access: forbidden split")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            actual = _project_json_string(line, "split")
            if actual == "test":
                continue
            if actual not in ("train", "validation"):
                raise ValueError("invalid development split")
            if split is None or actual == split:
                records.append(json.loads(line))
    return records


def identity_rows(records: list[dict], root: Path, cache: dict | None = None) -> list[dict]:
    cache = {} if cache is None else cache
    rows = []
    for index, record in enumerate(records):
        raw = str(record["image"])
        path = (root / raw).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("image outside dataset root")
        if path not in cache:
            cache[path] = sha256(path) if path.is_file() else None
        rows.append(
            {
                "image_path": token(raw),
                "normalized_path": token(str(path)),
                "image_sha256": cache[path],
                **{
                    key: token(str(record[key])) if record.get(key) else None for key in IDENTIFIERS
                },
                **{
                    key: record.get(key, "unknown")
                    for key in ("split", "target_class", "scene_type")
                },
            }
        )
        if (index + 1) % 1000 == 0 or index + 1 == len(records):
            print(f"[审计 {index + 1}/{len(records)}]", file=sys.stderr)
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "sample_count": len(rows),
        "independent": {key: len({r[key] for r in rows if r.get(key)}) for key in IDENTIFIERS},
        "missing": {key: sum(not r.get(key) for r in rows) for key in IDENTITY_FIELDS},
        "distribution": {
            key: dict(Counter(r[key] for r in rows))
            for key in ("split", "target_class", "scene_type")
        },
    }


def overlap(first: list[dict], second: list[dict]) -> dict:
    counts = {
        key: len({r[key] for r in first if r.get(key)} & {r[key] for r in second if r.get(key)})
        for key in IDENTITY_FIELDS
    }
    return {
        "intersection": counts,
        "image_leakage": any(counts[k] for k in IDENTITY_FIELDS[:3]),
        "source_family_leakage": counts["digital_source_id"] > 0,
        "group_leakage": counts["group_id"] > 0,
        "capture_session_leakage": counts["capture_session"] > 0,
        "source_family_verified": all(r.get("digital_source_id") for r in first + second)
        and bool(first and second),
    }


def load_test_identity_index(path: Path | None) -> list[dict] | None:
    """仅消费预先生成的 hash-only 索引；不提供从 test 生成索引的入口。"""
    if path is None:
        return None
    value = json.loads(path.read_text())
    if value.get("kind") != "geometry_identity_hash_index" or value.get("split") != "test":
        raise ValueError("invalid hash-only test index")
    rows = value.get("rows", [])
    if not rows:
        raise ValueError("empty test identity index")
    for row in rows:
        if set(row) != set(IDENTITY_FIELDS):
            raise ValueError("test identity index must contain only identity hashes")
        if any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in row.values()
        ):
            raise ValueError("incomplete test identity index")
    return [
        {**r, "split": "test", "target_class": "withheld", "scene_type": "withheld"} for r in rows
    ]


def audit(root: Path, target: Path | None = None, test_index: Path | None = None) -> dict:
    manifests = {
        "stage_b_train": root / "manifests/p2-public/stage-b.geometry.jsonl",
        "internal_validation": root / "manifests/p2-public/stage-b.geometry.jsonl",
        "calibration": root / "manifests/p2-public/calibration-public.geometry.jsonl",
        "smartdoc_validation": root / "manifests/smartdoc.geometry.jsonl",
    }
    slices, hashes, cache = {}, {}, {}
    for name, path in manifests.items():
        _assert_public_training_manifest(path)
        records = read_development(path, "train" if name == "stage_b_train" else "validation")
        if name == "internal_validation" and len(records) > 1000:
            indices = sorted(
                np.random.default_rng(20260902).choice(len(records), 1000, replace=False)
            )
            records = [records[int(i)] for i in indices]
        slices[name] = identity_rows(records, root, cache)
        hashes[name] = sha256(path)
    if target is not None and target.is_file():
        slices["target_development"] = identity_rows(read_development(target), root, cache)
        hashes["target_development"] = sha256(target)
    else:
        slices["target_development"] = []
    test = load_test_identity_index(test_index)
    if test is not None:
        slices["smartdoc_test"] = test
        hashes["test_identity_index"] = sha256(test_index)
    pairs = {
        f"{a}__{b}": overlap(slices[a], slices[b]) for a, b in itertools.combinations(slices, 2)
    }
    return {
        "kind": "p4_dataset_isolation",
        "metric_version": 2,
        "nce_normalization": "target_quad_bbox_diagonal",
        "manifest_sha256": hashes,
        "slices": {k: summarize(v) for k, v in slices.items()},
        "overlap": pairs,
        "test_data_used": False,
        "test_isolation": "verified_index" if test is not None else "test_isolation_unverified",
        "target_status": "requires_target_validation"
        if slices["target_development"]
        else "insufficient_target_domain_groups",
    }


def markdown(report: dict) -> str:
    lines = [
        "# P4 数据隔离审计",
        "",
        "| slice | samples | sources | groups | sessions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in report["slices"].items():
        ids = s["independent"]
        lines.append(
            f"| {name} | {s['sample_count']} | {ids['digital_source_id']} | "
            f"{ids['group_id']} | {ids['capture_session']} |"
        )
    lines += [
        "",
        "0 个 source 可能代表身份缺失，不能解释为无泄漏。",
        "",
        "| pair | path | SHA256 | source | group | session |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, p in report["overlap"].items():
        c = p["intersection"]
        lines.append(
            f"| {name} | {c['normalized_path']} | {c['image_sha256']} | "
            f"{c['digital_source_id']} | {c['group_id']} | {c['capture_session']} |"
        )
    lines += [
        "",
        f"Test isolation: {report['test_isolation']}。",
        "既有开发切片只作历史诊断；source 身份不完整或 test 独立性未验证时，禁止进入正式 fit/selection/evaluation。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--test-identity-index", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    result = audit(args.data_root.resolve(), args.target_manifest, args.test_identity_index)
    (args.output_directory / "dataset-isolation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_directory / "dataset-isolation.md").write_text(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
