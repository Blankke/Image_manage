#!/usr/bin/env python3
"""生成目标域标注 overlay、校验人工复核与确定性 group 分区。

用法：source .venv/bin/activate && which python
python scripts/prepare_p4_target_domain.py --manifest /data/manifests/p4-public-target/annotations.geometry.jsonl \
  --data-root /data --output-directory /runs/target-overlays
输出 review.jsonl 的 status 初始为 pending；人工看图后填写 approved、reviewer、reviewed_at。
随后将 image_sha256、annotation_sha256 和审核字段写入原标注的 overlay_review 对象，再运行验证。
不会创建真实标注或自动批准人工审核。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_p4_acceptance import _group_partition  # noqa: E402
from scripts.audit_p4_dataset_isolation import (  # noqa: E402
    IDENTIFIERS,
    read_development,
    sha256,
    token,
)

from screenrestore.io.image_loader import load_image  # noqa: E402

DOMAINS = ("postcard", "artwork", "screen", "poster")
REJECT_SCENES = {
    "partial",
    "multi-target",
    "gallery_multi_target",
    "curved",
    "severe_occlusion",
    "extreme_view",
}
CONDITIONS = {
    "frontal",
    "mild_perspective",
    "moderate_perspective",
    "near_border",
    "nested_layer",
    "weak_edge",
    "light_glare",
}


def annotation_sha256(record: dict) -> str:
    """审核绑定整条标注；修改任何标注后必须重新看 overlay。"""
    return token(
        json.dumps(
            {k: v for k, v in record.items() if k != "overlay_review"},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def group_partition(group_id: str) -> str:
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("missing group_id")
    return _group_partition(group_id)


def leakage_errors(records: list[dict]) -> list[str]:
    """source/subject/session 不得跨 group、split 或 development partition。"""
    errors, assignments = set(), {}
    for row in records:
        group = row.get("group_id")
        if not group:
            errors.add("missing_group_id")
            continue
        destination = (row.get("split"), group_partition(group))
        for key in IDENTIFIERS:
            if not row.get(key):
                errors.add(f"missing_{key}")
                continue
            value = row[key]
            identity = (key, value)
            previous = assignments.setdefault(identity, (group, destination))
            if previous != (group, destination):
                errors.add(f"{key}_leakage")
    # 同一 group 必须只对应一个 source family。
    families = defaultdict(set)
    for row in records:
        families[row.get("group_id")].add(row.get("digital_source_id"))
    if any(len(values) != 1 for values in families.values()):
        errors.add("group_source_family_inconsistent")
    return sorted(errors)


def validate_target(records: list[dict], root: Path, isolation: dict | None = None) -> dict:
    errors = set(leakage_errors(records))
    groups = {domain: set() for domain in DOMAINS}
    sessions, conditions = defaultdict(Counter), defaultdict(set)
    positives, negatives = Counter(), Counter()
    partitions = Counter()
    group_domains = defaultdict(set)
    images = set()
    image_families = {}
    for row in records:
        required = (
            "image",
            "split",
            "group_id",
            "capture_session",
            "device",
            "present",
            "target_class",
            "content_quad",
            "outer_quad",
            "visible",
            "occlusion",
            "glare_level",
            "domain",
            "subject_id",
            "digital_source_id",
            "source",
            "source_license",
            "scene_type",
        )
        if any(key not in row for key in required):
            errors.add("incomplete_geometry_manifest")
        if not row.get("source") or not row.get("source_license"):
            errors.add("missing_source_license")
        domain = row.get("domain")
        if domain not in DOMAINS:
            errors.add("invalid_domain")
            continue
        group = row.get("group_id")
        if group:
            groups[domain].add(group)
            group_domains[group].add(domain)
            partitions[group_partition(group)] += 1
        if row.get("split") != "validation":
            errors.add("split_leakage")
        expected_class = "artwork" if domain == "poster" else domain
        positive = row.get("present") is True
        if positive:
            positives[domain] += 1
            if row.get("target_class") != expected_class or row.get("in_scope") is not True:
                errors.add("positive_semantics_invalid")
            if (
                row.get("visible") is not True
                or row.get("ambiguous", False)
                or float(row.get("occlusion", 1)) > 0.25
                or row.get("glare_level") == "strong"
            ):
                errors.add("positive_visibility_invalid")
            if row.get("scene_type") in REJECT_SCENES:
                errors.add("hard_negative_disguised_as_positive")
            quad = np.asarray(row.get("content_quad"), dtype=float)
            if (
                quad.shape != (4, 2)
                or not np.isfinite(quad).all()
                or np.any((quad < 0) | (quad > 1))
            ):
                errors.add("invalid_normalized_quad")
            elif (
                not cv2.isContourConvex(quad.astype(np.float32))
                or cv2.contourArea(quad.astype(np.float32), oriented=True) <= 0
            ):
                errors.add("invalid_quad_order")
            sessions[group][row.get("capture_session")] += 1
            conditions[group].update(row.get("capture_conditions", []))
        else:
            negatives[domain] += 1
            if (
                row.get("target_class") != "none"
                or row.get("in_scope") is not False
                or row.get("content_quad") is not None
            ):
                errors.add("negative_semantics_invalid")
        if domain == "poster" and positive and row.get("scene_type") != "poster":
            errors.add("poster_scene_missing")
        path = (root / str(row.get("image", ""))).resolve()
        if path in images:
            errors.add("duplicate_image")
        images.add(path)
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.add("missing_or_unsafe_image")
            continue
        review = row.get("overlay_review", {})
        image_digest = sha256(path)
        previous_family = image_families.setdefault(image_digest, group)
        if previous_family != group:
            errors.add("image_leakage")
        if not (
            review.get("status") == "approved"
            and review.get("reviewer")
            and review.get("reviewed_at")
            and review.get("image_sha256") == image_digest
            and review.get("annotation_sha256") == annotation_sha256(row)
        ):
            errors.add("overlay_not_reviewed_or_stale")
    if any(len(domains) != 1 for domains in group_domains.values()):
        errors.add("group_domain_inconsistent")
    if any(len(value) < 8 for value in groups.values()):
        errors.add("insufficient_target_domain_groups")
    for domain, ids in groups.items():
        for group in ids:
            if len(sessions[group]) < 2 or any(
                not 6 <= count <= 10 for count in sessions[group].values()
            ):
                errors.add("insufficient_capture_sessions")
            if not conditions[group] >= CONDITIONS:
                errors.add("missing_capture_conditions")
        if positives[domain] == 0 or not 0.20 <= negatives[domain] / positives[domain] <= 0.30:
            errors.add("hard_negative_ratio_outside_protocol")
    if any(partitions[p] == 0 for p in ("fit", "selection", "evaluation")):
        errors.add("empty_development_partition")
    if isolation is None or isolation.get("test_isolation") != "verified_index":
        errors.add("test_isolation_unverified")
    if isolation is None or "target_development" not in isolation.get("slices", {}):
        errors.add("target_isolation_missing")
    else:
        comparisons = {
            part
            for pair in isolation.get("overlap", {})
            if "target_development" in pair
            for part in pair.split("__")
            if part != "target_development"
        }
        if not comparisons >= {
            "stage_b_train",
            "internal_validation",
            "calibration",
            "smartdoc_validation",
            "smartdoc_test",
        }:
            errors.add("incomplete_external_isolation_audit")
        for pair, audit in isolation.get("overlap", {}).items():
            if "target_development" not in pair:
                continue
            if any(
                audit[k]
                for k in (
                    "image_leakage",
                    "source_family_leakage",
                    "group_leakage",
                    "capture_session_leakage",
                )
            ):
                errors.add("external_dataset_overlap")
            if not audit["source_family_verified"]:
                errors.add("external_source_family_unverified")
    return {
        "status": "READY" if not errors else "BLOCKED: insufficient_target_domain_groups",
        "reasons": sorted(errors),
        "sample_count": len(records),
        "independent_groups_by_domain": {k: len(v) for k, v in groups.items()},
        "partition_samples": dict(partitions),
    }


def build_reviewed_manifest(records: list[dict], reviews: list[dict], destination: Path) -> None:
    """把人工审核结果绑定到标注快照；不自动把 pending 改为 approved。"""
    by_annotation = {}
    for review in reviews:
        key = review["annotation_sha256"]
        if key in by_annotation:
            raise ValueError("duplicate annotation review")
        by_annotation[key] = review
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        for record in records:
            key = annotation_sha256(record)
            if key not in by_annotation:
                raise ValueError("missing or stale annotation review")
            review = by_annotation[key]
            bound = {
                k: review[k]
                for k in ("status", "reviewer", "reviewed_at", "image_sha256", "annotation_sha256")
            }
            handle.write(json.dumps({**record, "overlay_review": bound}, ensure_ascii=False) + "\n")


def render_overlays(records: list[dict], root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "review.jsonl").open("x") as reviews:
        for index, record in enumerate(records):
            path = (root / record["image"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("image outside dataset root")
            rgb = load_image(path).original_rgb.copy()
            scale = np.array([rgb.shape[1] - 1, rgb.shape[0] - 1])
            for key, color in [("content_quad", (0, 255, 255)), ("outer_quad", (255, 0, 255))]:
                if record.get(key) is not None:
                    quad = np.asarray(record[key], dtype=float)
                    if (
                        quad.shape != (4, 2)
                        or not np.isfinite(quad).all()
                        or np.any((quad < 0) | (quad > 1))
                    ):
                        raise ValueError("invalid normalized annotation quad")
                    points = np.rint(quad * scale).astype(np.int32)
                    cv2.polylines(rgb, [points], True, color, max(2, rgb.shape[1] // 500))
                    for corner, point in enumerate(points):
                        cv2.putText(
                            rgb, str(corner), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                        )
            name = f"sample-{index:05d}.png"
            Image.fromarray(rgb).save(output / name)
            reviews.write(
                json.dumps(
                    {
                        "overlay": name,
                        "image_sha256": sha256(path),
                        "annotation_sha256": annotation_sha256(record),
                        "status": "pending",
                        "reviewer": "",
                        "reviewed_at": "",
                    }
                )
                + "\n"
            )
            print(f"[overlay {index + 1}/{len(records)}]", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args(argv)
    records = read_development(args.manifest)
    if args.reviews is not None:
        if args.output_manifest is None or args.output_directory is not None:
            parser.error("--reviews 必须搭配 --output-manifest")
        repo = Path(__file__).resolve().parents[1]
        if args.output_manifest.resolve().is_relative_to(repo):
            raise ValueError("目标域标注清单必须写在仓库外")
        reviews = [
            json.loads(line) for line in args.reviews.read_text().splitlines() if line.strip()
        ]
        build_reviewed_manifest(records, reviews, args.output_manifest)
    else:
        if args.output_directory is None or args.output_manifest is not None:
            parser.error("overlay 模式必须提供 --output-directory")
        render_overlays(records, args.data_root, args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
