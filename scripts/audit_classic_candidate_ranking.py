#!/usr/bin/env python3
"""审核传统四边形候选的召回与运行时排序，不把 oracle 排名用于自动接受。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_classic_candidate_ranking.py \
      --data-root /Users/caozichen/screenrestore-data \
      --manifest /Users/caozichen/screenrestore-data/manifests/p9/syngallery-reviewed-both-20260917.geometry.jsonl \
      --split test --max-candidates 12 \
      --output /Users/caozichen/screenrestore-runs/candidate-audit/syngallery-test.json

报告分别记录运行时 top-1 与 GT oracle best。oracle 结果只回答候选池是否包含正确四角，
不能作为无人值守接受策略，也不能与 e2e_auto 指标混用。脚本只读取显式公开 data-root，
拒绝路径中含 private 的数据，并显示逐样本进度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from screenrestore.geometry import OnnxQuadDetector
from screenrestore.geometry.classic import detect_classic_candidates
from screenrestore.io.image_loader import load_image
from screenrestore.validation.geometry_benchmark import corner_metrics

FEATURE_NAMES = (
    "area",
    "rectangularity",
    "edge_strength",
    "center",
    "side_balance",
    "is_classic_contour",
    "is_classic_hough",
    "is_classic_profile",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示全部样本")
    parser.add_argument("--ranker", type=Path, help="可选的公开数据线性候选排序器 JSON")
    parser.add_argument("--quad-model", type=Path, help="可选 QuadLocator；按粗 content quad 重合度重排")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_candidates <= 32 or args.max_samples < 0:
        raise ValueError("max-candidates 须位于 1..32，max-samples 不能为负数")
    root = args.data_root.expanduser().resolve()
    manifest = _public_file(args.manifest, root)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有候选审核：{output}")
    ranker = load_ranker(args.ranker) if args.ranker is not None else None
    detector = OnnxQuadDetector(args.quad_model) if args.quad_model is not None else None

    rows = _positive_rows(manifest, args.split)
    if args.max_samples:
        rows = balanced_group_limit(rows, args.max_samples)
    if not rows:
        raise ValueError(f"manifest 没有 split={args.split} 的正样本")

    cases: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        cases.append(_audit_record(row, root, args.max_candidates, index, ranker, detector))
        _progress(index, len(rows))
    report = {
        "protocol": "oracle_classic_candidate_ranking_diagnostic",
        "protocol_version": 1,
        "automatic_acceptance_eligible": False,
        "split": args.split,
        "max_candidates": args.max_candidates,
        "manifest_sha256": _sha256(manifest),
        "generator_sha256": _sha256(Path(__file__)),
        "ranker_sha256": _sha256(args.ranker.expanduser().resolve()) if args.ranker else None,
        "quad_model_sha256": _sha256(args.quad_model.expanduser().resolve()) if args.quad_model else None,
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def _positive_rows(manifest: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"manifest 第 {line_number} 行不是对象")
        if row.get("split") != split or not row.get("present") or not row.get("in_scope", True):
            continue
        if not row.get("group_id") or row.get("content_quad") is None:
            raise ValueError(f"manifest 第 {line_number} 行缺少 group_id/content_quad")
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["image"]))


def balanced_group_limit(rows: list[dict], limit: int) -> list[dict]:
    """按 group 轮询截取，避免 max-samples 被一个长连拍占满。"""

    if limit <= 0 or len(rows) <= limit:
        return rows
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    selected: list[dict] = []
    depth = 0
    ordered_groups = sorted(groups)
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if depth < len(groups[group]):
                selected.append(groups[group][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _audit_record(
    row: dict,
    root: Path,
    max_candidates: int,
    sample_number: int,
    ranker: dict[str, np.ndarray] | None,
    detector: OnnxQuadDetector | None,
) -> dict:
    image_path = _public_file(root / str(row["image"]), root)
    image = load_image(image_path).original_rgb
    height, width = image.shape[:2]
    normalized = np.asarray(row["content_quad"], np.float32)
    if normalized.shape != (4, 2) or np.any(~np.isfinite(normalized)) or np.any((normalized < 0) | (normalized > 1)):
        raise ValueError(f"content_quad 无效：{row['image']}")
    truth = normalized * np.asarray([width - 1, height - 1], np.float32)
    candidates = detect_classic_candidates(image, max_candidates=max_candidates)
    prediction = detector.predict(image) if detector is not None else None
    scored: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, 1):
        nce, iou, max_corner_error = corner_metrics(candidate.corners, truth)
        ranker_score = linear_ranker_score(candidate_feature_vector(candidate), ranker)
        scored.append(
            {
                "runtime_rank": rank,
                "source": candidate.source,
                "runtime_confidence": round(float(candidate.confidence), 8),
                "corner_nce": round(float(nce), 8),
                "quad_iou": round(float(iou), 8),
                "max_corner_error_px": round(float(max_corner_error), 4),
                "ranker_score": round(ranker_score, 8) if ranker_score is not None else None,
            }
        )
    oracle = max(scored, key=lambda item: float(item["quad_iou"])) if scored else None
    reranked = (
        max(scored, key=lambda item: float(item["ranker_score"]))
        if scored and ranker is not None
        else None
    )
    agreement = None
    if scored and prediction is not None and prediction.content_quad is not None:
        agreement_index = max(
            range(len(candidates)),
            key=lambda index: corner_metrics(candidates[index].corners, prediction.content_quad)[1],
        )
        agreement = dict(scored[agreement_index])
        agreement["model_agreement_iou"] = round(
            float(corner_metrics(candidates[agreement_index].corners, prediction.content_quad)[1]),
            8,
        )
    return {
        "sample_id": f"sample-{sample_number:04d}",
        "image": str(row["image"]),
        "group_id": str(row["group_id"]),
        "domain": str(row.get("domain", row.get("target_class", "unknown"))),
        "candidate_count": len(scored),
        "runtime_top1": scored[0] if scored else None,
        "reranked_top1": reranked,
        "model_agreement_top1": agreement,
        "model_target_class": prediction.target_class.value if prediction is not None else None,
        "oracle_best": oracle,
    }


def candidate_feature_vector(candidate) -> np.ndarray:
    """把传统候选转换为训练和审核共用的稳定特征顺序。"""

    return np.asarray(
        [
            float(candidate.scores.get("area", 0.0)),
            float(candidate.scores.get("rectangularity", 0.0)),
            float(candidate.scores.get("edge_strength", 0.0)),
            float(candidate.scores.get("center", 0.0)),
            float(candidate.scores.get("side_balance", 0.0)),
            float(candidate.source == "classic_contour"),
            float(candidate.source == "classic_hough"),
            float(candidate.source == "classic_profile"),
        ],
        np.float64,
    )


def load_ranker(path: Path) -> dict[str, np.ndarray]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if value.get("kind") != "screenrestore_classic_candidate_linear_ranker":
        raise ValueError("ranker kind 不符合候选排序器契约")
    if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("ranker feature_names 与当前候选特征不一致")
    ranker = {
        name: np.asarray(value[name], np.float64)
        for name in ("feature_mean", "feature_scale", "weights")
    }
    if any(array.shape != (len(FEATURE_NAMES),) for array in ranker.values()) or np.any(
        ranker["feature_scale"] <= 0
    ):
        raise ValueError("ranker 参数形状或 scale 无效")
    return ranker


def linear_ranker_score(
    features: np.ndarray, ranker: dict[str, np.ndarray] | None
) -> float | None:
    if ranker is None:
        return None
    standardized = (features - ranker["feature_mean"]) / ranker["feature_scale"]
    return float(standardized @ ranker["weights"])


def summarize_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    """汇总运行时排序与 oracle 召回；测试可直接验证边界条件。"""

    top = [case["runtime_top1"] for case in cases if case.get("runtime_top1")]
    reranked = [case["reranked_top1"] for case in cases if case.get("reranked_top1")]
    agreement = [
        case["model_agreement_top1"] for case in cases if case.get("model_agreement_top1")
    ]
    oracle = [case["oracle_best"] for case in cases if case.get("oracle_best")]
    top_ious = [float(item["quad_iou"]) for item in top]
    oracle_ious = [float(item["quad_iou"]) for item in oracle]
    reranked_ious = [float(item["quad_iou"]) for item in reranked]
    agreement_ious = [float(item["quad_iou"]) for item in agreement]
    ranks = Counter(int(item["runtime_rank"]) for item in oracle)

    def strict(item: dict[str, object]) -> bool:
        return float(item["quad_iou"]) >= 0.93 and float(item["corner_nce"]) <= 0.01

    return {
        "sample_count": len(cases),
        "independent_group_count": len({str(case["group_id"]) for case in cases}),
        "candidate_coverage": round(len(top) / max(1, len(cases)), 8),
        "top1_iou_0_9_count": sum(value >= 0.9 for value in top_ious),
        "top1_iou_0_9_rate": round(sum(value >= 0.9 for value in top_ious) / max(1, len(cases)), 8),
        "oracle_iou_0_9_count": sum(value >= 0.9 for value in oracle_ious),
        "oracle_iou_0_9_recall": round(sum(value >= 0.9 for value in oracle_ious) / max(1, len(cases)), 8),
        "top1_strict_count": sum(strict(item) for item in top),
        "reranked_iou_0_9_count": sum(value >= 0.9 for value in reranked_ious),
        "reranked_iou_0_9_rate": (
            round(sum(value >= 0.9 for value in reranked_ious) / max(1, len(cases)), 8)
            if reranked
            else None
        ),
        "reranked_strict_count": sum(strict(item) for item in reranked) if reranked else None,
        "reranked_iou_median": (
            round(statistics.median(reranked_ious), 8) if reranked_ious else None
        ),
        "model_agreement_coverage": (
            round(len(agreement) / max(1, len(cases)), 8) if agreement else None
        ),
        "model_agreement_iou_0_9_count": sum(value >= 0.9 for value in agreement_ious),
        "model_agreement_iou_0_9_rate": (
            round(sum(value >= 0.9 for value in agreement_ious) / max(1, len(cases)), 8)
            if agreement
            else None
        ),
        "model_agreement_strict_count": sum(strict(item) for item in agreement) if agreement else None,
        "model_agreement_iou_median": (
            round(statistics.median(agreement_ious), 8) if agreement_ious else None
        ),
        "oracle_strict_count": sum(strict(item) for item in oracle),
        "top1_iou_median": round(statistics.median(top_ious), 8) if top_ious else None,
        "oracle_iou_median": round(statistics.median(oracle_ious), 8) if oracle_ious else None,
        "oracle_best_runtime_rank_counts": {str(rank): count for rank, count in sorted(ranks.items())},
    }


def _public_file(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root) or any(
        "private" in part.casefold() for part in resolved.relative_to(root).parts
    ) or not resolved.is_file():
        raise ValueError(f"候选审核只能读取 data-root 下的公开文件：{path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(done: int, total: int) -> None:
    if done < total and done % max(1, total // 40):
        return
    filled = round(24 * done / max(1, total))
    print(
        f"\r候选排序审核 [{'#' * filled}{'-' * (24 - filled)}] {done}/{total}",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
