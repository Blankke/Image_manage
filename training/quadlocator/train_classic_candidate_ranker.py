#!/usr/bin/env python3
"""用公开 train/validation 分区训练轻量传统候选排序器，并冻结后评测 test。

使用范例：
    source .venv/bin/activate
    which python
    python training/quadlocator/train_classic_candidate_ranker.py \
      --data-root /Users/caozichen/screenrestore-data \
      --manifest /Users/caozichen/screenrestore-data/manifests/p9/syngallery-reviewed-both-20260917.geometry.jsonl \
      --output /Users/caozichen/screenrestore-runs/candidate-ranker/syngallery-linear.json

训练只使用 train 的候选对，validation 只选择 checkpoint，test 在权重冻结后读取并报告。
该排序器没有内容层语义，产物明确禁止单独用于自动接受；它只用于验证候选排序能否
缩小 oracle 召回与运行时 top-1 的差距。脚本显示缓存与训练进度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audit_classic_candidate_ranking import (  # noqa: E402
    FEATURE_NAMES,
    _positive_rows,
    _public_file,
    balanced_group_limit,
    candidate_feature_vector,
)

from screenrestore.geometry.classic import detect_classic_candidates  # noqa: E402
from screenrestore.io.image_loader import load_image  # noqa: E402
from screenrestore.validation.geometry_benchmark import corner_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--minimum-iou-gap", type=float, default=0.03)
    parser.add_argument(
        "--max-samples-per-manifest",
        type=int,
        default=600,
        help="每份清单每个 split 的 group 轮询上限；0 表示全部",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_candidates <= 32 or args.steps < 20 or args.max_samples_per_manifest < 0:
        raise ValueError("max-candidates 须位于 1..32，steps 至少为 20")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 < args.minimum_iou_gap < 1:
        raise ValueError("学习率、weight decay 或 minimum IoU gap 无效")
    root = args.data_root.expanduser().resolve()
    manifests = [_public_file(path, root) for path in args.manifest]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有候选排序器：{output}")
    _validate_group_isolation(manifests)

    cached = {
        split: _load_samples(
            manifests,
            root,
            split,
            args.max_candidates,
            args.max_samples_per_manifest,
        )
        for split in ("train", "validation")
    }
    mean, scale = _feature_normalization(cached["train"])
    pairwise = _pairwise_matrix(cached["train"], mean, scale, args.minimum_iou_gap)
    if pairwise.shape[0] < 20:
        raise ValueError("训练候选对不足，不能拟合排序器")
    weights, selection = fit_pairwise_ranker(
        pairwise,
        cached["validation"],
        mean,
        scale,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # test 直到权重冻结后才读取，避免其参与训练或 checkpoint 选择。
    test_samples = _load_samples(
        manifests,
        root,
        "test",
        args.max_candidates,
        args.max_samples_per_manifest,
    )
    report = {
        "kind": "screenrestore_classic_candidate_linear_ranker",
        "format_version": 1,
        "automatic_acceptance_eligible": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.round(10).tolist(),
        "feature_scale": scale.round(10).tolist(),
        "weights": weights.round(10).tolist(),
        "training": {
            "manifest_sha256": {str(path): _sha256(path) for path in manifests},
            "generator_sha256": _sha256(Path(__file__)),
            "max_candidates": args.max_candidates,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "minimum_iou_gap": args.minimum_iou_gap,
            "max_samples_per_manifest": args.max_samples_per_manifest,
            "pair_count": int(pairwise.shape[0]),
            "train_sample_count": len(cached["train"]),
            "validation_sample_count": len(cached["validation"]),
            "test_loaded_after_selection": True,
            "selection": selection,
        },
        "metrics": {
            "train": ranking_metrics(cached["train"], weights, mean, scale),
            "validation": ranking_metrics(cached["validation"], weights, mean, scale),
            "test": ranking_metrics(test_samples, weights, mean, scale),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def _load_samples(
    manifests: list[Path],
    root: Path,
    split: str,
    max_candidates: int,
    max_samples_per_manifest: int,
) -> list[dict]:
    rows: list[dict] = []
    for manifest in manifests:
        selected = _positive_rows(manifest, split)
        if max_samples_per_manifest:
            selected = balanced_group_limit(selected, max_samples_per_manifest)
        rows.extend(selected)
    samples: list[dict] = []
    for index, row in enumerate(rows, 1):
        image = load_image(_public_file(root / str(row["image"]), root)).original_rgb
        height, width = image.shape[:2]
        truth = np.asarray(row["content_quad"], np.float32) * np.asarray(
            [width - 1, height - 1], np.float32
        )
        items = []
        for candidate in detect_classic_candidates(image, max_candidates=max_candidates):
            nce, iou, _max_error = corner_metrics(candidate.corners, truth)
            items.append(
                {
                    "features": candidate_feature_vector(candidate),
                    "iou": float(iou),
                    "nce": float(nce),
                }
            )
        samples.append({"group_id": str(row["group_id"]), "items": items})
        _progress(index, len(rows), f"缓存 {split} 候选")
    if not samples:
        raise ValueError(f"manifest 缺少 split={split} 的公开正样本")
    return samples


def _feature_normalization(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.vstack([item["features"] for sample in samples for item in sample["items"]])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _pairwise_matrix(
    samples: list[dict], mean: np.ndarray, scale: np.ndarray, minimum_gap: float
) -> np.ndarray:
    pairs: list[np.ndarray] = []
    for sample in samples:
        items = sample["items"]
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                gap = float(items[left]["iou"] - items[right]["iou"])
                if abs(gap) < minimum_gap:
                    continue
                better, worse = (items[left], items[right]) if gap > 0 else (items[right], items[left])
                pairs.append((better["features"] - worse["features"]) / scale)
    return np.vstack(pairs) if pairs else np.empty((0, len(FEATURE_NAMES)), np.float64)


def fit_pairwise_ranker(
    pairs: np.ndarray,
    validation: list[dict],
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, dict]:
    """用确定性 Adam 最小化 pairwise logistic loss，以 validation 冻结 checkpoint。"""

    weights = np.zeros(pairs.shape[1], np.float64)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    best_weights = weights.copy()
    best_key: tuple[float, ...] | None = None
    best_step = 0
    checkpoints: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        margins = np.clip(pairs @ weights, -40, 40)
        probabilities = 1.0 / (1.0 + np.exp(margins))
        gradient = -(pairs.T @ probabilities) / pairs.shape[0] + weight_decay * weights
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * np.square(gradient)
        first_hat = first / (1 - 0.9**step)
        second_hat = second / (1 - 0.999**step)
        weights -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        if step == 1 or step % 20 == 0 or step == steps:
            metrics = ranking_metrics(validation, weights, mean, scale)
            loss = float(np.mean(np.logaddexp(0.0, -pairs @ weights)))
            key = (
                float(metrics["reranked_strict_count"]),
                float(metrics["reranked_iou_0_9_count"]),
                float(metrics["reranked_iou_median"]),
                -loss,
            )
            if best_key is None or key > best_key:
                best_key, best_weights, best_step = key, weights.copy(), step
            checkpoints.append({"step": step, "pairwise_loss": round(loss, 8), **metrics})
            _progress(step, steps, "训练候选排序器")
    return best_weights, {
        "best_step": best_step,
        "criterion": ["validation_strict", "validation_iou_0_9", "validation_median", "pairwise_loss"],
        "checkpoints": checkpoints,
    }


def ranking_metrics(
    samples: list[dict], weights: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> dict[str, object]:
    baseline, reranked, oracle = [], [], []
    for sample in samples:
        items = sample["items"]
        if not items:
            continue
        baseline.append(items[0])
        reranked.append(
            max(items, key=lambda item: float(((item["features"] - mean) / scale) @ weights))
        )
        oracle.append(max(items, key=lambda item: float(item["iou"])))

    def count_iou(items: list[dict], threshold: float) -> int:
        return sum(float(item["iou"]) >= threshold for item in items)

    def strict(items: list[dict]) -> int:
        return sum(float(item["iou"]) >= 0.93 and float(item["nce"]) <= 0.01 for item in items)

    def median(items: list[dict]) -> float | None:
        return round(float(np.median([item["iou"] for item in items])), 8) if items else None

    return {
        "sample_count": len(samples),
        "candidate_coverage": round(len(baseline) / max(1, len(samples)), 8),
        "baseline_iou_0_9_count": count_iou(baseline, 0.9),
        "baseline_strict_count": strict(baseline),
        "baseline_iou_median": median(baseline),
        "reranked_iou_0_9_count": count_iou(reranked, 0.9),
        "reranked_strict_count": strict(reranked),
        "reranked_iou_median": median(reranked),
        "oracle_iou_0_9_count": count_iou(oracle, 0.9),
        "oracle_strict_count": strict(oracle),
        "oracle_iou_median": median(oracle),
    }


def _validate_group_isolation(manifests: list[Path]) -> None:
    assignments: dict[str, str] = {}
    images: set[str] = set()
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            group, split = str(row.get("group_id", "")), str(row.get("split", ""))
            image = str(row.get("image", ""))
            if image in images:
                raise ValueError(f"多份清单重复图片：{image}")
            images.add(image)
            if group in assignments and assignments[group] != split:
                raise ValueError(f"group 跨 split：{group}")
            assignments[group] = split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(done: int, total: int, message: str) -> None:
    if done < total and done % max(1, total // 40):
        return
    filled = round(24 * done / max(1, total))
    print(
        f"\r{message} [{'#' * filled}{'-' * (24 - filled)}] {done}/{total}",
        end="\n" if done == total else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
