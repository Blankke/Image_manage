"""拟合严格正确率 logistic 校准器。

使用示例：
    python -m training.quadlocator.correctness_calibrator \
      --input validation-features.jsonl --output correctness-calibrator.json \
      --manifest-sha256 <sha256> --minimum-precision 0.99

输入每行必须包含 ``strict_correct`` 布尔值以及 ``features`` 数值对象。该工具只能读取
validation/calibration 冻结预测，禁止把 test 清单传入拟合过程。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from screenrestore.geometry.calibration import CorrectnessCalibrator


def fit_calibrator(
    feature_rows: list[dict[str, float]],
    labels: list[bool],
    *,
    manifest_sha256: str,
    minimum_precision: float = 0.99,
    iterations: int = 1200,
) -> CorrectnessCalibrator:
    """用确定性全批量梯度下降拟合小型 L2 logistic 模型。"""

    if not feature_rows or len(feature_rows) != len(labels):
        raise ValueError("校准特征与标签必须非空且长度一致")
    names = tuple(sorted(feature_rows[0]))
    if any(tuple(sorted(row)) != names for row in feature_rows):
        raise ValueError("所有校准记录必须使用同一特征集合")
    matrix = np.asarray([[row[name] for name in names] for row in feature_rows], np.float64)
    target = np.asarray(labels, np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    values = (matrix - means) / scales
    coefficients = np.zeros(values.shape[1], np.float64)
    intercept = 0.0
    for step in range(iterations):
        logits = values @ coefficients + intercept
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = probability - target
        learning_rate = 0.08 / (1.0 + step / 400.0)
        coefficients -= learning_rate * ((values.T @ error) / len(target) + 1e-3 * coefficients)
        intercept -= learning_rate * float(error.mean())
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(values @ coefficients + intercept, -30.0, 30.0))
    )
    threshold = _select_threshold(probabilities, target >= 0.5, minimum_precision)
    return CorrectnessCalibrator(
        feature_names=names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        threshold=threshold,
        manifest_sha256=manifest_sha256,
    )


def _select_threshold(probabilities: np.ndarray, labels: np.ndarray, minimum_precision: float) -> float:
    candidates = np.unique(np.clip(probabilities, 1e-6, 1.0 - 1e-6))[::-1]
    selected = 1.0 - 1e-6
    best_accepted = -1
    for threshold in candidates:
        accepted = probabilities >= threshold
        count = int(accepted.sum())
        precision = float(labels[accepted].mean()) if count else 1.0
        if precision >= minimum_precision and count > best_accepted:
            selected = float(threshold)
            best_accepted = count
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拟合 P3 几何严格正确率校准器")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.99)
    args = parser.parse_args(argv)
    rows: list[dict[str, float]] = []
    labels: list[bool] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") == "test":
            raise ValueError("test 预测禁止参与校准器拟合")
        features = record.get("features")
        if not isinstance(features, dict):
            raise ValueError("每条记录必须包含 features 对象")
        rows.append({str(key): float(value) for key, value in features.items()})
        labels.append(bool(record.get("strict_correct", False)))
    calibrator = fit_calibrator(
        rows,
        labels,
        manifest_sha256=args.manifest_sha256,
        minimum_precision=args.minimum_precision,
    )
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖校准器：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibrator.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[########################] 校准器完成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
