"""自动几何严格正确率的可审计 logistic 校准器。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIBRATOR_VERSION = "strict-correct-logistic-v1"


@dataclass(frozen=True, slots=True)
class CorrectnessCalibrator:
    """纯 JSON 参数的 logistic 模型，不依赖训练框架或 pickle。"""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float
    manifest_sha256: str
    version: str = CALIBRATOR_VERSION

    def __post_init__(self) -> None:
        lengths = {len(self.feature_names), len(self.means), len(self.scales), len(self.coefficients)}
        if len(lengths) != 1 or not self.feature_names:
            raise ValueError("校准器特征、均值、尺度与系数长度必须一致且非空")
        if any(float(value) <= 0.0 for value in self.scales):
            raise ValueError("校准器尺度必须为正数")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("校准阈值必须位于 (0,1)")

    def predict_probability(self, features: Mapping[str, float]) -> float:
        """对一条推理时可得特征输出 ``P(strict_correct)``。"""

        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"校准特征缺失：{missing}")
        values = np.asarray([features[name] for name in self.feature_names], np.float64)
        standardized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        logit = float(np.dot(standardized, np.asarray(self.coefficients)) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CorrectnessCalibrator:
        version = str(data.get("version", ""))
        if version != CALIBRATOR_VERSION:
            raise ValueError(f"不支持的校准器版本：{version}")
        return cls(
            feature_names=tuple(str(value) for value in data["feature_names"]),  # type: ignore[arg-type]
            means=tuple(float(value) for value in data["means"]),  # type: ignore[arg-type]
            scales=tuple(float(value) for value in data["scales"]),  # type: ignore[arg-type]
            coefficients=tuple(float(value) for value in data["coefficients"]),  # type: ignore[arg-type]
            intercept=float(data["intercept"]),
            threshold=float(data["threshold"]),
            manifest_sha256=str(data["manifest_sha256"]),
            version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> CorrectnessCalibrator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("校准器 JSON 顶层必须是对象")
        return cls.from_dict(data)


__all__ = ["CALIBRATOR_VERSION", "CorrectnessCalibrator"]
