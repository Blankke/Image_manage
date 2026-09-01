"""QuadLocator 角点热图的唯一解码规范。

训练验证、ONNX 运行时、overlay 和 benchmark 都应调用本模块，避免同一热图在不同
入口得到不同四角。热图输入是 logits；坐标输出位于热图像素空间。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DECODER_VERSION = "quad-peak-local-softargmax-v1"


@dataclass(frozen=True, slots=True)
class CornerDecoderSpec:
    """可写入 checkpoint 与报告的稳定解码契约。"""

    version: str = DECODER_VERSION
    local_window: int = 5
    nms_radius: int = 3
    minimum_peak: float = 0.05

    def __post_init__(self) -> None:
        if self.local_window not in (5, 7) or self.local_window % 2 == 0:
            raise ValueError("局部 soft-argmax 窗口必须为 5 或 7")
        if self.nms_radius < 1:
            raise ValueError("NMS 半径必须至少为 1")
        if not 0.0 <= self.minimum_peak < 1.0:
            raise ValueError("minimum_peak 必须位于 [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CornerPeakDiagnostics:
    """单角原始多峰证据，数值均可直接写入 JSON。"""

    peak1: float
    peak2: float
    peak_difference: float
    peak_ratio: float
    peak_distance: float
    normalized_entropy: float
    local_sharpness: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class DecodedCorners:
    """四通道角点热图的结构化解码结果。"""

    coordinates: np.ndarray | None
    confidences: tuple[float, float, float, float]
    diagnostics: tuple[
        CornerPeakDiagnostics,
        CornerPeakDiagnostics,
        CornerPeakDiagnostics,
        CornerPeakDiagnostics,
    ]
    spec: CornerDecoderSpec


def decode_corner_logits(
    logits: np.ndarray,
    spec: CornerDecoderSpec | None = None,
) -> DecodedCorners:
    """按全局峰、NMS 第二峰和局部 sigmoid soft-argmax 解码四角。

    接受 ``4×H×W`` 或 ``1×4×H×W`` logits。第二峰只用于歧义诊断，不会把坐标
    拉向远处候选；这正是相对旧版全图质心/softmax 的关键区别。
    """

    active_spec = spec or CornerDecoderSpec()
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("角点 logits 必须为 4×H×W 或 1×4×H×W")
    if not np.all(np.isfinite(values)):
        raise ValueError("角点 logits 包含非有限值")
    probabilities = _sigmoid(values)
    coordinates: list[list[float]] = []
    confidences: list[float] = []
    diagnostics: list[CornerPeakDiagnostics] = []
    for heatmap in probabilities:
        decoded = _decode_one(heatmap, active_spec)
        diagnostics.append(decoded)
        confidences.append(decoded.peak1)
        if decoded.peak1 >= active_spec.minimum_peak:
            coordinates.append([decoded.x, decoded.y])
    coordinate_array = (
        np.asarray(coordinates, dtype=np.float32) if len(coordinates) == 4 else None
    )
    return DecodedCorners(
        coordinate_array,
        tuple(confidences),  # type: ignore[arg-type]
        tuple(diagnostics),  # type: ignore[arg-type]
        active_spec,
    )


def _decode_one(heatmap: np.ndarray, spec: CornerDecoderSpec) -> CornerPeakDiagnostics:
    height, width = heatmap.shape
    flat_index = int(np.argmax(heatmap))
    peak_y, peak_x = divmod(flat_index, width)
    peak1 = float(heatmap[peak_y, peak_x])

    suppressed = heatmap.copy()
    yy, xx = np.ogrid[:height, :width]
    suppressed[(xx - peak_x) ** 2 + (yy - peak_y) ** 2 <= spec.nms_radius**2] = -1.0
    second_index = int(np.argmax(suppressed))
    second_y, second_x = divmod(second_index, width)
    peak2 = max(0.0, float(suppressed[second_y, second_x]))

    radius = spec.local_window // 2
    y0, y1 = max(0, peak_y - radius), min(height, peak_y + radius + 1)
    x0, x1 = max(0, peak_x - radius), min(width, peak_x + radius + 1)
    local = heatmap[y0:y1, x0:x1]
    local_total = max(float(local.sum()), 1e-8)
    local_y, local_x = np.indices(local.shape, dtype=np.float32)
    decoded_x = float(((local_x + x0) * local).sum() / local_total)
    decoded_y = float(((local_y + y0) * local).sum() / local_total)

    global_weights = heatmap / max(float(heatmap.sum()), 1e-8)
    entropy = -float(np.sum(global_weights * np.log(np.clip(global_weights, 1e-12, 1.0))))
    normalized_entropy = entropy / max(float(np.log(max(2, heatmap.size))), 1e-8)
    local_mean_without_peak = (local_total - peak1) / max(1, local.size - 1)
    local_sharpness = float(np.clip(peak1 - local_mean_without_peak, 0.0, 1.0))
    peak_distance = float(np.hypot(second_x - peak_x, second_y - peak_y))
    return CornerPeakDiagnostics(
        peak1=peak1,
        peak2=peak2,
        peak_difference=max(0.0, peak1 - peak2),
        peak_ratio=peak1 / max(peak2, 1e-8),
        peak_distance=peak_distance,
        normalized_entropy=float(np.clip(normalized_entropy, 0.0, 1.0)),
        local_sharpness=local_sharpness,
        x=decoded_x,
        y=decoded_y,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


__all__ = [
    "DECODER_VERSION",
    "CornerDecoderSpec",
    "CornerPeakDiagnostics",
    "DecodedCorners",
    "decode_corner_logits",
]
