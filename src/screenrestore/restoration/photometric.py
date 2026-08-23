"""受约束的摄影光度校正参数与纯函数。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MonotonicToneCurve:
    """以单调控制点表达的全局色调响应。"""

    input_knots: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    output_knots: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        inputs = np.asarray(self.input_knots, dtype=np.float32)
        outputs = np.asarray(self.output_knots, dtype=np.float32)
        if len(inputs) < 2 or inputs.shape != outputs.shape:
            raise ValueError("tone curve 输入输出控制点数量必须相同且不少于 2")
        if np.any(np.diff(inputs) <= 0) or np.any(np.diff(outputs) < 0):
            raise ValueError("tone curve 控制点必须单调")
        if inputs[0] != 0.0 or inputs[-1] != 1.0:
            raise ValueError("tone curve 输入范围必须覆盖 0..1")
        if np.any((outputs < 0) | (outputs > 1)):
            raise ValueError("tone curve 输出必须位于 0..1")


@dataclass(frozen=True, slots=True)
class PhotometricEstimate:
    """PhotoCalibNet 应输出的物理受限参数，而非新 RGB 图。"""

    white_balance_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_matrix: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    exposure_stops: float = 0.0
    tone_curve: MonotonicToneCurve = field(default_factory=MonotonicToneCurve)
    illumination_gain: np.ndarray | None = None
    confidence: float = 0.0
    source: str = "manual"

    def __post_init__(self) -> None:
        gains = np.asarray(self.white_balance_gains, dtype=np.float32)
        matrix = np.asarray(self.color_matrix, dtype=np.float32)
        if gains.shape != (3,) or np.any(~np.isfinite(gains)) or np.any(gains <= 0):
            raise ValueError("白平衡增益必须是三个有限正数")
        if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
            raise ValueError("色彩矩阵必须是有限的 3×3 数组")
        if not np.isfinite(self.exposure_stops) or abs(self.exposure_stops) > 3.0:
            raise ValueError("曝光校正必须位于 ±3 stops")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("光度估计置信度必须位于 [0,1]")
        if self.illumination_gain is not None:
            field_values = np.asarray(self.illumination_gain, dtype=np.float32)
            if field_values.ndim not in (2, 3) or np.any(~np.isfinite(field_values)):
                raise ValueError("illumination gain 必须是有限二维或三维数组")
            object.__setattr__(self, "illumination_gain", field_values.copy())


def apply_photometric_correction(
    image_rgb: np.ndarray,
    estimate: PhotometricEstimate,
    *,
    max_gain: float = 1.6,
) -> np.ndarray:
    """以白平衡、矩阵、平滑光场和单调曲线校正 RGB float32。

    所有变化均能追溯到有限参数；函数不会使用纹理生成先验，也不会原地改写输入。
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.float32:
        raise ValueError("光度校正需要 H×W×3 RGB float32 图像")
    if not 1.0 <= max_gain <= 3.0:
        raise ValueError("max_gain 必须位于 1..3")
    output = np.clip(image_rgb, 0.0, 1.0).astype(np.float32, copy=True)
    gains = np.clip(np.asarray(estimate.white_balance_gains, np.float32), 1.0 / max_gain, max_gain)
    output *= gains.reshape(1, 1, 3)
    if estimate.illumination_gain is not None:
        illumination = estimate.illumination_gain
        if illumination.ndim == 2:
            illumination = illumination[:, :, None]
        illumination = cv2.resize(
            illumination,
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        if illumination.ndim == 2:
            illumination = illumination[:, :, None]
        output *= np.clip(illumination, 1.0 / max_gain, max_gain)
    matrix = np.asarray(estimate.color_matrix, dtype=np.float32)
    # 限制矩阵偏离 identity 的幅度，阻止预测器借色彩矩阵完成内容重绘。
    matrix = np.eye(3, dtype=np.float32) + np.clip(
        matrix - np.eye(3, dtype=np.float32),
        -0.35,
        0.35,
    )
    output = np.einsum("...c,dc->...d", output, matrix, optimize=True)
    output *= float(2.0 ** estimate.exposure_stops)
    curve_input = np.asarray(estimate.tone_curve.input_knots, np.float32)
    curve_output = np.asarray(estimate.tone_curve.output_knots, np.float32)
    output = np.interp(np.clip(output, 0.0, 1.0), curve_input, curve_output).astype(np.float32)
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0))
