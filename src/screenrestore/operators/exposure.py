"""可预测范围的曝光、Gamma、色调和色彩算子。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import require_range, to_float, to_uint8


@dataclass
class ExposureParameters(ParameterModel):
    """基础色调参数；滑杆中性值均为 0，Gamma 中性值为 1。"""

    exposure: float = 0.0
    gamma: float = 1.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    blacks: float = 0.0
    whites: float = 0.0
    saturation: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0

    def validate(self) -> None:
        require_range("exposure", self.exposure, -4.0, 4.0)
        require_range("gamma", self.gamma, 0.2, 5.0)
        for name in (
            "contrast",
            "highlights",
            "shadows",
            "blacks",
            "whites",
            "saturation",
            "temperature",
            "tint",
        ):
            require_range(name, float(getattr(self, name)), -1.0, 1.0)


class ExposureOperator(ImageOperator[ExposureParameters]):
    """在浮点 RGB 上执行稳定、连续的基础色调变换。"""

    id = "exposure"
    display_name = "曝光与 Gamma"
    parameter_type = ExposureParameters

    def default_parameters(self) -> ExposureParameters:
        return ExposureParameters()

    def apply(
        self,
        image: np.ndarray,
        params: ExposureParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        self.validate(params)
        value = to_float(image)
        context.cancellation.check()
        # 曝光以摄影档位为单位，Gamma 使用常见的 1/gamma 映射。
        value = np.clip(value * (2.0**params.exposure), 0.0, 1.0)
        value = np.power(np.maximum(value, 1e-6), 1.0 / params.gamma)

        luminance = np.sum(value * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=2, keepdims=True)
        shadow_mask = np.square(1.0 - luminance)
        highlight_mask = np.square(luminance)
        value += params.shadows * 0.45 * shadow_mask * (1.0 - value)
        value += params.highlights * 0.45 * highlight_mask * (1.0 - value)
        value += params.blacks * 0.12 * np.clip(1.0 - luminance * 4.0, 0.0, 1.0)
        value += params.whites * 0.12 * np.clip((luminance - 0.75) * 4.0, 0.0, 1.0)

        contrast_factor = 2.0**params.contrast
        value = (value - 0.5) * contrast_factor + 0.5
        luminance = np.sum(value * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=2, keepdims=True)
        value = luminance + (value - luminance) * (1.0 + params.saturation)

        # 温度近似沿红/蓝轴，Tint 沿绿/洋红轴；幅度受限以保持可预测。
        color_gain = np.array(
            [
                1.0 + 0.18 * params.temperature + 0.08 * params.tint,
                1.0 - 0.12 * params.tint,
                1.0 - 0.18 * params.temperature + 0.08 * params.tint,
            ],
            dtype=np.float32,
        )
        value *= color_gain.reshape(1, 1, 3)
        return to_uint8(value)

