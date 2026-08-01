"""可预测范围的曝光、Gamma、色调和色彩算子。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from screenrestore.core.color import linear_to_srgb, srgb_to_linear
from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


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
    auto_black_level_strength: float = 0.0
    black_level_quantile: float = 0.01
    target_black_level: float = 0.002
    max_black_level_correction: float = 0.025
    black_level_activation_ceiling: float = 0.06
    auto_black_contrast: float = 0.0
    auto_white_background_strength: float = 0.0
    white_background_min_area: float = 0.65
    white_background_saturation_ceiling: float = 0.12
    white_background_luminance_floor: float = 0.55
    white_background_quantile: float = 0.9
    target_white_background: float = 0.985
    max_white_background_gain: float = 1.35

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
        require_range("auto_black_level_strength", self.auto_black_level_strength, 0.0, 1.0)
        require_range("black_level_quantile", self.black_level_quantile, 0.001, 0.2)
        require_range("target_black_level", self.target_black_level, 0.0, 0.1)
        require_range(
            "max_black_level_correction", self.max_black_level_correction, 0.001, 0.2
        )
        require_range(
            "black_level_activation_ceiling",
            self.black_level_activation_ceiling,
            0.005,
            0.3,
        )
        require_range("auto_black_contrast", self.auto_black_contrast, 0.0, 0.3)
        require_range(
            "auto_white_background_strength",
            self.auto_white_background_strength,
            0.0,
            1.0,
        )
        require_range("white_background_min_area", self.white_background_min_area, 0.1, 0.95)
        require_range(
            "white_background_saturation_ceiling",
            self.white_background_saturation_ceiling,
            0.01,
            0.5,
        )
        require_range(
            "white_background_luminance_floor",
            self.white_background_luminance_floor,
            0.2,
            0.95,
        )
        require_range("white_background_quantile", self.white_background_quantile, 0.5, 0.99)
        require_range("target_white_background", self.target_white_background, 0.7, 1.0)
        require_range("max_white_background_gain", self.max_white_background_gain, 1.0, 4.0)


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
        require_rgb_float(image)
        if params == ExposureParameters():
            return image.copy()
        value = image.copy()
        context.cancellation.check()
        # 摄影曝光必须在线性光中乘以 2^EV；其他 UI 色调仍在感知 sRGB 空间操作。
        if params.exposure != 0.0:
            linear = srgb_to_linear(value)
            linear = np.clip(linear * (2.0**params.exposure), 0.0, 1.0)
            value = linear_to_srgb(linear)
        if params.auto_white_background_strength > 0.0:
            value = _correct_evidence_gated_white_background(value, params, context)
        black_correction = 0.0
        observed_black = 0.0
        if params.auto_black_level_strength > 0.0:
            luminance = np.sum(
                value * np.array([0.2126, 0.7152, 0.0722], np.float32),
                axis=2,
            )
            observed_black = float(np.quantile(luminance, params.black_level_quantile))
            # 没有足够暗的内容时，单图无法区分“抬黑”和明亮场景，必须保持中性。
            if observed_black <= params.black_level_activation_ceiling:
                black_correction = min(
                    params.max_black_level_correction,
                    max(0.0, observed_black - params.target_black_level)
                    * params.auto_black_level_strength,
                )
                if black_correction > 0.0:
                    value = np.clip(
                        (value - black_correction) / max(0.65, 1.0 - black_correction),
                        0.0,
                        1.0,
                    )
            context.metadata["auto_black_level"] = {
                "observed_quantile": observed_black,
                "correction": black_correction,
                "activated": black_correction > 0.0,
            }
        value = np.power(np.maximum(value, 1e-6), 1.0 / params.gamma)

        luminance = np.sum(value * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=2, keepdims=True)
        shadow_mask = np.square(1.0 - luminance)
        highlight_mask = np.square(luminance)
        value += params.shadows * 0.45 * shadow_mask * (1.0 - value)
        value += params.highlights * 0.45 * highlight_mask * (1.0 - value)
        value += params.blacks * 0.12 * np.clip(1.0 - luminance * 4.0, 0.0, 1.0)
        value += params.whites * 0.12 * np.clip((luminance - 0.75) * 4.0, 0.0, 1.0)

        adaptive_contrast = params.auto_black_contrast if black_correction > 0.0 else 0.0
        contrast_factor = 2.0 ** (params.contrast + adaptive_contrast)
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
        return clip_float(value)


def _correct_evidence_gated_white_background(
    value: np.ndarray,
    params: ExposureParameters,
    context: ProcessingContext,
) -> np.ndarray:
    """仅在大面积低饱和亮底存在时，用线性光增益恢复屏幕白场。"""

    luminance = np.sum(
        value * np.array([0.2126, 0.7152, 0.0722], np.float32),
        axis=2,
    )
    channel_max = np.max(value, axis=2)
    channel_min = np.min(value, axis=2)
    saturation = (channel_max - channel_min) / np.maximum(channel_max, 1e-6)
    evidence_mask = (
        (saturation <= params.white_background_saturation_ceiling)
        & (luminance >= params.white_background_luminance_floor)
    )
    evidence_area = float(np.mean(evidence_mask))
    observed_white = 0.0
    applied_gain = 1.0
    if evidence_area >= params.white_background_min_area:
        observed_white = float(
            np.quantile(luminance[evidence_mask], params.white_background_quantile)
        )
        if observed_white < params.target_white_background:
            observed_linear = _srgb_scalar_to_linear(observed_white)
            target_linear = _srgb_scalar_to_linear(params.target_white_background)
            requested_gain = target_linear / max(observed_linear, 1e-6)
            bounded_gain = min(params.max_white_background_gain, max(1.0, requested_gain))
            applied_gain = 1.0 + (
                bounded_gain - 1.0
            ) * params.auto_white_background_strength
            if applied_gain > 1.0:
                linear = srgb_to_linear(value)
                value = linear_to_srgb(np.clip(linear * applied_gain, 0.0, 1.0))
    context.metadata["auto_white_background"] = {
        "evidence_area": evidence_area,
        "observed_quantile": observed_white,
        "gain": applied_gain,
        "activated": applied_gain > 1.0,
    }
    return value


def _srgb_scalar_to_linear(value: float) -> float:
    """转换标量 sRGB，避免为白场统计构造临时三通道图。"""

    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4
