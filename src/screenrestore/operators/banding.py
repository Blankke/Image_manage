"""单图水平/垂直亮度条带估计与有限幅校正。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from screenrestore.core.color import linear_to_srgb, srgb_to_linear
from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


class BandingDirection(StrEnum):
    """条带变化方向；水平条带对应逐行曲线。"""

    AUTO = "auto"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class BandingParameters(ParameterModel):
    """条带估计参数。"""

    direction: BandingDirection = BandingDirection.AUTO
    smooth_scale: float = 32.0
    max_correction: float = 0.22
    strength: float = 0.65
    broad_haze_strength: float = 0.0
    broad_haze_scale: float = 90.0
    black_level_quantile: float = 0.04
    max_haze_correction: float = 0.12
    show_curve: bool = False

    def validate(self) -> None:
        require_range("smooth_scale", self.smooth_scale, 4.0, 400.0)
        require_range("max_correction", self.max_correction, 0.01, 0.5)
        require_range("strength", self.strength, 0.0, 1.0)
        require_range("broad_haze_strength", self.broad_haze_strength, 0.0, 2.0)
        require_range("broad_haze_scale", self.broad_haze_scale, 12.0, 600.0)
        require_range("black_level_quantile", self.black_level_quantile, 0.01, 0.3)
        require_range("max_haze_correction", self.max_haze_correction, 0.01, 0.35)


class BandingOperator(ImageOperator[BandingParameters]):
    """从亮度的行/列统计中分离趋势与周期分量，再施加受限增益。"""

    id = "banding"
    display_name = "条带校正"
    parameter_type = BandingParameters

    def default_parameters(self) -> BandingParameters:
        return BandingParameters()

    def apply(
        self,
        image: np.ndarray,
        params: BandingParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        if params.strength == 0 and params.broad_haze_strength == 0:
            return image.copy()
        context.report(0.1, "估计条带方向")
        haze_corrected, haze_profile = _suppress_broad_haze(image, params)
        lab = cv2.cvtColor(haze_corrected, cv2.COLOR_RGB2LAB)
        lightness = lab[..., 0] / 100.0
        # 二维低通先压制文字、细线和像素级纹理，避免其支配行列统计。
        content_suppressed = cv2.GaussianBlur(lightness, (0, 0), 2.2)
        row_profile = np.mean(content_suppressed, axis=1)
        column_profile = np.mean(content_suppressed, axis=0)
        row_periodic, row_trend, row_energy = _separate_profile(row_profile, params.smooth_scale)
        column_periodic, column_trend, column_energy = _separate_profile(
            column_profile, params.smooth_scale
        )
        direction = params.direction
        if direction == BandingDirection.AUTO:
            direction = (
                BandingDirection.HORIZONTAL
                if row_energy >= column_energy
                else BandingDirection.VERTICAL
            )
        if direction == BandingDirection.HORIZONTAL:
            periodic, trend = row_periodic, row_trend
            profile = row_profile
            gain = _limited_gain(profile, trend, periodic, params.max_correction)
            gain_field = gain[:, None]
        else:
            periodic, trend = column_periodic, column_trend
            profile = column_profile
            gain = _limited_gain(profile, trend, periodic, params.max_correction)
            gain_field = gain[None, :]
        context.report(0.55, "应用有限幅亮度校正")
        corrected_lightness = np.clip(lightness * gain_field, 0.0, 1.0)
        mixed = lightness * (1.0 - params.strength) + corrected_lightness * params.strength
        output_lab = lab.copy()
        output_lab[..., 0] = np.clip(mixed * 100.0, 0.0, 100.0)
        context.metadata["banding"] = {
            "direction": direction.value,
            "horizontal_energy": row_energy,
            "vertical_energy": column_energy,
            "profile": profile.tolist() if params.show_curve else [],
            "trend": trend.tolist() if params.show_curve else [],
            "periodic": periodic.tolist() if params.show_curve else [],
            "gain": gain.tolist() if params.show_curve else [],
            "broad_haze": haze_profile.tolist() if params.show_curve else [],
        }
        context.report(1.0, "条带校正完成")
        return clip_float(cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB))


def _suppress_broad_haze(
    image_rgb: np.ndarray,
    params: BandingParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """估计全宽同步抬升的黑位，并按加性光幕模型恢复。

    手机拍摄投影/银幕时，滚动快门、环境散射和镜头鬼影可能形成很宽的水平
    灰雾带。普通周期增益会把它误当作内容趋势。这里使用每行低分位亮度近似
    黑位，减去大尺度趋势后只压制正向抬升，再用 ``I=(T-v)/(1-v)`` 恢复。
    该假设无法重建已经饱和或被反射覆盖的细节，因此校正量始终有限幅。
    """

    height = image_rgb.shape[0]
    empty = np.zeros(height, dtype=np.float32)
    if params.broad_haze_strength == 0:
        return image_rgb.copy(), empty
    source = srgb_to_linear(image_rgb)
    luminance = np.sum(
        source * np.array([0.2126, 0.7152, 0.0722], np.float32),
        axis=2,
    )
    black_profile = np.quantile(
        luminance,
        params.black_level_quantile,
        axis=1,
    ).astype(np.float32)
    slow_trend = gaussian_filter1d(
        black_profile,
        params.broad_haze_scale,
        mode="reflect",
    )
    haze = gaussian_filter1d(black_profile - slow_trend, 6.0, mode="reflect")
    haze = np.clip(
        haze * params.broad_haze_strength,
        0.0,
        params.max_haze_correction,
    ).astype(np.float32)
    veil = haze[:, None, None]
    restored = np.clip(
        (source - veil) / np.maximum(1.0 - veil, 0.65),
        0.0,
        1.0,
    )
    return linear_to_srgb(clip_float(restored)), haze


def _separate_profile(profile: np.ndarray, smooth_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """分离缓慢趋势，并以远离直流的频谱能量衡量周期条带。"""

    trend = gaussian_filter1d(profile.astype(np.float32), smooth_scale, mode="reflect")
    periodic = profile - trend
    periodic = gaussian_filter1d(periodic, 1.0, mode="reflect")
    windowed = (periodic - periodic.mean()) * np.hanning(len(periodic))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(periodic))
    usable = frequencies >= max(2.0 / max(1, len(periodic)), 0.006)
    periodic_energy = float(np.sum(spectrum[usable]))
    total_energy = float(np.sum(spectrum) + 1e-8)
    rms = float(np.sqrt(np.mean(np.square(periodic))))
    energy = (periodic_energy / total_energy) * rms / max(0.03, float(np.mean(profile)))
    return periodic, trend, float(energy)


def _limited_gain(
    profile: np.ndarray,
    trend: np.ndarray,
    periodic: np.ndarray,
    max_correction: float,
) -> np.ndarray:
    """用趋势/观测获得增益，并做幅度和空间平滑限制。"""

    estimated_observation = trend + periodic
    raw = trend / np.maximum(estimated_observation, 0.05)
    raw = gaussian_filter1d(raw, 0.8, mode="reflect")
    limited = np.clip(raw, 1.0 - max_correction, 1.0 + max_correction)
    # 保持平均亮度，避免把校正误当成整体曝光调整。
    limited /= max(1e-6, float(np.mean(limited)))
    return np.clip(limited, 1.0 - max_correction, 1.0 + max_correction).astype(np.float32)
