"""低频照明场估计与不均匀光照校正。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


class IlluminationEstimator(StrEnum):
    """照明场估计方式。"""

    GAUSSIAN = "gaussian"
    MORPHOLOGY = "morphology"


class IlluminationModel(StrEnum):
    """图像与照明场的组合模型。"""

    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"


@dataclass
class IlluminationParameters(ParameterModel):
    """低频照明校正参数。"""

    estimator: IlluminationEstimator = IlluminationEstimator.GAUSSIAN
    model: IlluminationModel = IlluminationModel.MULTIPLICATIVE
    radius: int = 90
    strength: float = 0.25
    protect_shadows: float = 0.6
    show_field: bool = False

    def validate(self) -> None:
        if not 8 <= self.radius <= 600:
            raise ValueError("radius 必须位于 8..600")
        require_range("strength", self.strength, 0.0, 1.0)
        require_range("protect_shadows", self.protect_shadows, 0.0, 1.0)


class IlluminationOperator(ImageOperator[IlluminationParameters]):
    """仅校正 LAB 亮度通道，保留彩色内容与暗部层次。"""

    id = "illumination"
    display_name = "光照不均匀校正"
    parameter_type = IlluminationParameters

    def default_parameters(self) -> IlluminationParameters:
        return IlluminationParameters()

    def apply(
        self,
        image: np.ndarray,
        params: IlluminationParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        if params.strength == 0:
            return image.copy()
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness = lab[..., 0] / 100.0
        context.report(0.15, "估计低频照明场")
        if params.estimator == IlluminationEstimator.GAUSSIAN:
            field = cv2.GaussianBlur(lightness, (0, 0), params.radius / 3.0)
        else:
            kernel_size = min(params.radius * 2 + 1, min(lightness.shape) | 1)
            kernel_size = max(3, kernel_size | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            field = cv2.morphologyEx(lightness, cv2.MORPH_OPEN, kernel)
        target = float(np.median(field))
        if params.model == IlluminationModel.MULTIPLICATIVE:
            corrected = lightness * target / np.maximum(field, 0.04)
        else:
            corrected = lightness + (target - field)
        corrected = np.clip(corrected, 0.0, 1.0)
        shadow_protection = 1.0 - params.protect_shadows * np.square(1.0 - lightness)
        mix = params.strength * shadow_protection
        output_lightness = lightness * (1.0 - mix) + corrected * mix
        output_lab = lab.copy()
        output_lab[..., 0] = np.clip(output_lightness * 100.0, 0.0, 100.0)
        if params.show_field:
            context.metadata["illumination_field"] = np.clip(
                cv2.resize(field, (min(512, field.shape[1]), min(512, field.shape[0]))), 0, 1
            )
        context.report(1.0, "光照校正完成")
        return clip_float(cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB))
