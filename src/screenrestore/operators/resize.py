"""固定在流水线末端的输出缩放。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


class ResizeMode(StrEnum):
    """输出缩放模式。"""

    ORIGINAL = "original"
    SCALE = "scale"
    FIT = "fit"


@dataclass
class ResizeParameters(ParameterModel):
    """输出尺寸参数。"""

    mode: ResizeMode = ResizeMode.ORIGINAL
    scale: float = 1.0
    max_width: int = 3840
    max_height: int = 2160

    def validate(self) -> None:
        require_range("scale", self.scale, 0.05, 8.0)
        if not 1 <= self.max_width <= 65535 or not 1 <= self.max_height <= 65535:
            raise ValueError("目标宽高必须位于 1..65535")


class ResizeOperator(ImageOperator[ResizeParameters]):
    """按比例或包围盒缩放，且不会把 FIT 模式意外放大。"""

    id = "resize"
    display_name = "输出缩放"
    parameter_type = ResizeParameters
    reorderable = False

    def default_parameters(self) -> ResizeParameters:
        return ResizeParameters()

    def apply(
        self,
        image: np.ndarray,
        params: ResizeParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        if params.mode == ResizeMode.ORIGINAL:
            return image.copy()
        height, width = image.shape[:2]
        if params.mode == ResizeMode.SCALE:
            scale = params.scale
        else:
            scale = min(1.0, params.max_width / width, params.max_height / height)
        if abs(scale - 1.0) < 1e-6:
            return image.copy()
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        return clip_float(cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=interpolation,
        ))
