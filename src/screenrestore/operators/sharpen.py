"""带阈值和明暗保护的非锐化蒙版。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


@dataclass
class SharpenParameters(ParameterModel):
    """Unsharp Mask 参数。"""

    radius: float = 1.2
    amount: float = 0.6
    threshold: float = 0.02
    highlight_protection: float = 0.35
    shadow_protection: float = 0.25

    def validate(self) -> None:
        require_range("radius", self.radius, 0.1, 10.0)
        require_range("amount", self.amount, 0.0, 3.0)
        require_range("threshold", self.threshold, 0.0, 0.25)
        require_range("highlight_protection", self.highlight_protection, 0.0, 1.0)
        require_range("shadow_protection", self.shadow_protection, 0.0, 1.0)


class SharpenOperator(ImageOperator[SharpenParameters]):
    """使用 Gaussian 低通构造细节层，不依赖固定 3×3 核。"""

    id = "sharpen"
    display_name = "锐化"
    parameter_type = SharpenParameters

    def default_parameters(self) -> SharpenParameters:
        return SharpenParameters()

    def apply(
        self,
        image: np.ndarray,
        params: SharpenParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        self.validate(params)
        require_rgb_float(image)
        source = image
        context.cancellation.check()
        if params.amount == 0:
            return image.copy()
        blurred = cv2.GaussianBlur(source, (0, 0), params.radius)
        detail = source - blurred
        magnitude = np.max(np.abs(detail), axis=2, keepdims=True)
        threshold_mask = np.clip((magnitude - params.threshold) / max(0.005, params.threshold + 0.01), 0, 1)
        luminance = np.sum(source * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=2, keepdims=True)
        highlight_mask = 1.0 - params.highlight_protection * np.square(luminance)
        shadow_mask = 1.0 - params.shadow_protection * np.square(1.0 - luminance)
        result = source + detail * params.amount * threshold_mask * highlight_mask * shadow_mask
        return clip_float(result)
