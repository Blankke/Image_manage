"""CPU 友好的多种降噪算法。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float, to_float, to_uint8


class DenoiseMode(StrEnum):
    """降噪模式。"""

    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    NLM = "nlm"
    LUMA_CHROMA = "luma_chroma"


@dataclass
class DenoiseParameters(ParameterModel):
    """统一降噪参数，strength 控制最终混合。"""

    mode: DenoiseMode = DenoiseMode.LUMA_CHROMA
    strength: float = 0.25
    radius: float = 1.2
    color_sigma: float = 35.0
    luma_strength: float = 3.0
    chroma_strength: float = 7.0

    def validate(self) -> None:
        require_range("strength", self.strength, 0.0, 1.0)
        require_range("radius", self.radius, 0.1, 10.0)
        require_range("color_sigma", self.color_sigma, 1.0, 200.0)
        require_range("luma_strength", self.luma_strength, 0.0, 30.0)
        require_range("chroma_strength", self.chroma_strength, 0.0, 40.0)


class DenoiseOperator(ImageOperator[DenoiseParameters]):
    """Gaussian、双边、彩色 NLM 和亮色分离降噪。"""

    id = "denoise"
    display_name = "降噪"
    parameter_type = DenoiseParameters

    def default_parameters(self) -> DenoiseParameters:
        return DenoiseParameters()

    def apply(
        self,
        image: np.ndarray,
        params: DenoiseParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        context.cancellation.check()
        if params.strength == 0:
            return image.copy()
        if params.mode == DenoiseMode.GAUSSIAN:
            filtered = cv2.GaussianBlur(image, (0, 0), params.radius)
        elif params.mode == DenoiseMode.BILATERAL:
            diameter = max(3, int(round(params.radius * 4)) | 1)
            filtered = cv2.bilateralFilter(
                image, diameter, params.color_sigma, max(1.0, params.radius * 3)
            )
        elif params.mode == DenoiseMode.NLM:
            # OpenCV 的接口以 BGR 命名但对通道顺序对称；为契约清晰仍显式转换。
            image_bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
            search_window = 15 if context.preview else 21
            filtered_bgr = cv2.fastNlMeansDenoisingColored(
                image_bgr,
                None,
                params.luma_strength,
                params.chroma_strength,
                7,
                search_window,
            )
            filtered = to_float(cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB))
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
            luma_sigma = max(0.1, params.radius * max(0.2, params.luma_strength / 5.0))
            chroma_sigma = max(0.1, params.radius * max(0.3, params.chroma_strength / 5.0))
            y_filtered = cv2.GaussianBlur(y_channel, (0, 0), luma_sigma)
            sigma_color = params.color_sigma / 255.0
            cr_filtered = cv2.bilateralFilter(
                cr_channel,
                0,
                sigma_color,
                chroma_sigma * 3,
            )
            cb_filtered = cv2.bilateralFilter(
                cb_channel,
                0,
                sigma_color,
                chroma_sigma * 3,
            )
            filtered = cv2.cvtColor(
                cv2.merge((y_filtered, cr_filtered, cb_filtered)), cv2.COLOR_YCrCb2RGB
            )
        return clip_float(
            cv2.addWeighted(image, 1.0 - params.strength, filtered, params.strength, 0)
        )
