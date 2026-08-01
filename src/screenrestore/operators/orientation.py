"""显式 90 度旋转算子；EXIF Orientation 已由加载器处理。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import require_rgb_float


@dataclass
class OrientationParameters(ParameterModel):
    """显式旋转角度。"""

    rotation: int = 0

    def validate(self) -> None:
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("旋转仅支持 0/90/180/270")


class OrientationOperator(ImageOperator[OrientationParameters]):
    """无插值损失地旋转 RGB 图像。"""

    id = "orientation"
    display_name = "方向与旋转"
    parameter_type = OrientationParameters
    reorderable = False

    def default_parameters(self) -> OrientationParameters:
        return OrientationParameters()

    def apply(
        self,
        image: np.ndarray,
        params: OrientationParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        context.cancellation.check()
        if params.rotation == 0:
            return image.copy()
        code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[params.rotation]
        return cv2.rotate(image, code)
