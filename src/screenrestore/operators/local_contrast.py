"""只处理 LAB 亮度通道的 CLAHE 算子。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import require_range, require_rgb_u8


@dataclass
class ClaheParameters(ParameterModel):
    """局部对比度参数。"""

    clip_limit: float = 1.5
    tile_grid_size: int = 8
    strength: float = 0.35

    def validate(self) -> None:
        require_range("clip_limit", self.clip_limit, 0.1, 8.0)
        if not 2 <= self.tile_grid_size <= 32:
            raise ValueError("tile_grid_size 必须位于 2..32")
        require_range("strength", self.strength, 0.0, 1.0)


class ClaheOperator(ImageOperator[ClaheParameters]):
    """在 LAB 的 L 通道上应用 CLAHE 并与原亮度混合。"""

    id = "clahe"
    display_name = "局部对比度"
    parameter_type = ClaheParameters

    def default_parameters(self) -> ClaheParameters:
        return ClaheParameters()

    def apply(
        self,
        image: np.ndarray,
        params: ClaheParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_u8(image)
        self.validate(params)
        context.cancellation.check()
        if params.strength == 0:
            return image.copy()
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced = cv2.createCLAHE(
            clipLimit=params.clip_limit,
            tileGridSize=(params.tile_grid_size, params.tile_grid_size),
        ).apply(lightness)
        mixed = cv2.addWeighted(lightness, 1.0 - params.strength, enhanced, params.strength, 0)
        return cv2.cvtColor(cv2.merge((mixed, channel_a, channel_b)), cv2.COLOR_LAB2RGB)

