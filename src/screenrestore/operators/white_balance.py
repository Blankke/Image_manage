"""有限幅白平衡算法。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from screenrestore.core.color import linear_to_srgb, srgb_to_linear
from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


class WhiteBalanceMode(StrEnum):
    """白平衡模式。"""

    OFF = "off"
    GRAY_WORLD = "gray_world"
    WHITE_PATCH = "white_patch"
    MANUAL = "manual"
    NEUTRAL_POINT = "neutral_point"


@dataclass
class WhiteBalanceParameters(ParameterModel):
    """白平衡参数，增益上限避免极端色偏。"""

    mode: WhiteBalanceMode = WhiteBalanceMode.GRAY_WORLD
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    neutral_x: float = 0.5
    neutral_y: float = 0.5
    max_gain: float = 2.5
    strength: float = 1.0

    def validate(self) -> None:
        for name, value in (
            ("red_gain", self.red_gain),
            ("green_gain", self.green_gain),
            ("blue_gain", self.blue_gain),
        ):
            require_range(name, value, 0.25, 4.0)
        require_range("neutral_x", self.neutral_x, 0.0, 1.0)
        require_range("neutral_y", self.neutral_y, 0.0, 1.0)
        require_range("max_gain", self.max_gain, 1.0, 4.0)
        require_range("strength", self.strength, 0.0, 1.0)


class WhiteBalanceOperator(ImageOperator[WhiteBalanceParameters]):
    """Gray World、White Patch、手动增益和中性点白平衡。"""

    id = "white_balance"
    display_name = "白平衡"
    parameter_type = WhiteBalanceParameters

    def default_parameters(self) -> WhiteBalanceParameters:
        return WhiteBalanceParameters()

    def apply(
        self,
        image: np.ndarray,
        params: WhiteBalanceParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        self.validate(params)
        require_rgb_float(image)
        context.cancellation.check()
        if params.mode == WhiteBalanceMode.OFF or params.strength == 0:
            return image.copy()
        source = srgb_to_linear(image)
        if params.mode == WhiteBalanceMode.GRAY_WORLD:
            reference = np.mean(source, axis=(0, 1))
            target = float(np.mean(reference))
            gains = target / np.maximum(reference, 1e-4)
        elif params.mode == WhiteBalanceMode.WHITE_PATCH:
            reference = np.percentile(source.reshape(-1, 3), 99.2, axis=0)
            gains = float(np.mean(reference)) / np.maximum(reference, 1e-4)
        elif params.mode == WhiteBalanceMode.NEUTRAL_POINT:
            y = round(params.neutral_y * (source.shape[0] - 1))
            x = round(params.neutral_x * (source.shape[1] - 1))
            radius = 3
            patch = source[
                max(0, y - radius) : min(source.shape[0], y + radius + 1),
                max(0, x - radius) : min(source.shape[1], x + radius + 1),
            ]
            reference = np.mean(patch, axis=(0, 1))
            gains = float(np.mean(reference)) / np.maximum(reference, 1e-4)
        else:
            gains = np.array([params.red_gain, params.green_gain, params.blue_gain])
        gains = np.clip(gains, 1 / params.max_gain, params.max_gain).astype(np.float32)
        balanced = np.clip(source * gains.reshape(1, 1, 3), 0.0, 1.0)
        result = source * (1.0 - params.strength) + balanced * params.strength
        context.metadata["white_balance_gains"] = gains.tolist()
        return clip_float(linear_to_srgb(result))
