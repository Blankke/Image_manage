"""反光抑制模式测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.operators.reflection import (
    ReflectionMode,
    ReflectionOperator,
    ReflectionParameters,
)


def test_gradient_dct_mode_preserves_contract_and_reduces_soft_halo() -> None:
    height, width = 96, 144
    image = np.full((height, width, 3), 45, np.float32)
    image[24:76, 36:112] = (80, 125, 170)
    yy, xx = np.indices((height, width), dtype=np.float32)
    halo = 52.0 * np.exp(-0.5 * (np.square((xx - 72) / 31) + np.square((yy - 47) / 20)))
    degraded = np.clip(image + halo[..., None], 0, 255).astype(np.uint8)
    result = ReflectionOperator().apply(
        degraded,
        ReflectionParameters(
            mode=ReflectionMode.GRADIENT_DCT,
            gradient_threshold=0.018,
            strength=0.45,
        ),
        ProcessingContext(),
    )
    assert result.dtype == np.uint8
    assert result.shape == degraded.shape
    source_gray = cv2.cvtColor(degraded, cv2.COLOR_RGB2GRAY).astype(np.float32)
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_background_spread = float(source_gray[:20].std())
    result_background_spread = float(result_gray[:20].std())
    assert result_background_spread < source_background_spread
    assert float(result_gray[50, 110] - result_gray[50, 115]) > 8.0
