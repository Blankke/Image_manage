"""经典增强算子的契约与基本质量测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.operators.denoise import DenoiseMode, DenoiseOperator, DenoiseParameters
from screenrestore.operators.exposure import ExposureOperator, ExposureParameters
from screenrestore.operators.local_contrast import ClaheOperator, ClaheParameters
from screenrestore.operators.sharpen import SharpenOperator, SharpenParameters
from screenrestore.operators.white_balance import (
    WhiteBalanceMode,
    WhiteBalanceOperator,
    WhiteBalanceParameters,
)


def test_gray_world_reduces_channel_cast() -> None:
    image = np.full((80, 120, 3), (180, 100, 70), dtype=np.uint8)
    output = WhiteBalanceOperator().apply(
        image,
        WhiteBalanceParameters(mode=WhiteBalanceMode.GRAY_WORLD),
        ProcessingContext(),
    )
    before = np.ptp(image.mean(axis=(0, 1)))
    after = np.ptp(output.mean(axis=(0, 1)))
    assert after < before * 0.1


def test_exposure_gamma_is_monotonic_and_non_destructive() -> None:
    image = np.tile(np.arange(256, dtype=np.uint8), (20, 1))[..., None]
    image = np.repeat(image, 3, axis=2)
    original = image.copy()
    output = ExposureOperator().apply(
        image,
        ExposureParameters(exposure=0.5, gamma=1.2),
        ProcessingContext(),
    )
    assert np.array_equal(image, original)
    assert np.all(np.diff(output[10, :, 0].astype(np.int16)) >= 0)
    assert float(output.mean()) > float(image.mean())


def test_denoise_reduces_flat_region_variance() -> None:
    generator = np.random.default_rng(7)
    clean = np.full((96, 96, 3), 128, np.uint8)
    noise = generator.normal(0, 18, clean.shape)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    output = DenoiseOperator().apply(
        noisy,
        DenoiseParameters(mode=DenoiseMode.GAUSSIAN, strength=0.8, radius=1.4),
        ProcessingContext(),
    )
    assert float(output.std()) < float(noisy.std()) * 0.55


def test_clahe_and_unsharp_preserve_rgb_contract() -> None:
    gradient = np.tile(np.linspace(80, 150, 160, dtype=np.uint8), (100, 1))
    image = np.dstack((gradient, gradient, gradient))
    cv2.line(image, (80, 0), (80, 99), (200, 200, 200), 2)
    contrasted = ClaheOperator().apply(image, ClaheParameters(strength=0.4), ProcessingContext())
    sharpened = SharpenOperator().apply(
        contrasted,
        SharpenParameters(radius=1.0, amount=0.8),
        ProcessingContext(),
    )
    assert sharpened.shape == image.shape
    assert sharpened.dtype == np.uint8
    assert not np.shares_memory(sharpened, image)

