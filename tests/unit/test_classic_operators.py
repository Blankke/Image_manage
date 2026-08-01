"""经典增强算子的契约与基本质量测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.operators.dehalo import DehaloOperator, DehaloParameters
from screenrestore.operators.denoise import DenoiseMode, DenoiseOperator, DenoiseParameters
from screenrestore.operators.exposure import ExposureOperator, ExposureParameters
from screenrestore.operators.local_contrast import ClaheOperator, ClaheParameters
from screenrestore.operators.sharpen import SharpenOperator, SharpenParameters
from screenrestore.operators.white_balance import (
    WhiteBalanceMode,
    WhiteBalanceOperator,
    WhiteBalanceParameters,
)


def _working(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def test_gray_world_reduces_channel_cast() -> None:
    image = np.full((80, 120, 3), (180, 100, 70), dtype=np.uint8)
    output = WhiteBalanceOperator().apply(
        _working(image),
        WhiteBalanceParameters(mode=WhiteBalanceMode.GRAY_WORLD, max_gain=4.0),
        ProcessingContext(),
    )
    before = np.ptp(_working(image).mean(axis=(0, 1)))
    after = np.ptp(output.mean(axis=(0, 1)))
    assert after < before * 0.1


def test_exposure_gamma_is_monotonic_and_non_destructive() -> None:
    image = np.tile(np.arange(256, dtype=np.uint8), (20, 1))[..., None]
    image = np.repeat(image, 3, axis=2)
    original = image.copy()
    output = ExposureOperator().apply(
        _working(image),
        ExposureParameters(exposure=0.5, gamma=1.2),
        ProcessingContext(),
    )
    assert np.array_equal(image, original)
    assert np.all(np.diff(output[10, :, 0].astype(np.int16)) >= 0)
    assert float(output.mean()) > float(_working(image).mean())


def test_auto_black_level_requires_dark_scene_evidence() -> None:
    operator = ExposureOperator()
    raised_black = np.linspace(0.02, 0.8, 72, dtype=np.float32)[None, :, None]
    raised_black = np.broadcast_to(raised_black, (48, 72, 3)).copy()
    context = ProcessingContext()
    corrected = operator.apply(
        raised_black,
        ExposureParameters(
            auto_black_level_strength=1.0,
            black_level_quantile=0.01,
            target_black_level=0.002,
            max_black_level_correction=0.025,
        ),
        context,
    )
    assert corrected.dtype == np.float32
    assert float(corrected.min()) < float(raised_black.min())
    metadata = context.metadata["auto_black_level"]
    assert isinstance(metadata, dict) and metadata["activated"] is True

    bright = np.full((32, 40, 3), 0.4, np.float32)
    bright_context = ProcessingContext()
    unchanged = operator.apply(
        bright,
        ExposureParameters(auto_black_level_strength=1.0),
        bright_context,
    )
    assert np.allclose(unchanged, bright, atol=1e-6)
    bright_metadata = bright_context.metadata["auto_black_level"]
    assert isinstance(bright_metadata, dict) and bright_metadata["activated"] is False


def test_auto_white_background_requires_large_low_chroma_bright_area() -> None:
    operator = ExposureOperator()
    gray_screen = np.full((60, 80, 3), 0.78, np.float32)
    gray_screen[20:40, 30:50] = (0.7, 0.15, 0.1)
    context = ProcessingContext()
    corrected = operator.apply(
        gray_screen,
        ExposureParameters(auto_white_background_strength=1.0),
        context,
    )
    assert float(corrected.mean()) > float(gray_screen.mean()) + 0.08
    metadata = context.metadata["auto_white_background"]
    assert isinstance(metadata, dict) and metadata["activated"] is True
    assert float(metadata["evidence_area"]) > 0.8

    ordinary_photo = np.full((60, 80, 3), (0.75, 0.42, 0.2), np.float32)
    ordinary_context = ProcessingContext()
    unchanged = operator.apply(
        ordinary_photo,
        ExposureParameters(auto_white_background_strength=1.0),
        ordinary_context,
    )
    assert np.allclose(unchanged, ordinary_photo, atol=1e-6)
    ordinary_metadata = ordinary_context.metadata["auto_white_background"]
    assert isinstance(ordinary_metadata, dict) and ordinary_metadata["activated"] is False


def test_dehalo_is_evidence_gated_and_reduces_bright_surround() -> None:
    image = np.full((80, 120, 3), 0.12, np.float32)
    image[30:50, 50:70] = 0.95
    bloom = cv2.GaussianBlur(image, (0, 0), 2.4)
    bloomed = np.clip(image * 0.82 + bloom * 0.18, 0.0, 1.0).astype(np.float32)
    context = ProcessingContext()
    corrected = DehaloOperator().apply(
        bloomed,
        DehaloParameters(strength=0.8, max_scene_median=0.3),
        context,
    )
    surround = np.ones(image.shape[:2], dtype=bool)
    surround[28:52, 48:72] = False
    assert float(corrected[surround].mean()) < float(bloomed[surround].mean())
    # 高亮核心不是待扣除的扩散层，防止把真实发光对象整体压暗。
    assert np.allclose(corrected[40, 60], bloomed[40, 60], atol=2e-3)
    assert context.metadata["dehalo"]["activated"] is True

    bright_scene = np.full((40, 60, 3), 0.8, np.float32)
    bright_context = ProcessingContext()
    unchanged = DehaloOperator().apply(
        bright_scene,
        DehaloParameters(),
        bright_context,
    )
    assert np.array_equal(unchanged, bright_scene)
    assert bright_context.metadata["dehalo"]["activated"] is False


def test_denoise_reduces_flat_region_variance() -> None:
    generator = np.random.default_rng(7)
    clean = np.full((96, 96, 3), 128, np.uint8)
    noise = generator.normal(0, 18, clean.shape)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    output = DenoiseOperator().apply(
        _working(noisy),
        DenoiseParameters(mode=DenoiseMode.GAUSSIAN, strength=0.8, radius=1.4),
        ProcessingContext(),
    )
    assert float(output.std()) < float(_working(noisy).std()) * 0.55


def test_clahe_and_unsharp_preserve_rgb_contract() -> None:
    gradient = np.tile(np.linspace(80, 150, 160, dtype=np.uint8), (100, 1))
    image = np.dstack((gradient, gradient, gradient))
    cv2.line(image, (80, 0), (80, 99), (200, 200, 200), 2)
    working = _working(image)
    contrasted = ClaheOperator().apply(
        working,
        ClaheParameters(strength=0.4),
        ProcessingContext(),
    )
    sharpened = SharpenOperator().apply(
        contrasted,
        SharpenParameters(radius=1.0, amount=0.8),
        ProcessingContext(),
    )
    assert sharpened.shape == image.shape
    assert sharpened.dtype == np.float32
    assert not np.shares_memory(sharpened, working)
