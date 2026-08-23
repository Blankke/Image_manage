"""受约束摄影校正和残差恢复的数值边界测试。"""

from __future__ import annotations

import numpy as np
import pytest

from screenrestore.restoration import (
    MonotonicToneCurve,
    PhotometricEstimate,
    apply_bounded_residual,
    apply_photometric_correction,
)


def test_identity_photometric_correction_is_pixel_stable_and_read_only() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((48, 64, 3), dtype=np.float32)
    source_copy = image.copy()

    output = apply_photometric_correction(image, PhotometricEstimate())

    assert output.dtype == np.float32
    assert np.max(np.abs(output - image)) < 1e-6
    assert np.array_equal(image, source_copy)
    assert not np.shares_memory(output, image)


def test_photometric_parameters_remain_bounded_and_monotonic() -> None:
    ramp = np.linspace(0, 1, 80, dtype=np.float32)[None, :, None]
    image = np.broadcast_to(ramp, (24, 80, 3)).copy()
    estimate = PhotometricEstimate(
        white_balance_gains=(1.2, 1.0, 0.85),
        color_matrix=((1.08, -0.03, 0.0), (0.01, 1.02, 0.0), (0.0, -0.02, 1.06)),
        exposure_stops=0.15,
        tone_curve=MonotonicToneCurve(
            input_knots=(0.0, 0.3, 0.7, 1.0),
            output_knots=(0.0, 0.25, 0.76, 1.0),
        ),
        illumination_gain=np.linspace(0.9, 1.1, 12, dtype=np.float32).reshape(3, 4),
        confidence=0.8,
    )

    output = apply_photometric_correction(image, estimate)

    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert np.all(np.diff(output[12, :, 1]) >= -1e-6)


def test_non_monotonic_tone_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="单调"):
        MonotonicToneCurve((0.0, 0.5, 1.0), (0.0, 0.8, 0.7))


def test_residual_cannot_exceed_declared_delta() -> None:
    image = np.full((20, 30, 3), 0.5, np.float32)
    residual = np.full_like(image, 2.0)

    output = apply_bounded_residual(image, residual, 0.75, max_delta=0.04)

    assert np.allclose(output, 0.53)
    assert np.all(image == 0.5)
