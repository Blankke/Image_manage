"""屏幕专用恢复的合成质量测试。"""

from __future__ import annotations

import numpy as np
from tests.synthetic.generators import add_banding, add_color_moire, smooth_texture

from screenrestore.core.operator import ProcessingContext
from screenrestore.operators.banding import (
    BandingDirection,
    BandingOperator,
    BandingParameters,
)
from screenrestore.operators.demoire import (
    DemoireMode,
    DemoireOperator,
    DemoireParameters,
    frequency_spectrum,
    moire_heatmap,
)


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def test_horizontal_banding_correction_reduces_error() -> None:
    clean = smooth_texture()
    degraded = add_banding(clean, "horizontal")
    context = ProcessingContext()
    corrected = BandingOperator().apply(
        degraded,
        BandingParameters(
            direction=BandingDirection.AUTO,
            smooth_scale=34,
            max_correction=0.24,
            strength=0.9,
            show_curve=True,
        ),
        context,
    )
    assert _mae(corrected, clean) < _mae(degraded, clean) * 0.7
    assert context.metadata["banding"]["direction"] == "horizontal"
    assert context.metadata["banding"]["gain"]


def test_vertical_banding_correction_reduces_error() -> None:
    clean = smooth_texture()
    degraded = add_banding(clean, "vertical")
    corrected = BandingOperator().apply(
        degraded,
        BandingParameters(
            direction=BandingDirection.AUTO,
            smooth_scale=34,
            max_correction=0.24,
            strength=0.9,
        ),
        ProcessingContext(),
    )
    assert _mae(corrected, clean) < _mae(degraded, clean) * 0.72


def test_broad_haze_correction_lowers_additive_veil_error() -> None:
    """宽光幕会同步抬升黑位，低分位偏置校正应降低真实像素误差。"""

    clean = smooth_texture(420, 300)
    yy = np.arange(clean.shape[0], dtype=np.float32)
    veil = 0.11 * np.exp(-0.5 * np.square((yy - 155) / 30.0))
    degraded = np.clip(
        clean.astype(np.float32) / 255.0 * (1.0 - veil[:, None, None])
        + veil[:, None, None],
        0.0,
        1.0,
    )
    degraded_u8 = np.rint(degraded * 255.0).astype(np.uint8)
    corrected = BandingOperator().apply(
        degraded_u8,
        BandingParameters(
            strength=0.0,
            broad_haze_strength=1.6,
            broad_haze_scale=85.0,
            black_level_quantile=0.03,
            max_haze_correction=0.16,
        ),
        ProcessingContext(),
    )
    assert _mae(corrected, clean) < _mae(degraded_u8, clean) * 0.82


def test_chroma_demoire_reduces_synthetic_color_error() -> None:
    clean = smooth_texture()
    degraded = add_color_moire(clean)
    heat = moire_heatmap(degraded)
    corrected = DemoireOperator().apply(
        degraded,
        DemoireParameters(
            mode=DemoireMode.CHROMA,
            strength=0.9,
            chroma_radius=3.2,
            edge_protection=0.6,
            heat_threshold=0.05,
        ),
        ProcessingContext(),
    )
    assert heat.shape == clean.shape[:2]
    assert 0.0 <= float(heat.min()) <= float(heat.max()) <= 1.0
    assert _mae(corrected, clean) < _mae(degraded, clean)


def test_frequency_spectrum_has_rgb_display_contract() -> None:
    spectrum = frequency_spectrum(add_color_moire(smooth_texture(180, 120)))
    assert spectrum.shape == (120, 180, 3)
    assert spectrum.dtype == np.uint8
