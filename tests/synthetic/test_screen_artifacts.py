"""屏幕专用恢复的合成质量测试。"""

from __future__ import annotations

import numpy as np

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
from tests.synthetic.generators import add_banding, add_color_moire, smooth_texture


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    first_float = first.astype(np.float32) / (255.0 if first.dtype == np.uint8 else 1.0)
    second_float = second.astype(np.float32) / (255.0 if second.dtype == np.uint8 else 1.0)
    return float(np.mean(np.abs(first_float - second_float)))


def _working(image: np.ndarray) -> np.ndarray:
    """把合成器的 uint8 输出送入 float32 算子契约。"""

    return image.astype(np.float32) / 255.0


def test_horizontal_banding_correction_reduces_error() -> None:
    clean = smooth_texture()
    degraded = add_banding(clean, "horizontal")
    context = ProcessingContext()
    corrected = BandingOperator().apply(
        _working(degraded),
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
        _working(degraded),
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
        _working(degraded_u8),
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
    heat = moire_heatmap(_working(degraded))
    corrected = DemoireOperator().apply(
        _working(degraded),
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


def test_joint_demoire_reduces_luminance_grid_and_preserves_hard_edge() -> None:
    height, width = 120, 180
    clean = np.full((height, width, 3), 0.25, np.float32)
    clean[:, width // 2 :] = 0.72
    xx = np.arange(width, dtype=np.float32)[None, :, None]
    grid = 0.045 * np.sin(2.0 * np.pi * xx / 3.4)
    degraded = np.clip(clean + grid, 0.0, 1.0).astype(np.float32)
    context = ProcessingContext()
    corrected = DemoireOperator().apply(
        degraded,
        DemoireParameters(
            mode=DemoireMode.JOINT_EDGE_AWARE,
            strength=1.0,
            chroma_radius=2.5,
            edge_protection=0.65,
        ),
        context,
    )
    flat_mask = np.ones((height, width), dtype=bool)
    flat_mask[:, width // 2 - 8 : width // 2 + 8] = False
    before_error = np.mean(np.abs(degraded - clean), axis=2)[flat_mask].mean()
    after_error = np.mean(np.abs(corrected - clean), axis=2)[flat_mask].mean()
    assert float(after_error) < float(before_error) * 0.8
    before_step = float(degraded[:, width // 2 + 4].mean() - degraded[:, width // 2 - 4].mean())
    after_step = float(corrected[:, width // 2 + 4].mean() - corrected[:, width // 2 - 4].mean())
    assert after_step > before_step * 0.9
    assert context.metadata["demoire"]["mode"] == "joint_edge_aware"


def test_frequency_spectrum_has_rgb_display_contract() -> None:
    spectrum = frequency_spectrum(_working(add_color_moire(smooth_texture(180, 120))))
    assert spectrum.shape == (120, 180, 3)
    assert spectrum.dtype == np.uint8
