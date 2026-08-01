"""合成测试卡与退化生成器契约测试。"""

from __future__ import annotations

import numpy as np

from tests.synthetic.degradations import (
    defocus_blur,
    gaussian_noise,
    jpeg_degradation,
    local_glow,
    motion_blur,
    perspective_degradation,
    poisson_noise,
    rotate_degradation,
    tone_degradation,
)
from tests.synthetic.degradations import (
    test_chart as make_test_chart,
)


def test_all_degradations_preserve_rgb_uint8_contract() -> None:
    chart = make_test_chart(320, 180)
    degraded_images = [
        rotate_degradation(chart, 4),
        tone_degradation(chart),
        gaussian_noise(chart),
        poisson_noise(chart),
        jpeg_degradation(chart),
        motion_blur(chart),
        defocus_blur(chart),
        local_glow(chart),
    ]
    perspective, corners = perspective_degradation(chart)
    assert corners.shape == (4, 2)
    assert perspective.ndim == 3
    for degraded in degraded_images:
        assert degraded.shape == chart.shape
        assert degraded.dtype == np.uint8
