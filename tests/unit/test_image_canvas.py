"""Qt 图像颜色转换烟雾测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.ui.image_canvas import rgb_to_qimage


def test_rgb_to_qimage_keeps_channel_order() -> None:
    rgb = np.array([[[250, 20, 10]]], dtype=np.uint8)
    image = rgb_to_qimage(rgb)
    color = image.pixelColor(0, 0)
    assert (color.red(), color.green(), color.blue()) == (250, 20, 10)

