"""图像加载与代理预览测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from screenrestore.io.image_loader import load_image


def test_load_image_supports_chinese_path_and_proxy(tmp_path: Path) -> None:
    path = tmp_path / "屏幕照片.png"
    source = np.zeros((100, 240, 3), dtype=np.uint8)
    source[..., 0] = 123
    Image.fromarray(source, "RGB").save(path)

    document = load_image(path)
    proxy = document.proxy(120)

    assert document.path == path.resolve()
    assert (document.width, document.height) == (240, 100)
    assert proxy.shape == (50, 120, 3)
    assert not document.original_rgb.flags.writeable
    assert document.proxy(120) is proxy

