"""分块边缘、整数放大和加权融合测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.inference.tiled_inference import tiled_inference


def test_tiled_identity_has_no_seams_on_edge_tiles() -> None:
    generator = np.random.default_rng(11)
    image = generator.integers(0, 256, (137, 191, 3), dtype=np.uint8)
    output = tiled_inference(
        image,
        lambda tile: tile.copy(),
        ProcessingContext(),
        tile_size=64,
        overlap=17,
        padding=9,
    )
    assert np.array_equal(output, image)


def test_tiled_inference_supports_two_times_output() -> None:
    image = np.arange(45 * 71 * 3, dtype=np.uint16).reshape(45, 71, 3).astype(np.uint8)
    output = tiled_inference(
        image,
        lambda tile: np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1),
        ProcessingContext(),
        tile_size=32,
        overlap=8,
        padding=4,
    )
    expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    assert np.array_equal(output, expected)

