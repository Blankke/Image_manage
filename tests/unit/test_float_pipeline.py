"""float32 流水线精度与标准 sRGB 传递函数测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.core.color import linear_to_srgb, srgb_to_linear
from screenrestore.core.operator import ProcessingContext
from screenrestore.core.pipeline import ImagePipeline, OperatorState
from screenrestore.operators.denoise import DenoiseOperator, DenoiseParameters
from screenrestore.operators.exposure import ExposureOperator, ExposureParameters
from screenrestore.operators.local_contrast import ClaheOperator, ClaheParameters
from screenrestore.operators.sharpen import SharpenOperator, SharpenParameters
from screenrestore.operators.white_balance import (
    WhiteBalanceMode,
    WhiteBalanceOperator,
    WhiteBalanceParameters,
)


def test_srgb_linear_roundtrip_is_precise() -> None:
    values = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    image = np.repeat(values[None, :, None], 3, axis=2)
    restored = linear_to_srgb(srgb_to_linear(image))
    assert restored.dtype == np.float32
    assert float(np.max(np.abs(restored - image))) < 2e-7


def test_photographic_exposure_multiplies_linear_light() -> None:
    source = np.full((8, 8, 3), 0.25, np.float32)
    output = ExposureOperator().apply(
        source,
        ExposureParameters(exposure=1.0),
        ProcessingContext(),
    )
    expected = linear_to_srgb(np.clip(srgb_to_linear(source) * 2.0, 0.0, 1.0))
    assert np.allclose(output, expected, atol=1e-7)


def test_neutral_real_operators_preserve_sub_uint8_dark_gradient() -> None:
    gradient = np.linspace(0.035, 0.045, 2048, dtype=np.float32)
    source = np.repeat(gradient[None, :, None], 3, axis=2)
    pipeline = ImagePipeline(
        [
            OperatorState(ExposureOperator(), True, ExposureParameters()),
            OperatorState(
                WhiteBalanceOperator(),
                True,
                WhiteBalanceParameters(mode=WhiteBalanceMode.OFF),
            ),
            OperatorState(DenoiseOperator(), True, DenoiseParameters(strength=0.0)),
            OperatorState(ClaheOperator(), True, ClaheParameters(strength=0.0)),
            OperatorState(SharpenOperator(), True, SharpenParameters(amount=0.0)),
        ]
    )
    output = pipeline.process(source, source_id="dark-gradient")
    assert output.dtype == np.float32
    assert np.array_equal(output, source)
    # 该区间只有约 3 个 uint8 台阶；保留全部 2048 个值可证明节点间没有量化。
    assert np.unique(output[0, :, 0]).size == 2048
