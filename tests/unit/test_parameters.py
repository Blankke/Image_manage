"""参数模型序列化测试。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from screenrestore.core.parameters import ParameterModel


class DemoMode(StrEnum):
    FAST = "fast"
    QUALITY = "quality"


@dataclass
class DemoParameters(ParameterModel):
    amount: float = 1.0
    mode: DemoMode = DemoMode.FAST


def test_parameter_roundtrip_restores_enum() -> None:
    params = DemoParameters(amount=0.25, mode=DemoMode.QUALITY)
    restored = DemoParameters.from_dict(params.to_dict())
    assert restored == params


def test_parameter_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="未知参数"):
        DemoParameters.from_dict({"missing": 1})
