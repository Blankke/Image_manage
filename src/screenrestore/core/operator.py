"""统一图像算子协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

from .cancellation import CancellationToken
from .parameters import ParameterModel

P = TypeVar("P", bound=ParameterModel)
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class ProcessingContext:
    """一次处理调用共享的取消、进度和质量上下文。"""

    cancellation: CancellationToken = field(default_factory=CancellationToken)
    progress: ProgressCallback | None = None
    preview: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def report(self, fraction: float, message: str) -> None:
        """报告规范化进度并立刻检查取消。"""

        self.cancellation.check()
        if self.progress is not None:
            self.progress(float(np.clip(fraction, 0.0, 1.0)), message)


class ImageOperator(ABC, Generic[P]):
    """所有 RGB 图像恢复步骤必须实现的接口。"""

    id: str
    display_name: str
    parameter_type: type[P]
    version: int = 1
    reorderable: bool = True

    @abstractmethod
    def default_parameters(self) -> P:
        """返回独立的默认参数对象。"""

    @abstractmethod
    def apply(self, image: np.ndarray, params: P, context: ProcessingContext) -> np.ndarray:
        """处理 RGB uint8 图像，且不得原地修改输入。"""

    def validate(self, params: P) -> None:
        """在处理前验证参数。"""

        params.validate()

    def estimate_cost(self, shape: tuple[int, ...]) -> float:
        """返回用于进度分配的相对成本。"""

        return max(1.0, float(np.prod(shape[:2])) / 1_000_000.0)
