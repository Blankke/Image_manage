"""可选推理后端统一协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from screenrestore.core.operator import ProcessingContext


class InferenceError(RuntimeError):
    """转换为用户可读文本的模型后端错误。"""


class InferenceBackend(ABC):
    """所有可选本地模型后端的最小接口。"""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """返回后端是否可运行及说明。"""

    @abstractmethod
    def run(self, image_rgb: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """在本地运行模型并返回 RGB float32 [0,1] 图像。"""
