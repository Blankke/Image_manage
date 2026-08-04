"""可选推理后端统一协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from screenrestore.core.operator import ProcessingContext


class InferenceError(RuntimeError):
    """转换为用户可读文本的模型后端错误。"""


class InferenceBackend(ABC):
    """所有可选本地模型后端的最小接口 (image → image)。"""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """返回后端是否可运行及说明。"""

    @abstractmethod
    def run(self, image_rgb: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """在本地运行模型并返回 RGB float32 [0,1] 图像。"""


@dataclass
class Detection:
    """单个检测结果。"""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x, y, w, h) 归一化 [0,1]
    mask: np.ndarray | None = None  # 二值掩码，与输入图像同尺寸


@dataclass
class AnalysisResult:
    """语义分析后端的统一输出 (image → metadata)。

    与 InferenceBackend.run() 互补：语义模型输出结构化元数据而非图像。
    """

    labels: dict[str, float] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    properties: dict[str, float] = field(default_factory=dict)

    def top_label(self) -> tuple[str, float]:
        """返回最高置信度标签。"""
        if not self.labels:
            return ("other", 0.0)
        return max(self.labels.items(), key=lambda x: x[1])


class AnalysisBackend(ABC):
    """语义分析模型后端 (image → metadata)。

    与 InferenceBackend 并行：用于场景分类、目标检测、语义分割等
    输出结构化元数据而非图像的模型。
    """

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """返回后端是否可运行及说明。"""

    @abstractmethod
    def run_analysis(
        self,
        image_rgb: np.ndarray,
        context: ProcessingContext,
    ) -> AnalysisResult:
        """运行语义分析并返回元数据。"""
