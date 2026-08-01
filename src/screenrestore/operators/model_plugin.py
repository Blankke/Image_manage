"""把可选本地模型清单接入与 GUI/CLI 共用的非破坏流水线。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel
from screenrestore.inference.backend import InferenceError
from screenrestore.inference.factory import create_inference_backend
from screenrestore.inference.model_manifest import load_manifest

from ._utils import require_range, require_rgb_u8


@dataclass
class ModelPluginParameters(ParameterModel):
    """可选模型清单路径与非破坏混合强度。"""

    manifest_path: str = ""
    strength: float = 1.0

    def validate(self) -> None:
        require_range("strength", self.strength, 0.0, 1.0)
        if len(self.manifest_path) > 4096:
            raise ValueError("模型清单路径过长")


class ModelPluginOperator(ImageOperator[ModelPluginParameters]):
    """按清单惰性创建本地后端；默认禁用且不导入任何可选运行时。"""

    id = "model_plugin"
    display_name = "可选模型恢复/超分"
    parameter_type = ModelPluginParameters

    def default_parameters(self) -> ModelPluginParameters:
        return ModelPluginParameters()

    def estimate_cost(self, shape: tuple[int, ...]) -> float:
        return super().estimate_cost(shape) * 12.0

    def apply(
        self,
        image: np.ndarray,
        params: ModelPluginParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_u8(image)
        self.validate(params)
        if not params.manifest_path.strip():
            raise InferenceError("请先在可选模型步骤中填写模型清单路径")
        manifest_path = Path(params.manifest_path).expanduser()
        manifest = load_manifest(manifest_path)
        backend = create_inference_backend(manifest)
        available, reason = backend.is_available()
        if not available:
            raise InferenceError(reason)
        restored = backend.run(image, context)
        if params.strength >= 1.0:
            return restored
        baseline = cv2.resize(
            image,
            (restored.shape[1], restored.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return cv2.addWeighted(
            baseline,
            1.0 - params.strength,
            restored,
            params.strength,
            0,
        )
