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
from screenrestore.inference.model_manifest import ModelRole, load_manifest

from ._utils import clip_float, require_range, require_rgb_float


@dataclass
class ModelPluginParameters(ParameterModel):
    """可选模型清单、模型内部强度与非破坏混合强度。"""

    manifest_path: str = ""
    blend_strength: float = 1.0
    model_strength: float = 0.25
    denoise_strength: float = 0.25
    output_scale: float = 1.0

    def validate(self) -> None:
        require_range("blend_strength", self.blend_strength, 0.0, 1.0)
        require_range("model_strength", self.model_strength, 0.0, 1.0)
        require_range("denoise_strength", self.denoise_strength, 0.0, 1.0)
        require_range("output_scale", self.output_scale, 1.0, 4.0)
        if len(self.manifest_path) > 4096:
            raise ValueError("模型清单路径过长")


class _RoleModelOperator(ImageOperator[ModelPluginParameters]):
    """按明确角色惰性创建本地后端。"""

    parameter_type = ModelPluginParameters
    role: ModelRole

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
        require_rgb_float(image)
        self.validate(params)
        if not params.manifest_path.strip():
            raise InferenceError("请先在可选模型步骤中填写模型清单路径")
        manifest_path = Path(params.manifest_path).expanduser()
        manifest = load_manifest(manifest_path)
        if manifest.role != self.role:
            raise InferenceError(
                f"模型 {manifest.id} 的角色是 {manifest.role.value}，不能用于 {self.role.value} 节点"
            )
        backend = create_inference_backend(manifest)
        available, reason = backend.is_available()
        if not available:
            raise InferenceError(reason)
        previous_options = context.metadata.get("model_options")
        context.metadata["model_options"] = {
            "model_strength": params.model_strength,
            "denoise_strength": params.denoise_strength,
            "output_scale": params.output_scale,
        }
        try:
            restored = backend.run(image, context)
        finally:
            if previous_options is None:
                context.metadata.pop("model_options", None)
            else:
                context.metadata["model_options"] = previous_options
        context.metadata[self.id] = {
            "manifest_id": manifest.id,
            "role": manifest.role.value,
            "task": manifest.task,
            "model_strength": params.model_strength,
            "denoise_strength": params.denoise_strength,
            "output_scale": params.output_scale,
            "claim": (
                "observed-restoration-prior"
                if manifest.role == ModelRole.RESTORATION
                else "perceptual-generated-detail"
            ),
        }
        if params.blend_strength >= 1.0:
            return restored
        baseline = cv2.resize(
            image,
            (restored.shape[1], restored.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return clip_float(cv2.addWeighted(
            baseline,
            1.0 - params.blend_strength,
            restored,
            params.blend_strength,
            0,
        ))


class RestorationModelOperator(_RoleModelOperator):
    """忠实恢复模型节点，位于传统屏幕伪影处理之后。"""

    id = "restoration_model"
    display_name = "AI 恢复模型"
    role = ModelRole.RESTORATION


class EnhancementModelOperator(_RoleModelOperator):
    """感知增强/超分节点，允许生成统计纹理。"""

    id = "enhancement_model"
    display_name = "AI 感知增强"
    role = ModelRole.ENHANCEMENT
