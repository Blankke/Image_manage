"""由已验证模型清单创建惰性推理后端。"""

from __future__ import annotations

from .backend import InferenceBackend, InferenceError
from .external_process import ExternalProcessBackend
from .model_manifest import ModelManifest
from .onnx_backend import OnnxBackend
from .openvino_backend import OpenVinoBackend


def create_inference_backend(manifest: ModelManifest) -> InferenceBackend:
    """创建清单声明的后端，不提前导入可选运行时。"""

    if manifest.type == "external_process":
        return ExternalProcessBackend(manifest)
    if manifest.type == "onnx":
        return OnnxBackend(manifest)
    if manifest.type == "openvino":
        return OpenVinoBackend(manifest)
    raise InferenceError(f"不支持的模型类型：{manifest.type}")
