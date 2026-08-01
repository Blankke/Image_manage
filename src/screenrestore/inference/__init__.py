"""不污染核心依赖的可选模型推理后端。"""

from .backend import InferenceBackend, InferenceError
from .external_process import ExternalProcessBackend
from .model_manifest import ModelManifest, discover_manifests, load_manifest

__all__ = [
    "ExternalProcessBackend",
    "InferenceBackend",
    "InferenceError",
    "ModelManifest",
    "discover_manifests",
    "load_manifest",
]

