"""惰性导入 OpenVINO 的可选本地后端。"""

from __future__ import annotations

import numpy as np

from screenrestore.core.operator import ProcessingContext

from .backend import InferenceBackend, InferenceError
from .model_manifest import ModelManifest
from .onnx_backend import _tensor_to_rgb


class OpenVinoBackend(InferenceBackend):
    """适用于单输入 NCHW 图像模型的 OpenVINO CPU 骨架。"""

    def __init__(self, manifest: ModelManifest) -> None:
        if manifest.type != "openvino" or manifest.model_path is None:
            raise InferenceError("OpenVinoBackend 需要 openvino 清单")
        self.manifest = manifest
        self._compiled = None

    def is_available(self) -> tuple[bool, str]:
        """检查可选依赖和模型文件。"""

        try:
            import openvino  # noqa: F401
        except ImportError:
            return False, "未安装 OpenVINO；核心经典流水线不受影响"
        model_path = self.manifest.resolve_path(self.manifest.model_path or "")
        if not model_path.is_file():
            return False, f"未找到 OpenVINO/ONNX 模型：{model_path}"
        return True, str(model_path)

    def run(self, image_rgb: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """在 OpenVINO CPU 设备上运行模型。"""

        available, reason = self.is_available()
        if not available:
            raise InferenceError(reason)
        context.cancellation.check()
        try:
            from openvino import Core

            if self._compiled is None:
                core = Core()
                model = core.read_model(
                    str(self.manifest.resolve_path(self.manifest.model_path or ""))
                )
                self._compiled = core.compile_model(model, "CPU")
            tensor = image_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            result = self._compiled([tensor])
            output = next(iter(result.values()))
        except Exception as exc:  # noqa: BLE001 - 可选运行时错误统一转换
            raise InferenceError(f"OpenVINO 推理失败：{exc}") from exc
        context.cancellation.check()
        return _tensor_to_rgb(output)

