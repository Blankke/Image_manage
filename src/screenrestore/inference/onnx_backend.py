"""惰性导入 onnxruntime 的可选本地后端。"""

from __future__ import annotations

import numpy as np

from screenrestore.core.operator import ProcessingContext

from .backend import InferenceBackend, InferenceError
from .model_manifest import ModelManifest


class OnnxBackend(InferenceBackend):
    """适用于单输入、NCHW RGB 图像模型的 ONNX Runtime 骨架。"""

    def __init__(self, manifest: ModelManifest) -> None:
        if manifest.type != "onnx" or manifest.model_path is None:
            raise InferenceError("OnnxBackend 需要 onnx 清单")
        self.manifest = manifest
        self._session = None

    def is_available(self) -> tuple[bool, str]:
        """检查可选依赖和模型文件。"""

        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "未安装 onnxruntime；核心经典流水线不受影响"
        model_path = self.manifest.resolve_path(self.manifest.model_path or "")
        if not model_path.is_file():
            return False, f"未找到 ONNX 模型：{model_path}"
        return True, str(model_path)

    def run(self, image_rgb: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """以 CPUExecutionProvider 运行简单图像到图像模型。"""

        available, reason = self.is_available()
        if not available:
            raise InferenceError(reason)
        context.cancellation.check()
        try:
            import onnxruntime as ort

            if self._session is None:
                self._session = ort.InferenceSession(
                    str(self.manifest.resolve_path(self.manifest.model_path or "")),
                    providers=["CPUExecutionProvider"],
                )
            input_name = self._session.get_inputs()[0].name
            tensor = image_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            output = self._session.run(None, {input_name: tensor})[0]
        except Exception as exc:  # noqa: BLE001 - 可选运行时错误统一转换
            raise InferenceError(f"ONNX 推理失败：{exc}") from exc
        context.cancellation.check()
        return _tensor_to_rgb(output)


def _tensor_to_rgb(output: np.ndarray) -> np.ndarray:
    if output.ndim == 4:
        output = output[0]
    if output.ndim != 3:
        raise InferenceError("ONNX 输出必须是三维或四维图像张量")
    if output.shape[0] in (1, 3, 4):
        output = output[:3].transpose(1, 2, 0)
    if output.shape[2] == 1:
        output = np.repeat(output, 3, axis=2)
    if output.shape[2] != 3:
        raise InferenceError("ONNX 输出必须包含三个 RGB 通道")
    if np.issubdtype(output.dtype, np.floating):
        output = output * 255.0
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)

