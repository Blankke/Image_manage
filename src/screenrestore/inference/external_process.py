"""不使用 shell 的外部图像模型进程后端。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from screenrestore.core.cancellation import ProcessingCancelled
from screenrestore.core.operator import ProcessingContext

from .backend import InferenceBackend, InferenceError
from .model_manifest import ModelManifest


class ExternalProcessBackend(InferenceBackend):
    """通过 ASCII 临时路径桥接可选 ncnn-vulkan 等本地程序。"""

    def __init__(self, manifest: ModelManifest) -> None:
        if manifest.type != "external_process" or manifest.executable is None:
            raise InferenceError("ExternalProcessBackend 需要 external_process 清单")
        self.manifest = manifest

    def is_available(self) -> tuple[bool, str]:
        """检查清单声明的可执行程序。"""

        executable = self._resolve_executable()
        if not executable.is_file():
            return False, f"未找到外部程序：{executable}"
        for required_file in self.manifest.required_files:
            resolved = self.manifest.resolve_path(required_file)
            if not resolved.is_file():
                return False, f"未找到模型依赖：{resolved}"
        return True, str(executable)

    def run(self, image_rgb: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """写入临时 PNG、运行外部程序、读取结果并自动清理。"""

        if image_rgb.dtype != np.float32 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise InferenceError("外部后端需要 H×W×3 RGB float32 图像")
        available, reason = self.is_available()
        if not available:
            raise InferenceError(reason)
        executable = self._resolve_executable()
        context.cancellation.check()
        with tempfile.TemporaryDirectory(prefix="ScreenRestore-model-") as temp_directory:
            temp_root = Path(temp_directory)
            input_path = temp_root / "input.png"
            output_path = temp_root / "output.png"
            input_u8 = np.clip(np.rint(image_rgb * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(input_u8, "RGB").save(input_path, format="PNG")
            placeholders = {
                "input": str(input_path),
                "output": str(output_path),
                "temp": str(temp_root),
                "manifest_dir": str(
                    self.manifest.manifest_path.parent
                    if self.manifest.manifest_path is not None
                    else Path.cwd()
                ),
                "model_strength": str(
                    context.metadata.get("model_options", {}).get("model_strength", 1.0)
                    if isinstance(context.metadata.get("model_options"), dict)
                    else 1.0
                ),
                "denoise_strength": str(
                    context.metadata.get("model_options", {}).get("denoise_strength", 1.0)
                    if isinstance(context.metadata.get("model_options"), dict)
                    else 1.0
                ),
                "output_scale": str(
                    context.metadata.get("model_options", {}).get("output_scale", 1.0)
                    if isinstance(context.metadata.get("model_options"), dict)
                    else 1.0
                ),
            }
            arguments = [item.format_map(placeholders) for item in self.manifest.arguments]
            command = [str(executable), *arguments]
            context.report(0.05, f"启动 {self.manifest.name}")
            try:
                process = subprocess.Popen(  # noqa: S603 - 仅运行用户显式安装并由清单指定的程序
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    shell=False,
                )
            except OSError as exc:
                raise InferenceError(f"无法启动外部程序：{executable}") from exc
            started = time.monotonic()
            while process.poll() is None:
                if context.cancellation.is_cancelled:
                    _terminate_process(process)
                    raise ProcessingCancelled("外部模型处理已取消")
                if time.monotonic() - started > self.manifest.timeout_seconds:
                    _terminate_process(process)
                    raise InferenceError(
                        f"外部模型运行超时（{self.manifest.timeout_seconds:.0f} 秒）"
                    )
                time.sleep(0.05)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                details = (stderr or stdout or "无输出").strip()[-4000:]
                raise InferenceError(f"外部模型退出码 {process.returncode}：{details}")
            if not output_path.is_file():
                raise InferenceError("外部模型未生成约定的输出文件")
            try:
                with Image.open(output_path) as opened:
                    output_u8 = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
            except (OSError, UnidentifiedImageError) as exc:
                raise InferenceError("外部模型输出不是有效图像") from exc
            context.metadata["external_stdout"] = stdout[-4000:]
            context.metadata["external_stderr"] = stderr[-4000:]
            context.report(1.0, f"{self.manifest.name} 完成")
            return np.ascontiguousarray(output_u8.astype(np.float32) / 255.0)

    def _resolve_executable(self) -> Path:
        """解析清单程序；``{python}`` 明确表示当前已激活虚拟环境解释器。"""

        if self.manifest.executable == "{python}":
            # 保留 venv 的入口路径；resolve() 会越过符号链接并丢失虚拟环境 site-packages。
            return Path(sys.executable).absolute()
        return self.manifest.resolve_path(self.manifest.executable or "")


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """先温和终止，短暂等待后再强制结束。"""

    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)
