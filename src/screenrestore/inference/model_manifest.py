"""本地可选模型清单读取与验证。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend import InferenceError


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """描述外部进程或可选 Python 运行时模型。"""

    id: str
    name: str
    type: str
    executable: str | None = None
    arguments: list[str] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    model_path: str | None = None
    supports_tiling: bool = False
    license: str = "UNKNOWN"
    homepage: str = ""
    timeout_seconds: float = 600.0
    manifest_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], manifest_path: Path | None = None) -> ModelManifest:
        """严格解析模型清单。"""

        required = ("id", "name", "type")
        if any(not isinstance(data.get(key), str) or not data[key] for key in required):
            raise InferenceError("模型清单必须包含非空 id、name 和 type")
        backend_type = str(data["type"])
        if backend_type not in {"external_process", "onnx", "openvino"}:
            raise InferenceError(f"不支持的模型类型：{backend_type}")
        executable = data.get("executable")
        model_path = data.get("model_path")
        if backend_type == "external_process" and not isinstance(executable, str):
            raise InferenceError("外部进程模型必须声明 executable")
        if backend_type in {"onnx", "openvino"} and not isinstance(model_path, str):
            raise InferenceError(f"{backend_type} 模型必须声明 model_path")
        arguments = data.get("arguments", [])
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise InferenceError("arguments 必须是字符串数组")
        required_files = data.get("required_files", [])
        if not isinstance(required_files, list) or not all(
            isinstance(item, str) and item for item in required_files
        ):
            raise InferenceError("required_files 必须是非空字符串数组")
        timeout = float(data.get("timeout_seconds", 600.0))
        if not 1.0 <= timeout <= 86_400:
            raise InferenceError("timeout_seconds 必须位于 1..86400")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            type=backend_type,
            executable=executable,
            arguments=list(arguments),
            required_files=list(required_files),
            model_path=model_path,
            supports_tiling=bool(data.get("supports_tiling", False)),
            license=str(data.get("license", "UNKNOWN")),
            homepage=str(data.get("homepage", "")),
            timeout_seconds=timeout,
            manifest_path=manifest_path,
        )

    def resolve_path(self, configured_path: str) -> Path:
        """相对清单目录解析可执行程序或模型路径。"""

        path = Path(configured_path).expanduser()
        if not path.is_absolute() and self.manifest_path is not None:
            path = self.manifest_path.parent / path
        # 不解引用可执行程序符号链接；Python venv 依赖该入口保留环境语义。
        return path.absolute()


def load_manifest(path: str | Path) -> ModelManifest:
    """从 UTF-8 JSON 加载单个模型清单。"""

    manifest_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(f"无法读取模型清单：{manifest_path}") from exc
    if not isinstance(data, dict):
        raise InferenceError("模型清单根节点必须是对象")
    return ModelManifest.from_dict(data, manifest_path)


def discover_manifests(directory: str | Path) -> tuple[list[ModelManifest], list[str]]:
    """发现目录中的 JSON 清单；坏清单作为错误返回而不阻止应用启动。"""

    root = Path(directory).expanduser()
    if not root.is_dir():
        return [], []
    manifests: list[ModelManifest] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*.json")):
        try:
            manifests.append(load_manifest(path))
        except InferenceError as exc:
            errors.append(str(exc))
    return manifests, errors
