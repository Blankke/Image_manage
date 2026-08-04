"""本地可选模型清单读取与验证。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .backend import InferenceError


class ModelRole(StrEnum):
    """模型在流水线中的语义位置。

    ANALYSIS:       场景分类、目标检测、语义分割 → 输出 SceneContext
    RESTORATION:    去噪、去模糊、去摩尔纹 → 输出同尺寸 RGB 图像
    RECONSTRUCTION: 反光去除、inpainting、特定退化逆变换 → 输出同尺寸 RGB 图像
    ENHANCEMENT:    超分、感知增强、生成纹理 → 输出可不同尺寸 RGB 图像
    """

    ANALYSIS = "analysis"
    RESTORATION = "restoration"
    RECONSTRUCTION = "reconstruction"
    ENHANCEMENT = "enhancement"


ALLOWED_MODEL_TASKS = {
    # ANALYSIS
    "scene_classification",
    "object_detection",
    "segmentation",
    "artifact_segmentation",
    # RESTORATION
    "deblur",
    "demoire",
    "deband",
    "denoise",
    "screen_restoration",
    # RECONSTRUCTION
    "reflection_removal",
    "inpainting",
    "generative_reconstruction",
    # ENHANCEMENT
    "perceptual_restoration",
    "super_resolution",
}


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """描述外部进程或可选 Python 运行时模型。"""

    id: str
    name: str
    type: str
    role: ModelRole
    task: str
    executable: str | None = None
    arguments: list[str] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    model_path: str | None = None
    supports_tiling: bool = False
    tile_size: int = 512
    tile_overlap: int = 32
    tile_padding: int = 16
    license: str = "UNKNOWN"
    homepage: str = ""
    timeout_seconds: float = 600.0
    manifest_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], manifest_path: Path | None = None) -> ModelManifest:
        """严格解析模型清单。"""

        required = ("id", "name", "type", "role", "task")
        if any(not isinstance(data.get(key), str) or not data[key] for key in required):
            raise InferenceError("模型清单必须包含非空 id、name、type、role 和 task")
        backend_type = str(data["type"])
        if backend_type not in {"external_process", "onnx", "openvino"}:
            raise InferenceError(f"不支持的模型类型：{backend_type}")
        try:
            role = ModelRole(str(data["role"]))
        except ValueError as exc:
            raise InferenceError(
                "模型 role 必须是 analysis / restoration / reconstruction / enhancement"
            ) from exc
        task = str(data["task"])
        if task not in ALLOWED_MODEL_TASKS:
            raise InferenceError(f"不支持的模型任务：{task}")
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
        tile_size = int(data.get("tile_size", 512))
        tile_overlap = int(data.get("tile_overlap", 32))
        tile_padding = int(data.get("tile_padding", 16))
        if tile_size < 16 or not 0 <= tile_overlap < tile_size:
            raise InferenceError("模型清单 tile_size/tile_overlap 无效")
        if not 0 <= tile_padding < tile_size // 2:
            raise InferenceError("模型清单 tile_padding 无效")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            type=backend_type,
            role=role,
            task=task,
            executable=executable,
            arguments=list(arguments),
            required_files=list(required_files),
            model_path=model_path,
            supports_tiling=bool(data.get("supports_tiling", False)),
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_padding=tile_padding,
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
