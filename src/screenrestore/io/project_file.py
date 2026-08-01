"""`.screenrestore.json` 非破坏项目文件读写。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screenrestore import __version__
from screenrestore.core.image_document import ImageDocument
from screenrestore.core.pipeline import ImagePipeline, OperatorRegistry
from screenrestore.core.presets import PresetId

PROJECT_FORMAT_VERSION = 2
PROJECT_SUFFIX = ".screenrestore.json"


class ProjectFileError(RuntimeError):
    """项目文件格式或读写错误。"""


@dataclass(slots=True)
class LoadedProject:
    """已解析的项目状态与源图警告。"""

    path: Path
    source_path: Path
    source_hash: str
    pipeline: ImagePipeline
    preset: PresetId
    model_config: dict[str, Any]
    warnings: list[str]


def save_project(
    path: str | Path,
    document: ImageDocument,
    pipeline: ImagePipeline,
    preset: PresetId,
    model_config: dict[str, Any] | None = None,
) -> Path:
    """原子写入项目；源图路径相对项目目录保存。"""

    destination = _normalized_project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_path = os.path.relpath(document.path, destination.parent)
    except ValueError:
        # Windows 不同盘符无法表达相对路径，此时保留绝对路径。
        source_path = str(document.path)
    geometry = pipeline.state("geometry").params.to_dict()
    payload = {
        "format_version": PROJECT_FORMAT_VERSION,
        "application_version": __version__,
        "source": {
            "path": source_path,
            "sha256": document.content_hash,
            "width": document.width,
            "height": document.height,
        },
        "geometry": {
            "corners": geometry.get("corners"),
            "output_ratio": geometry.get("ratio_mode"),
        },
        "pipeline": pipeline.to_dict(),
        "preset": preset.value,
        "models": model_config or {},
    }
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_path, destination)
    except OSError as exc:
        raise ProjectFileError(f"无法保存项目：{destination}") from exc
    return destination


def load_project(path: str | Path, registry: OperatorRegistry) -> LoadedProject:
    """读取项目和流水线，并检查源图存在性；哈希由加载图像后核对。"""

    source_file = Path(path).expanduser().resolve()
    try:
        data = json.loads(source_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectFileError(f"无法读取项目：{source_file}") from exc
    if not isinstance(data, dict) or data.get("format_version") != PROJECT_FORMAT_VERSION:
        raise ProjectFileError("不支持的项目格式版本")
    source = data.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise ProjectFileError("项目缺少有效源图路径")
    referenced = Path(source["path"])
    resolved_source = referenced if referenced.is_absolute() else (source_file.parent / referenced)
    resolved_source = resolved_source.resolve()
    warnings: list[str] = []
    if not resolved_source.is_file():
        warnings.append("原图不存在，请重新定位")
    pipeline_data = data.get("pipeline")
    if not isinstance(pipeline_data, dict):
        raise ProjectFileError("项目缺少有效流水线")
    raw_operators = pipeline_data.get("operators")
    if not isinstance(raw_operators, list):
        raise ProjectFileError("项目缺少完整算子列表")
    project_operator_ids = {
        item.get("id") for item in raw_operators if isinstance(item, dict)
    }
    expected_operator_ids = set(registry.ids)
    if project_operator_ids != expected_operator_ids:
        missing = sorted(str(item) for item in expected_operator_ids - project_operator_ids)
        extra = sorted(str(item) for item in project_operator_ids - expected_operator_ids)
        details = []
        if missing:
            details.append(f"缺少 {', '.join(missing)}")
        if extra:
            details.append(f"未知 {', '.join(extra)}")
        raise ProjectFileError("项目流水线版本不完整：" + "；".join(details))
    try:
        pipeline = ImagePipeline.from_dict(pipeline_data, registry)
        preset = PresetId(str(data.get("preset", PresetId.CUSTOM.value)))
    except (ValueError, TypeError) as exc:
        raise ProjectFileError(f"项目参数无效：{exc}") from exc
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ProjectFileError("models 必须是对象")
    return LoadedProject(
        path=source_file,
        source_path=resolved_source,
        source_hash=str(source.get("sha256", "")),
        pipeline=pipeline,
        preset=preset,
        model_config=models,
        warnings=warnings,
    )


def relocate_source(project: LoadedProject, new_source: str | Path) -> None:
    """在内存中为缺失源图重新定位，不自动覆盖项目文件。"""

    candidate = Path(new_source).expanduser().resolve()
    if not candidate.is_file():
        raise ProjectFileError(f"重新定位的图像不存在：{candidate}")
    project.source_path = candidate
    project.warnings = [item for item in project.warnings if "不存在" not in item]


def verify_project_source(project: LoadedProject, document: ImageDocument) -> list[str]:
    """核对已加载源图哈希，变化时警告但允许继续。"""

    warnings = list(project.warnings)
    if project.source_hash and document.content_hash != project.source_hash:
        warnings.append("原图哈希与项目记录不同，结果可能无法复现")
    return warnings


def _normalized_project_path(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if not str(destination).lower().endswith(PROJECT_SUFFIX):
        destination = destination.with_name(destination.name + PROJECT_SUFFIX)
    return destination
