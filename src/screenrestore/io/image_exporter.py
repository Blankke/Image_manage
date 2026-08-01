"""PNG/JPEG/WebP/TIFF 全分辨率结果导出。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


class ExportFormat(StrEnum):
    """支持的导出格式。"""

    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    TIFF = "TIFF"


@dataclass(slots=True)
class ExportOptions:
    """图像编码和元数据策略。"""

    format: ExportFormat
    quality: int = 92
    keep_exif: bool = True
    remove_gps: bool = True
    overwrite: bool = False

    def validate(self) -> None:
        """验证编码质量。"""

        if not 1 <= self.quality <= 100:
            raise ValueError("导出质量必须位于 1..100")


class ImageExportError(RuntimeError):
    """用户可读的导出错误。"""


def encode_image_bytes(
    image_rgb: np.ndarray,
    output_format: ExportFormat = ExportFormat.PNG,
    quality: int = 92,
) -> bytes:
    """为 Web 响应编码图像，不写临时文件或携带源图隐私元数据。"""

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ImageExportError("编码需要 H×W×3 RGB uint8 图像")
    if not 1 <= quality <= 100:
        raise ImageExportError("编码质量必须位于 1..100")
    options: dict[str, object] = {}
    if output_format in (ExportFormat.JPEG, ExportFormat.WEBP):
        options["quality"] = quality
    if output_format == ExportFormat.JPEG:
        options.update(optimize=True, subsampling=0)
    stream = BytesIO()
    try:
        Image.fromarray(image_rgb, "RGB").save(
            stream,
            format=output_format.value,
            **options,
        )
    except OSError as exc:
        raise ImageExportError("无法编码 Web 输出图像") from exc
    return stream.getvalue()


def infer_export_format(path: str | Path) -> ExportFormat:
    """从输出扩展名推断编码格式。"""

    suffix = Path(path).suffix.lower()
    mapping = {
        ".png": ExportFormat.PNG,
        ".jpg": ExportFormat.JPEG,
        ".jpeg": ExportFormat.JPEG,
        ".webp": ExportFormat.WEBP,
        ".tif": ExportFormat.TIFF,
        ".tiff": ExportFormat.TIFF,
    }
    try:
        return mapping[suffix]
    except KeyError as exc:
        raise ImageExportError(f"不支持的输出格式：{suffix}") from exc


def export_image(
    image_rgb: np.ndarray,
    path: str | Path,
    options: ExportOptions,
    source_path: str | Path | None = None,
) -> Path:
    """原子编码 RGB uint8 图像，并按策略保留 EXIF/删除 GPS。"""

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ImageExportError("导出需要 H×W×3 RGB uint8 图像")
    options.validate()
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not options.overwrite:
        raise ImageExportError(f"文件已存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {}
    if options.format in (ExportFormat.JPEG, ExportFormat.WEBP):
        save_options["quality"] = options.quality
    if options.format == ExportFormat.JPEG:
        save_options.update(optimize=True, subsampling=0)
    exif = _read_export_exif(source_path, options)
    if exif is not None:
        save_options["exif"] = exif
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
        Image.fromarray(image_rgb, "RGB").save(
            temp_path,
            format=options.format.value,
            **save_options,
        )
        os.replace(temp_path, destination)
    except OSError as exc:
        raise ImageExportError(f"无法导出图像：{destination}") from exc
    return destination


def _read_export_exif(
    source_path: str | Path | None,
    options: ExportOptions,
) -> bytes | None:
    if not options.keep_exif or source_path is None:
        return None
    try:
        with Image.open(source_path) as source:
            exif = source.getexif()
            if not exif:
                return None
            if options.remove_gps and 34853 in exif:
                del exif[34853]
            return exif.tobytes()
    except OSError:
        return None
