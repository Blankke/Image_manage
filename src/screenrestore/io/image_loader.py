"""支持 EXIF Orientation 和中文路径的图像加载。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from screenrestore.core.image_document import ImageDocument

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ImageLoadError(RuntimeError):
    """用户可读的图像加载错误。"""


def load_image(path: str | Path) -> ImageDocument:
    """加载图像为只读 RGB 文档，保留基础 EXIF 和文件元数据。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ImageLoadError(f"图像不存在：{source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"不支持的图像格式：{source.suffix}")
    try:
        with Image.open(source) as opened:
            exif = opened.getexif()
            metadata: dict[str, Any] = {
                "format": opened.format,
                "mode": opened.mode,
                "original_size": list(opened.size),
                "exif": {str(key): _safe_metadata_value(value) for key, value in exif.items()},
            }
            oriented = ImageOps.exif_transpose(opened)
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"无法读取图像：{source.name}") from exc
    return ImageDocument(
        path=source,
        original_rgb=rgb,
        metadata=metadata,
        content_hash=_sha256(source),
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件摘要，避免复制大文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_metadata_value(value: Any) -> Any:
    """把常见 EXIF 值约束为 JSON 可表达形式。"""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return str(value)[:512]

