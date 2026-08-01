"""支持 EXIF Orientation 和中文路径的图像加载。"""

from __future__ import annotations

import hashlib
from io import BytesIO
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


def decode_image_bytes(
    data: bytes,
    filename: str = "upload",
    max_pixels: int = 80_000_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把 Web/内存上传安全解码为独立 RGB 数组和有限元数据。

    文件名仅用于可读错误，不参与路径访问。像素上限会在完整解码前检查，避免压缩
    炸弹占用不可控内存。
    """

    if not data:
        raise ImageLoadError(f"上传图片为空：{Path(filename).name}")
    if max_pixels <= 0:
        raise ValueError("max_pixels 必须大于 0")
    safe_name = Path(filename).name[:240] or "upload"
    try:
        with Image.open(BytesIO(data)) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageLoadError(
                    f"图片像素数超过限制：{width}×{height}，最多 {max_pixels:,} 像素"
                )
            exif = opened.getexif()
            metadata: dict[str, Any] = {
                "filename": safe_name,
                "format": opened.format,
                "mode": opened.mode,
                "original_size": [width, height],
                "exif": {str(key): _safe_metadata_value(value) for key, value in exif.items()},
            }
            oriented = ImageOps.exif_transpose(opened)
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8).copy()
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"无法读取上传图片：{safe_name}") from exc
    return rgb, metadata


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
