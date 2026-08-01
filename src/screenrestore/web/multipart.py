"""基于 Python 标准库的有界 multipart/form-data 解析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from .service import UploadedImage


@dataclass(slots=True)
class MultipartData:
    """按字段名分组的文本字段和上传文件。"""

    fields: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, list[UploadedImage]] = field(default_factory=dict)

    def first(self, name: str, default_value: str = "") -> str:
        values = self.fields.get(name)
        return values[0] if values else default_value


def parse_multipart(
    content_type: str,
    body: bytes,
    *,
    max_parts: int = 64,
    max_field_bytes: int = 1_000_000,
) -> MultipartData:
    """解析已由 HTTP 层限制总长度的 multipart 请求。"""

    if "\r" in content_type or "\n" in content_type:
        raise ValueError("Content-Type 包含非法换行")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("接口需要 multipart/form-data")
    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    message = BytesParser(policy=default).parsebytes(envelope)
    if not message.is_multipart():
        raise ValueError("multipart 请求边界无效")
    result = MultipartData()
    parts = list(message.iter_parts())
    if len(parts) > max_parts:
        raise ValueError(f"multipart 字段数量不能超过 {max_parts}")
    for part in parts:
        if part.get_content_disposition() != "form-data" or part.is_multipart():
            raise ValueError("multipart 子项格式无效")
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name or len(name) > 80:
            raise ValueError("multipart 字段名无效")
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            safe_name = Path(filename.replace("\\", "/")).name.replace("\x00", "")[:240]
            result.files.setdefault(name, []).append(
                UploadedImage(safe_name or "upload", payload)
            )
            continue
        if len(payload) > max_field_bytes:
            raise ValueError(f"文本字段 {name} 过大")
        charset = part.get_content_charset() or "utf-8"
        try:
            value = payload.decode(charset)
        except (LookupError, UnicodeDecodeError) as exc:
            raise ValueError(f"文本字段 {name} 不是有效 UTF-8") from exc
        result.fields.setdefault(name, []).append(value)
    return result
