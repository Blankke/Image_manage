"""归档输出的像素来源与生成区域标记。"""

from .mask import PixelOrigin, ProvenanceMap
from .report import ArchiveVariant, ProvenanceReport

__all__ = ["ArchiveVariant", "PixelOrigin", "ProvenanceMap", "ProvenanceReport"]
