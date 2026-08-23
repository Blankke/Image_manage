"""Archive / Enhanced 输出的有限来源报告。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .mask import ProvenanceMap


class ArchiveVariant(StrEnum):
    """来源语义明确的产品输出层级。"""

    ARCHIVE = "archive"
    ENHANCED = "enhanced"


@dataclass(frozen=True, slots=True)
class ProvenanceReport:
    """不包含图片内容的归档来源摘要。"""

    variant: ArchiveVariant
    provenance: ProvenanceMap
    geometry: dict[str, Any] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.value,
            "pixel_origin_fraction": self.provenance.summary(),
            "geometry": self.geometry,
            "notes": list(self.notes),
        }
