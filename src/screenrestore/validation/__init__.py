"""实拍样本与授权参考图的离线质量验证工具。"""

from .reference_comparison import (
    ReferenceRegistration,
    align_for_comparison,
    compare_images,
    difference_heatmap,
    extract_reference_region,
    register_reference,
)

__all__ = [
    "ReferenceRegistration",
    "align_for_comparison",
    "compare_images",
    "difference_heatmap",
    "extract_reference_region",
    "register_reference",
]
