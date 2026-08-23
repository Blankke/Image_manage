"""实拍样本与授权参考图的离线质量验证工具。"""

from .geometry_benchmark import (
    GeometryGate,
    GeometryGroundTruth,
    aggregate_geometry_results,
    corner_metrics,
    evaluate_geometry_decision,
    polygon_iou,
)
from .reference_comparison import (
    ReferenceRegistration,
    align_for_comparison,
    compare_images,
    difference_heatmap,
    extract_reference_region,
    register_reference,
)

__all__ = [
    "GeometryGate",
    "GeometryGroundTruth",
    "ReferenceRegistration",
    "align_for_comparison",
    "aggregate_geometry_results",
    "compare_images",
    "corner_metrics",
    "difference_heatmap",
    "extract_reference_region",
    "evaluate_geometry_decision",
    "polygon_iou",
    "register_reference",
]
