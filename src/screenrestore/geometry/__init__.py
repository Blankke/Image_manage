"""平面内容定位、原图精修和单应校正领域层。"""

from .confidence import ConfidencePolicy
from .detector import ClassicQuadDetector, OnnxQuadDetector, QuadDetector
from .edge_refine import EdgeRefineParameters, refine_quad_edges
from .localizer import AutomaticGeometryService
from .rectify import (
    AspectRatioMode,
    InterpolationMode,
    estimate_aspect,
    estimate_output_size,
    estimate_rectified_aspect_ratio,
    order_corners,
    warp_perspective,
)
from .types import (
    AspectEstimate,
    EdgeRefinement,
    LocalizationDecision,
    LocalizationStatus,
    QuadPrediction,
    QuadrilateralCandidate,
    RejectionReason,
    TargetClass,
    TargetLayer,
    target_class_for_scene,
)

__all__ = [
    "AspectEstimate",
    "AspectRatioMode",
    "AutomaticGeometryService",
    "ClassicQuadDetector",
    "ConfidencePolicy",
    "EdgeRefineParameters",
    "EdgeRefinement",
    "InterpolationMode",
    "LocalizationDecision",
    "LocalizationStatus",
    "OnnxQuadDetector",
    "QuadDetector",
    "QuadPrediction",
    "QuadrilateralCandidate",
    "RejectionReason",
    "TargetClass",
    "TargetLayer",
    "estimate_aspect",
    "estimate_output_size",
    "estimate_rectified_aspect_ratio",
    "order_corners",
    "refine_quad_edges",
    "target_class_for_scene",
    "warp_perspective",
]
