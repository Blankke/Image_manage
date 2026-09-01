"""平面内容定位、原图精修和单应校正领域层。"""

from .calibration import CorrectnessCalibrator
from .confidence import CORRECTNESS_FEATURE_NAMES, ConfidencePolicy
from .detector import ClassicQuadDetector, OnnxQuadDetector, QuadDetector
from .edge_refine import EdgeRefineParameters, refine_quad_edges
from .localizer import AutomaticGeometryService
from .mappings import (
    InverseMap,
    RadialLensParameters,
    compose_inverse_maps,
    dense_grid_inverse_map,
    homography_inverse_map,
    identity_inverse_map,
    minimum_jacobian_determinant,
    orientation_inverse_map,
    radial_inverse_map,
    radial_straight_line_residual,
    remap_original_once,
    safe_radial_inverse_map,
)
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
    "CORRECTNESS_FEATURE_NAMES",
    "ConfidencePolicy",
    "CorrectnessCalibrator",
    "EdgeRefineParameters",
    "EdgeRefinement",
    "InterpolationMode",
    "InverseMap",
    "RadialLensParameters",
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
    "compose_inverse_maps",
    "dense_grid_inverse_map",
    "homography_inverse_map",
    "identity_inverse_map",
    "minimum_jacobian_determinant",
    "orientation_inverse_map",
    "radial_inverse_map",
    "radial_straight_line_residual",
    "remap_original_once",
    "safe_radial_inverse_map",
    "estimate_output_size",
    "estimate_rectified_aspect_ratio",
    "order_corners",
    "refine_quad_edges",
    "target_class_for_scene",
    "warp_perspective",
]
