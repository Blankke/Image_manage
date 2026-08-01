"""基础反光抑制与实验性小区域修复。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import require_range, require_rgb_u8
from .reflection_dct import suppress_reflection_dct


class ReflectionMode(StrEnum):
    """反光处理模式。"""

    HIGHLIGHT_MASK = "highlight_mask"
    GRADIENT_DCT = "gradient_dct_experimental"


class InpaintMethod(StrEnum):
    """OpenCV 局部修复方法。"""

    NONE = "none"
    TELEA = "telea"
    NAVIER_STOKES = "navier_stokes"


@dataclass
class ReflectionParameters(ParameterModel):
    """自动反光检测与手工多边形蒙版参数。"""

    mode: ReflectionMode = ReflectionMode.HIGHLIGHT_MASK
    bright_threshold: float = 0.9
    low_saturation_threshold: float = 0.18
    strength: float = 0.35
    feather_radius: float = 8.0
    inpaint_method: InpaintMethod = InpaintMethod.NONE
    inpaint_radius: float = 3.0
    include_polygons: list[list[list[float]]] = field(default_factory=list)
    exclude_polygons: list[list[list[float]]] = field(default_factory=list)
    show_mask: bool = False
    gradient_threshold: float = 0.02
    smoothness_lambda: float = 0.0
    curvature_weight: float = 1.0

    def validate(self) -> None:
        require_range("bright_threshold", self.bright_threshold, 0.5, 1.0)
        require_range("low_saturation_threshold", self.low_saturation_threshold, 0.0, 0.6)
        require_range("strength", self.strength, 0.0, 1.0)
        require_range("feather_radius", self.feather_radius, 0.0, 50.0)
        require_range("inpaint_radius", self.inpaint_radius, 1.0, 15.0)
        require_range("gradient_threshold", self.gradient_threshold, 0.0, 0.13)
        require_range("smoothness_lambda", self.smoothness_lambda, 0.0, 1.0)
        require_range("curvature_weight", self.curvature_weight, 0.05, 1.0)


class ReflectionOperator(ImageOperator[ReflectionParameters]):
    """压制可检测高光；只对小蒙版区域提供实验性 inpainting。"""

    id = "reflection"
    display_name = "反光抑制/实验性修复"
    parameter_type = ReflectionParameters

    def default_parameters(self) -> ReflectionParameters:
        return ReflectionParameters()

    def apply(
        self,
        image: np.ndarray,
        params: ReflectionParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_u8(image)
        self.validate(params)
        if params.mode == ReflectionMode.GRADIENT_DCT:
            context.report(0.08, "执行梯度 DCT 反光抑制")
            restored = suppress_reflection_dct(
                image.astype(np.float32) / 255.0,
                gradient_threshold=params.gradient_threshold,
                smoothness_lambda=params.smoothness_lambda,
                curvature_weight=params.curvature_weight,
                strength=params.strength,
                cancellation=context.cancellation,
            )
            context.metadata["reflection_mode"] = params.mode.value
            context.metadata["reflection_mask"] = None
            context.report(1.0, "梯度 DCT 反光抑制完成")
            return np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        auto = ((value >= params.bright_threshold) & (saturation <= params.low_saturation_threshold)) | (
            value >= 0.985
        )
        mask = auto.astype(np.uint8) * 255
        _paint_polygons(mask, params.include_polygons, 255)
        _paint_polygons(mask, params.exclude_polygons, 0)
        if params.feather_radius > 0:
            soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), params.feather_radius)
        else:
            soft_mask = mask.astype(np.float32) / 255.0
        # 大面积信息已经饱和时无法真实恢复，只压缩局部高光而不声称重建细节。
        compressed_value = np.minimum(value, params.bright_threshold) + (
            np.maximum(value - params.bright_threshold, 0.0) * 0.25
        )
        blend = (params.strength * soft_mask).astype(np.float32)
        hsv[..., 2] = np.clip((value * (1 - blend) + compressed_value * blend) * 255, 0, 255)
        suppressed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        area_ratio = float(np.mean(mask > 0))
        if params.inpaint_method != InpaintMethod.NONE and 0 < area_ratio <= 0.08:
            method = cv2.INPAINT_TELEA if params.inpaint_method == InpaintMethod.TELEA else cv2.INPAINT_NS
            suppressed_bgr = cv2.cvtColor(suppressed, cv2.COLOR_RGB2BGR)
            restored_bgr = cv2.inpaint(suppressed_bgr, mask, params.inpaint_radius, method)
            suppressed = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        context.metadata["reflection_mask"] = soft_mask if params.show_mask else None
        context.metadata["reflection_mask_area"] = area_ratio
        return suppressed


def _paint_polygons(mask: np.ndarray, polygons: list[list[list[float]]], value: int) -> None:
    height, width = mask.shape
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.float32)
        points *= np.array([width - 1, height - 1], np.float32)
        cv2.fillPoly(mask, [points.astype(np.int32)], value)
