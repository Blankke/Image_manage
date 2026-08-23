"""平面几何流水线算子。

四边形检测、置信度拒绝、原图边缘精修与单应数学位于 ``screenrestore.geometry``；
本模块只保留可序列化参数和流水线适配，避免算法重新散落到算子类中。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel
from screenrestore.geometry import AspectRatioMode, InterpolationMode, warp_perspective

from ._utils import clip_float, require_rgb_float


@dataclass
class GeometryParameters(ParameterModel):
    """几何算子参数；四角使用相对当前输入宽高的归一化坐标。"""

    corners: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    ratio_mode: AspectRatioMode = AspectRatioMode.AUTO
    custom_ratio: float = 16 / 9
    rotation: int = 0
    black_border: int = 0
    interpolation: InterpolationMode = InterpolationMode.LANCZOS
    auto_crop: bool = True

    def validate(self) -> None:
        if len(self.corners) != 4 or any(len(point) != 2 for point in self.corners):
            raise ValueError("几何校正必须提供四个二维角点")
        values = np.asarray(self.corners, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("角点必须是有限数值")
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise ValueError("归一化角点必须位于 [0, 1]")
        if self.custom_ratio <= 0 or not np.isfinite(self.custom_ratio):
            raise ValueError("自定义比例必须大于 0")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("旋转角度仅支持 0/90/180/270")
        if not 0 <= self.black_border <= 4096:
            raise ValueError("黑边宽度必须位于 0..4096")


class GeometryOperator(ImageOperator[GeometryParameters]):
    """把归一化四角映射为当前输入尺寸并执行透视校正。"""

    id = "geometry"
    display_name = "几何校正"
    parameter_type = GeometryParameters
    reorderable = False

    def default_parameters(self) -> GeometryParameters:
        return GeometryParameters()

    def apply(
        self,
        image: np.ndarray,
        params: GeometryParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        context.report(0.05, "准备透视校正")
        height, width = image.shape[:2]
        normalized = np.asarray(params.corners, dtype=np.float32)
        corners = normalized * np.array([width - 1, height - 1], dtype=np.float32)
        output, matrix = warp_perspective(
            image,
            corners,
            params.ratio_mode,
            params.custom_ratio,
            params.interpolation,
            params.rotation,
            params.black_border,
            params.auto_crop,
        )
        context.metadata["geometry_matrix"] = matrix.tolist()
        context.report(1.0, "透视校正完成")
        return clip_float(output)
