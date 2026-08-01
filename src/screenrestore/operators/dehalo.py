"""针对暗场高亮边缘的受限光学扩散光晕抑制。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


@dataclass
class DehaloParameters(ParameterModel):
    """高亮扩散层参数；自动模式只在暗场且存在有限高亮面积时触发。"""

    strength: float = 0.25
    highlight_threshold: float = 0.65
    core_radius: float = 0.7
    halo_radius: float = 2.2
    max_correction: float = 0.06
    auto_gate: bool = True
    max_scene_median: float = 0.32
    min_highlight_area: float = 0.02
    max_highlight_area: float = 0.25

    def validate(self) -> None:
        require_range("strength", self.strength, 0.0, 1.0)
        require_range("highlight_threshold", self.highlight_threshold, 0.4, 0.95)
        require_range("core_radius", self.core_radius, 0.3, 5.0)
        require_range("halo_radius", self.halo_radius, 0.8, 20.0)
        require_range("max_correction", self.max_correction, 0.0, 0.3)
        require_range("max_scene_median", self.max_scene_median, 0.05, 0.8)
        require_range("min_highlight_area", self.min_highlight_area, 0.0, 0.5)
        require_range("max_highlight_area", self.max_highlight_area, 0.01, 1.0)
        if self.halo_radius <= self.core_radius:
            raise ValueError("halo_radius 必须大于 core_radius")
        if self.max_highlight_area <= self.min_highlight_area:
            raise ValueError("max_highlight_area 必须大于 min_highlight_area")


class DehaloOperator(ImageOperator[DehaloParameters]):
    """估计高亮核心外的宽尺度散射光，不通过锐化制造新边缘。"""

    id = "dehalo"
    display_name = "高光光晕抑制"
    parameter_type = DehaloParameters

    def default_parameters(self) -> DehaloParameters:
        return DehaloParameters()

    def apply(
        self,
        image: np.ndarray,
        params: DehaloParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        if params.strength == 0.0 or params.max_correction == 0.0:
            return image.copy()

        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        luminance = ycrcb[..., 0]
        scene_median = float(np.median(luminance))
        highlight_area = float(np.mean(luminance > params.highlight_threshold))
        activated = not params.auto_gate or (
            scene_median <= params.max_scene_median
            and params.min_highlight_area <= highlight_area <= params.max_highlight_area
        )
        metadata: dict[str, float | bool] = {
            "scene_median": scene_median,
            "highlight_area": highlight_area,
            "activated": activated,
            "mean_correction": 0.0,
        }
        if not activated:
            context.metadata["dehalo"] = metadata
            return image.copy()

        context.report(0.2, "估计高亮扩散光晕")
        bright = np.clip(
            (luminance - params.highlight_threshold)
            / max(1e-6, 1.0 - params.highlight_threshold),
            0.0,
            1.0,
        )
        core = cv2.GaussianBlur(bright, (0, 0), params.core_radius)
        wide = cv2.GaussianBlur(bright, (0, 0), params.halo_radius)
        halo = np.clip(wide - core, 0.0, params.max_correction)
        # 高亮核心本身保持不变，只扣除其周围的正向宽扩散层。
        correction = halo * params.strength * np.clip(1.0 - bright, 0.0, 1.0)
        ycrcb[..., 0] = np.clip(luminance - correction, 0.0, 1.0)
        metadata["mean_correction"] = float(np.mean(correction))
        metadata["p95_correction"] = float(np.quantile(correction, 0.95))
        context.metadata["dehalo"] = metadata
        context.report(1.0, "高光光晕抑制完成")
        return clip_float(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
