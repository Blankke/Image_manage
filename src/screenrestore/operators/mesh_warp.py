"""弯曲银幕的规则控制网格校正。

控制点描述“校正后规则网格中的每个节点应从输入图的哪个位置采样”。输出像素到
输入像素的逆映射由二维样条生成，最后统一交给 OpenCV ``remap``，不会原地修改图像。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_rgb_float


class MeshInterpolation(StrEnum):
    """最终像素重采样方式。"""

    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS = "lanczos"


class MeshBorderMode(StrEnum):
    """控制网格采样触及边界时的策略。"""

    CONSTANT = "constant"
    REPLICATE = "replicate"
    REFLECT = "reflect"


@dataclass
class MeshWarpParameters(ParameterModel):
    """规则网格参数；控制点是 ``rows×columns×2`` 的归一化源坐标。"""

    rows: int = 5
    columns: int = 7
    control_points: list[list[list[float]]] = field(default_factory=list)
    strength: float = 1.0
    interpolation: MeshInterpolation = MeshInterpolation.CUBIC
    border_mode: MeshBorderMode = MeshBorderMode.REPLICATE

    def validate(self) -> None:
        if not 2 <= self.rows <= 15 or not 2 <= self.columns <= 15:
            raise ValueError("网格行列数必须位于 2..15")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("网格校正强度必须位于 [0, 1]")
        if not self.control_points:
            return
        points = np.asarray(self.control_points, dtype=np.float64)
        if points.shape != (self.rows, self.columns, 2):
            raise ValueError(
                f"控制网格必须是 {self.rows}×{self.columns}×2，实际为 {points.shape}"
            )
        if not np.all(np.isfinite(points)) or not np.all((points >= 0.0) & (points <= 1.0)):
            raise ValueError("控制网格坐标必须是 [0, 1] 内的有限数值")
        _validate_non_folded(points)


def regular_control_grid(rows: int, columns: int) -> np.ndarray:
    """创建 ``rows×columns`` 的归一化恒等控制网格。"""

    if rows < 2 or columns < 2:
        raise ValueError("恒等网格至少需要 2×2 个控制点")
    y_values = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    x_values = np.linspace(0.0, 1.0, columns, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    return np.stack((grid_x, grid_y), axis=2)


def curved_control_grid(
    rows: int,
    columns: int,
    horizontal_curve: float = 0.0,
    vertical_curve: float = 0.0,
) -> np.ndarray:
    """生成可作为手工编辑起点的对称弯曲银幕网格。

    正的 ``horizontal_curve`` 让上下边在中央向外弯，正的 ``vertical_curve``
    让左右边在中央向外弯；建议幅度位于 ``[-0.15, 0.15]``。
    """

    if not -0.25 <= horizontal_curve <= 0.25 or not -0.25 <= vertical_curve <= 0.25:
        raise ValueError("自动弯曲幅度必须位于 -0.25..0.25")
    grid = regular_control_grid(rows, columns).astype(np.float64)
    unit_x = grid[..., 0] * 2.0 - 1.0
    unit_y = grid[..., 1] * 2.0 - 1.0
    # 边界保持在原图内，中间节点按奇对称三次曲线向内/外弯，避免产生必然黑边。
    grid[..., 1] += (
        horizontal_curve
        * (1.0 - np.square(unit_x))
        * unit_y
        * (1.0 - np.square(unit_y))
    )
    grid[..., 0] += (
        vertical_curve
        * (1.0 - np.square(unit_y))
        * unit_x
        * (1.0 - np.square(unit_x))
    )
    grid = np.clip(grid, 0.0, 1.0)
    _validate_non_folded(grid)
    return grid.astype(np.float32)


def warp_mesh(
    image_rgb: np.ndarray,
    params: MeshWarpParameters,
) -> tuple[np.ndarray, dict[str, object]]:
    """按规则控制网格校正弯曲屏幕，并返回映射诊断。"""

    _validate_rgb(image_rgb)
    params.validate()
    height, width = image_rgb.shape[:2]
    regular = regular_control_grid(params.rows, params.columns).astype(np.float64)
    source = (
        np.asarray(params.control_points, dtype=np.float64)
        if params.control_points
        else regular.copy()
    )
    source = regular + (source - regular) * params.strength
    _validate_non_folded(source)

    node_y = np.linspace(0.0, 1.0, params.rows, dtype=np.float64)
    node_x = np.linspace(0.0, 1.0, params.columns, dtype=np.float64)
    output_y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    output_x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    spline_order_y = min(3, params.rows - 1)
    spline_order_x = min(3, params.columns - 1)
    source_x = RectBivariateSpline(
        node_y,
        node_x,
        source[..., 0],
        kx=spline_order_y,
        ky=spline_order_x,
        s=0.0,
    )(output_y, output_x)
    source_y = RectBivariateSpline(
        node_y,
        node_x,
        source[..., 1],
        kx=spline_order_y,
        ky=spline_order_x,
        s=0.0,
    )(output_y, output_x)
    valid = (source_x >= 0.0) & (source_x <= 1.0) & (source_y >= 0.0) & (source_y <= 1.0)
    map_x = np.clip(source_x * (width - 1), 0.0, width - 1).astype(np.float32)
    map_y = np.clip(source_y * (height - 1), 0.0, height - 1).astype(np.float32)
    interpolation = {
        MeshInterpolation.LINEAR: cv2.INTER_LINEAR,
        MeshInterpolation.CUBIC: cv2.INTER_CUBIC,
        MeshInterpolation.LANCZOS: cv2.INTER_LANCZOS4,
    }[params.interpolation]
    border_mode = {
        MeshBorderMode.CONSTANT: cv2.BORDER_CONSTANT,
        MeshBorderMode.REPLICATE: cv2.BORDER_REPLICATE,
        MeshBorderMode.REFLECT: cv2.BORDER_REFLECT_101,
    }[params.border_mode]
    output = cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=(0, 0, 0),
    )
    displacement = np.sqrt(
        np.square(source[..., 0] - regular[..., 0])
        + np.square(source[..., 1] - regular[..., 1])
    )
    metadata: dict[str, object] = {
        "grid_size": [params.columns, params.rows],
        "strength": params.strength,
        "max_normalized_displacement": float(displacement.max(initial=0.0)),
        "valid_fraction": float(valid.mean()),
    }
    return np.ascontiguousarray(output), metadata


class MeshWarpOperator(ImageOperator[MeshWarpParameters]):
    """在平面透视校正后消除弯曲银幕的非线性形变。"""

    id = "mesh_warp"
    display_name = "弯曲银幕网格校正"
    parameter_type = MeshWarpParameters
    reorderable = False

    def default_parameters(self) -> MeshWarpParameters:
        return MeshWarpParameters()

    def apply(
        self,
        image: np.ndarray,
        params: MeshWarpParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        context.report(0.05, "准备弯曲银幕网格")
        output, metadata = warp_mesh(image, params)
        context.metadata["mesh_warp"] = metadata
        context.report(1.0, "弯曲银幕校正完成")
        return clip_float(output)


def _validate_non_folded(points: np.ndarray) -> None:
    """拒绝翻折或退化单元，避免同一输出位置映射到相互交叉的源区域。"""

    signs: list[float] = []
    for row in range(points.shape[0] - 1):
        for column in range(points.shape[1] - 1):
            top_left = points[row, column]
            top_right = points[row, column + 1]
            bottom_left = points[row + 1, column]
            horizontal = top_right - top_left
            vertical = bottom_left - top_left
            cross = float(horizontal[0] * vertical[1] - horizontal[1] * vertical[0])
            if abs(cross) < 1e-7:
                raise ValueError("控制网格包含退化单元")
            signs.append(cross)
    if signs and (min(signs) <= 0.0 <= max(signs)):
        raise ValueError("控制网格发生翻折或相邻单元方向不一致")
    if signs and max(signs) < 0.0:
        raise ValueError("控制网格方向颠倒")


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError("网格模块需要 H×W×3 RGB uint8/float32 图像")
