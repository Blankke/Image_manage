"""方向、镜头、透视与轻度 dewarp 的统一逆映射和单次采样。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class InverseMap:
    """每个输出像素在上一级输入坐标系中的浮点采样位置。"""

    map_x: np.ndarray
    map_y: np.ndarray
    source: str

    def __post_init__(self) -> None:
        x = np.asarray(self.map_x, np.float32)
        y = np.asarray(self.map_y, np.float32)
        if x.ndim != 2 or x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("逆映射必须是形状一致的有限二维 map_x/map_y")
        object.__setattr__(self, "map_x", x.copy())
        object.__setattr__(self, "map_y", y.copy())

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.map_x.shape


@dataclass(frozen=True, slots=True)
class RadialLensParameters:
    """P3 experimental 的最小 Brown 径向参数集合。"""

    k1: float = 0.0
    k2: float = 0.0
    center_x: float = 0.5
    center_y: float = 0.5
    source: str = "identity"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("镜头参数置信度必须位于 [0,1]")
        _validate_radial_coefficients(self.k1, self.k2, self.center_x, self.center_y)


def identity_inverse_map(shape: tuple[int, ...], source: str = "identity") -> InverseMap:
    height, width = shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    return InverseMap(xx, yy, source)


def orientation_inverse_map(
    input_shape: tuple[int, ...],
    exif_orientation: int,
) -> InverseMap:
    """生成 EXIF 1/3/6/8 方向校正后的输出到原图映射。"""

    height, width = input_shape[:2]
    if exif_orientation == 1:
        return identity_inverse_map(input_shape, "orientation:1")
    if exif_orientation == 3:
        yy, xx = np.indices((height, width), dtype=np.float32)
        return InverseMap(width - 1 - xx, height - 1 - yy, "orientation:3")
    if exif_orientation == 6:
        yy, xx = np.indices((width, height), dtype=np.float32)
        return InverseMap(yy, height - 1 - xx, "orientation:6")
    if exif_orientation == 8:
        yy, xx = np.indices((width, height), dtype=np.float32)
        return InverseMap(width - 1 - yy, xx, "orientation:8")
    raise ValueError("P3 逆映射仅接受已归一化的 EXIF 方向 1/3/6/8")


def homography_inverse_map(
    source_to_output: np.ndarray,
    output_shape: tuple[int, int],
) -> InverseMap:
    """由输入到输出单应矩阵生成 output→input 逆映射。"""

    matrix = np.asarray(source_to_output, np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("单应矩阵必须是有限 3×3 数组")
    inverse = np.linalg.inv(matrix)
    height, width = output_shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    homogeneous = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ inverse.T
    denominator = homogeneous[..., 2]
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("单应逆映射在输出区域内出现无穷远点")
    return InverseMap(
        (homogeneous[..., 0] / denominator).astype(np.float32),
        (homogeneous[..., 1] / denominator).astype(np.float32),
        "homography",
    )


def radial_inverse_map(
    shape: tuple[int, ...],
    *,
    k1: float = 0.0,
    k2: float = 0.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> InverseMap:
    """Brown 径向模型的未畸变输出到原始畸变输入映射。"""

    _validate_radial_coefficients(k1, k2, center_x, center_y)
    if k1 == 0.0 and k2 == 0.0:
        return identity_inverse_map(shape, "radial:identity")
    height, width = shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    scale = float(max(width - 1, height - 1, 1))
    cx, cy = center_x * (width - 1), center_y * (height - 1)
    x = (xx - cx) / scale
    y = (yy - cy) / scale
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
    derivative = 1.0 + 3.0 * k1 * radius2 + 5.0 * k2 * radius2 * radius2
    if float(derivative.min()) <= 0.05:
        raise ValueError("径向模型不满足单调/双射安全门")
    return InverseMap(x * radial * scale + cx, y * radial * scale + cy, "radial:k1-k2")


def safe_radial_inverse_map(
    shape: tuple[int, ...],
    parameters: RadialLensParameters | None,
    *,
    minimum_confidence: float = 0.8,
) -> tuple[InverseMap, dict[str, object]]:
    """按 profile/user/estimator 已确定的参数执行安全门，异常时明确 bypass。"""

    if parameters is None or parameters.confidence < minimum_confidence:
        reason = "missing_parameters" if parameters is None else "low_confidence"
        return identity_inverse_map(shape, "radial:bypass"), {
            "applied": False,
            "reason": reason,
            "source": parameters.source if parameters else "none",
        }
    try:
        mapping = radial_inverse_map(
            shape,
            k1=parameters.k1,
            k2=parameters.k2,
            center_x=parameters.center_x,
            center_y=parameters.center_y,
        )
    except ValueError as exc:
        return identity_inverse_map(shape, "radial:bypass"), {
            "applied": False,
            "reason": str(exc),
            "source": parameters.source,
        }
    return mapping, {
        "applied": True,
        "reason": "accepted",
        "source": parameters.source,
        "confidence": parameters.confidence,
    }


def radial_straight_line_residual(mapping: InverseMap, samples: int = 9) -> float:
    """用映射后水平/竖直测试线的 TLS 残差衡量弯曲强度。"""

    height, width = mapping.output_shape
    residuals: list[float] = []
    for y in np.linspace(0, height - 1, samples):
        row = int(round(y))
        points = np.stack((mapping.map_x[row], mapping.map_y[row]), axis=1)
        residuals.append(_line_residual(points))
    for x in np.linspace(0, width - 1, samples):
        column = int(round(x))
        points = np.stack((mapping.map_x[:, column], mapping.map_y[:, column]), axis=1)
        residuals.append(_line_residual(points))
    return float(np.mean(residuals))


def dense_grid_inverse_map(
    displacement_grid: np.ndarray,
    output_shape: tuple[int, int],
) -> InverseMap:
    """把低分辨率归一化逆位移网格上采样为像素映射。"""

    grid = np.asarray(displacement_grid, np.float32)
    if grid.ndim != 3 or grid.shape[2] != 2 or min(grid.shape[:2]) < 2:
        raise ValueError("dense grid 必须为 rows×columns×2")
    if not np.all(np.isfinite(grid)) or float(np.max(np.abs(grid))) > 0.25:
        raise ValueError("dense grid 位移超出 P3 轻度 dewarp 范围")
    height, width = output_shape
    displacement = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx + displacement[..., 0] * max(1, width - 1)
    map_y = yy + displacement[..., 1] * max(1, height - 1)
    if minimum_jacobian_determinant(map_x, map_y) <= 0.0:
        raise ValueError("dense grid 出现折叠，未通过正 Jacobian 门")
    return InverseMap(map_x, map_y, f"dense-grid:{grid.shape[0]}x{grid.shape[1]}")


def compose_inverse_maps(maps: list[InverseMap]) -> InverseMap:
    """按正向处理顺序合成多级逆映射，返回最终输出到原始输入的坐标。"""

    if not maps:
        raise ValueError("至少需要一个逆映射")
    map_x = maps[0].map_x.copy()
    map_y = maps[0].map_y.copy()
    sources = [maps[0].source]
    for next_map in maps[1:]:
        map_x = cv2.remap(
            map_x,
            next_map.map_x,
            next_map.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        )
        map_y = cv2.remap(
            map_y,
            next_map.map_x,
            next_map.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        )
        sources.append(next_map.source)
    return InverseMap(map_x, map_y, " -> ".join(sources))


def remap_original_once(
    original_rgb: np.ndarray,
    inverse_map: InverseMap,
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """对只读原图执行唯一一次几何重采样。"""

    if original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
        raise ValueError("单次重映射输入必须为 H×W×3 RGB")
    return cv2.remap(
        original_rgb,
        inverse_map.map_x,
        inverse_map.map_y,
        interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def minimum_jacobian_determinant(map_x: np.ndarray, map_y: np.ndarray) -> float:
    dx_x = np.gradient(map_x, axis=1)
    dx_y = np.gradient(map_x, axis=0)
    dy_x = np.gradient(map_y, axis=1)
    dy_y = np.gradient(map_y, axis=0)
    return float(np.min(dx_x * dy_y - dx_y * dy_x))


def _validate_radial_coefficients(
    k1: float,
    k2: float,
    center_x: float,
    center_y: float,
) -> None:
    if not (-0.6 <= k1 <= 0.6 and -0.4 <= k2 <= 0.4):
        raise ValueError("径向参数超出 P3 experimental 安全范围")
    if not (0.35 <= center_x <= 0.65 and 0.35 <= center_y <= 0.65):
        raise ValueError("光心必须位于归一化画幅中心安全区")
    radius2 = np.linspace(0.0, 0.55, 256, dtype=np.float64)
    derivative = 1.0 + 3.0 * k1 * radius2 + 5.0 * k2 * radius2 * radius2
    if float(derivative.min()) <= 0.05:
        raise ValueError("径向模型不满足单调/双射安全门")


def _line_residual(points: np.ndarray) -> float:
    center = points.mean(axis=0)
    _u, _s, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    return float(np.median(np.abs((points - center) @ normal)))


__all__ = [
    "InverseMap",
    "RadialLensParameters",
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
]
