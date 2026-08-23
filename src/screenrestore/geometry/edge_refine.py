"""在原始分辨率上沿预测边法线拟合真实内容边界。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rectify import order_corners
from .types import EdgeRefinement, quadrilateral_is_valid


@dataclass(frozen=True, slots=True)
class EdgeRefineParameters:
    """高分辨率边缘精修的受限参数。"""

    band_ratio: float = 0.018
    min_band_pixels: int = 5
    max_band_pixels: int = 72
    samples_per_edge: int = 96
    min_valid_samples: int = 20
    max_corner_shift_ratio: float = 0.04
    minimum_support: float = 0.16

    def __post_init__(self) -> None:
        if not 0.002 <= self.band_ratio <= 0.08:
            raise ValueError("边缘搜索带比例必须位于 0.002..0.08")
        if not 3 <= self.min_band_pixels <= self.max_band_pixels:
            raise ValueError("边缘搜索带像素范围无效")
        if not 24 <= self.samples_per_edge <= 512:
            raise ValueError("每条边采样数必须位于 24..512")
        if not 8 <= self.min_valid_samples <= self.samples_per_edge:
            raise ValueError("最少有效采样数无效")


def refine_quad_edges(
    image_rgb: np.ndarray,
    coarse_corners: np.ndarray,
    boundary_map: np.ndarray | None = None,
    params: EdgeRefineParameters | None = None,
) -> EdgeRefinement:
    """沿四条粗边寻找梯度峰并用稳健总最小二乘拟合直线。

    该函数只在精修结果仍为合理凸四边形、位移受限且四边均有足够证据时接受结果。
    失败时返回原粗角点，避免一次不可靠拟合破坏学习模型已经选对的语义层级。
    """

    _validate_image(image_rgb)
    params = params or EdgeRefineParameters()
    coarse = order_corners(coarse_corners)
    height, width = image_rgb.shape[:2]
    diagonal = max(1.0, float(np.hypot(width, height)))
    band = int(
        np.clip(
            round(min(height, width) * params.band_ratio),
            params.min_band_pixels,
            params.max_band_pixels,
        )
    )
    if image_rgb.dtype == np.uint8:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        clipped = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
        gray = cv2.cvtColor(clipped, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.GaussianBlur(gray, (0, 0), 1.1)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    boundary: np.ndarray | None = None
    if boundary_map is not None:
        boundary = np.asarray(boundary_map, dtype=np.float32)
        if boundary.ndim == 3:
            boundary = np.squeeze(boundary)
        if boundary.ndim != 2:
            raise ValueError("boundary_map 必须是二维数组")
        boundary = cv2.resize(boundary, (width, height), interpolation=cv2.INTER_LINEAR)
        boundary = np.clip(boundary, 0.0, 1.0)

    lines: list[np.ndarray] = []
    support: list[float] = []
    for index in range(4):
        edge = _fit_edge(
            coarse[index],
            coarse[(index + 1) % 4],
            gradient_x,
            gradient_y,
            boundary,
            band,
            params,
        )
        if edge is None:
            return EdgeRefinement(
                corners=coarse,
                accepted=False,
                edge_support=(0.0, 0.0, 0.0, 0.0),
                corner_shifts=(0.0, 0.0, 0.0, 0.0),
                reason=f"edge_{index}_insufficient",
            )
        line, edge_support = edge
        lines.append(line)
        support.append(edge_support)

    intersections = []
    for index in range(4):
        point = _line_intersection(lines[(index - 1) % 4], lines[index])
        if point is None:
            return EdgeRefinement(
                corners=coarse,
                accepted=False,
                edge_support=tuple(support),  # type: ignore[arg-type]
                corner_shifts=(0.0, 0.0, 0.0, 0.0),
                reason="parallel_refined_edges",
            )
        intersections.append(point)
    refined = order_corners(np.asarray(intersections, dtype=np.float32))
    shifts = np.linalg.norm(refined - coarse, axis=1)
    max_shift = params.max_corner_shift_ratio * diagonal
    accepted = (
        quadrilateral_is_valid(refined, image_rgb.shape)
        and float(np.max(shifts)) <= max_shift
        and float(np.min(support)) >= params.minimum_support
    )
    reason = "" if accepted else "refinement_gate_failed"
    return EdgeRefinement(
        corners=refined if accepted else coarse,
        accepted=accepted,
        edge_support=tuple(float(value) for value in support),  # type: ignore[arg-type]
        corner_shifts=tuple(float(value) for value in shifts),  # type: ignore[arg-type]
        reason=reason,
    )


def _fit_edge(
    start: np.ndarray,
    end: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    boundary: np.ndarray | None,
    band: int,
    params: EdgeRefineParameters,
) -> tuple[np.ndarray, float] | None:
    vector = end.astype(np.float64) - start.astype(np.float64)
    length = float(np.linalg.norm(vector))
    if length < 16:
        return None
    tangent = vector / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    sample_count = int(np.clip(round(length / 6.0), params.min_valid_samples, params.samples_per_edge))
    offsets = np.arange(-band, band + 1, dtype=np.float64)
    points: list[np.ndarray] = []
    strengths: list[float] = []
    height, width = gradient_x.shape
    for position in np.linspace(0.04, 0.96, sample_count):
        base = start.astype(np.float64) + position * vector
        coordinates = base[None, :] + offsets[:, None] * normal[None, :]
        xs = np.rint(coordinates[:, 0]).astype(np.int32)
        ys = np.rint(coordinates[:, 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.count_nonzero(valid) < 3:
            continue
        xs = xs[valid]
        ys = ys[valid]
        responses = np.abs(gradient_x[ys, xs] * normal[0] + gradient_y[ys, xs] * normal[1])
        if boundary is not None:
            responses = responses * (0.55 + 0.45 * boundary[ys, xs])
        peak = int(np.argmax(responses))
        points.append(np.array([xs[peak], ys[peak]], dtype=np.float64))
        strengths.append(float(responses[peak]))
    if len(points) < params.min_valid_samples:
        return None
    point_array = np.asarray(points, dtype=np.float64)
    strength_array = np.asarray(strengths, dtype=np.float64)
    positive = strength_array[strength_array > 0]
    if positive.size < params.min_valid_samples:
        return None
    threshold = max(float(np.percentile(positive, 28)), 0.012)
    keep = strength_array >= threshold
    if np.count_nonzero(keep) < params.min_valid_samples:
        return None
    line = _robust_total_least_squares(point_array, strength_array, keep, params.min_valid_samples)
    if line is None:
        return None
    # Sobel 响应的 0.35 左右已经是很强的边；该归一化只用于接受/拒绝，不做概率解释。
    support = float(np.clip(np.median(strength_array[keep]) / 0.35, 0.0, 1.0))
    return line, support


def _robust_total_least_squares(
    points: np.ndarray,
    strengths: np.ndarray,
    keep: np.ndarray,
    minimum: int,
) -> np.ndarray | None:
    line: np.ndarray | None = None
    for _ in range(4):
        selected = points[keep]
        if len(selected) < minimum:
            return None
        weights = np.sqrt(np.maximum(strengths[keep], 1e-5))
        center = np.average(selected, axis=0, weights=weights)
        centered = selected - center
        covariance = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-8)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        line = np.array([normal[0], normal[1], -float(np.dot(normal, center))])
        residuals = np.abs(points @ normal + line[2])
        selected_residuals = residuals[keep]
        median = float(np.median(selected_residuals))
        mad = float(np.median(np.abs(selected_residuals - median))) + 0.25
        keep = keep & (residuals <= max(1.5, median + 3.0 * mad))
    return line


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-8:
        return None
    return (point[:2] / point[2]).astype(np.float32)


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError("边缘精修需要 H×W×3 RGB uint8/float32 图像")
