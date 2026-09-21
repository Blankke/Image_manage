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
    minimum_continuous_coverage: float = 0.18
    minimum_normal_alignment: float = 0.45
    max_residual_p95: float = 2.8
    max_area_drift: float = 0.12
    max_aspect_drift: float = 0.10

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
    edge_diagnostics: list[dict[str, float]] = []
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
        line, edge_support, diagnostics = edge
        lines.append(line)
        support.append(edge_support)
        edge_diagnostics.append(diagnostics)

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
    coarse_area = max(abs(float(cv2.contourArea(coarse))), 1e-6)
    refined_area = abs(float(cv2.contourArea(refined)))
    area_drift = abs(refined_area / coarse_area - 1.0)
    coarse_aspect = _quad_aspect(coarse)
    refined_aspect = _quad_aspect(refined)
    aspect_drift = abs(refined_aspect / max(coarse_aspect, 1e-6) - 1.0)
    accepted = (
        quadrilateral_is_valid(refined, image_rgb.shape)
        and float(np.max(shifts)) <= max_shift
        and float(np.min(support)) >= params.minimum_support
        and min(item["continuous_coverage"] for item in edge_diagnostics)
        >= params.minimum_continuous_coverage
        and min(item["normal_alignment"] for item in edge_diagnostics)
        >= params.minimum_normal_alignment
        and max(item["residual_p95"] for item in edge_diagnostics) <= params.max_residual_p95
        and area_drift <= params.max_area_drift
        and aspect_drift <= params.max_aspect_drift
    )
    reason = "" if accepted else "refinement_gate_failed"
    outcome = "neutral"
    if accepted and float(np.mean(shifts)) >= 0.25:
        outcome = "improved"
    elif not accepted:
        outcome = "rolled_back"
    return EdgeRefinement(
        corners=refined if accepted else coarse,
        accepted=accepted,
        edge_support=tuple(float(value) for value in support),  # type: ignore[arg-type]
        corner_shifts=tuple(float(value) for value in shifts),  # type: ignore[arg-type]
        attempted_corners=refined,
        reason=reason,
        residual_median=tuple(item["residual_median"] for item in edge_diagnostics),  # type: ignore[arg-type]
        residual_p95=tuple(item["residual_p95"] for item in edge_diagnostics),  # type: ignore[arg-type]
        continuous_coverage=tuple(item["continuous_coverage"] for item in edge_diagnostics),  # type: ignore[arg-type]
        gradient_normal_alignment=tuple(item["normal_alignment"] for item in edge_diagnostics),  # type: ignore[arg-type]
        boundary_consistency=tuple(item["boundary_consistency"] for item in edge_diagnostics),  # type: ignore[arg-type]
        area_drift=float(area_drift),
        aspect_drift=float(aspect_drift),
        outcome=outcome,
    )


def _fit_edge(
    start: np.ndarray,
    end: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    boundary: np.ndarray | None,
    band: int,
    params: EdgeRefineParameters,
) -> tuple[np.ndarray, float, dict[str, float]] | None:
    vector = end.astype(np.float64) - start.astype(np.float64)
    length = float(np.linalg.norm(vector))
    if length < 16:
        return None
    tangent = vector / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    sample_count = int(
        np.clip(round(length / 6.0), params.min_valid_samples, params.samples_per_edge)
    )
    offsets = np.arange(-band, band + 1, dtype=np.float64)
    bases: list[np.ndarray] = []
    response_rows: list[np.ndarray] = []
    raw_response_rows: list[np.ndarray] = []
    magnitude_rows: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    height, width = gradient_x.shape
    for position in np.linspace(0.04, 0.96, sample_count):
        base = start.astype(np.float64) + position * vector
        coordinates = base[None, :] + offsets[:, None] * normal[None, :]
        xs = np.rint(coordinates[:, 0]).astype(np.int32)
        ys = np.rint(coordinates[:, 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.count_nonzero(valid) < 3:
            continue
        valid_indices = np.flatnonzero(valid)
        valid_xs = xs[valid]
        valid_ys = ys[valid]
        raw_responses = np.abs(
            gradient_x[valid_ys, valid_xs] * normal[0]
            + gradient_y[valid_ys, valid_xs] * normal[1]
        )
        responses = raw_responses.copy()
        boundary_values = np.zeros_like(responses)
        if boundary is not None:
            boundary_values = boundary[valid_ys, valid_xs]
            responses = responses * (0.55 + 0.45 * boundary_values)
        response_row = np.full(len(offsets), -np.inf, dtype=np.float64)
        raw_response_row = np.zeros(len(offsets), dtype=np.float64)
        magnitude_row = np.zeros(len(offsets), dtype=np.float64)
        boundary_row = np.zeros(len(offsets), dtype=np.float64)
        response_row[valid_indices] = responses
        raw_response_row[valid_indices] = raw_responses
        magnitude_row[valid_indices] = np.hypot(
            gradient_x[valid_ys, valid_xs], gradient_y[valid_ys, valid_xs]
        )
        boundary_row[valid_indices] = boundary_values
        bases.append(base)
        response_rows.append(response_row)
        raw_response_rows.append(raw_response_row)
        magnitude_rows.append(magnitude_row)
        boundary_rows.append(boundary_row)
    if len(bases) < params.min_valid_samples:
        return None
    response_matrix = np.asarray(response_rows, dtype=np.float64)
    peak_indices = _straight_edge_peak_indices(response_matrix, offsets, band)
    base_array = np.asarray(bases, dtype=np.float64)
    selected_offsets = offsets[peak_indices]
    point_array = base_array + selected_offsets[:, None] * normal[None, :]
    point_array[:, 0] = np.clip(np.rint(point_array[:, 0]), 0, width - 1)
    point_array[:, 1] = np.clip(np.rint(point_array[:, 1]), 0, height - 1)
    row_indices = np.arange(len(peak_indices))
    strength_array = response_matrix[row_indices, peak_indices]
    raw_strength = np.asarray(raw_response_rows)[row_indices, peak_indices]
    magnitudes = np.asarray(magnitude_rows)[row_indices, peak_indices]
    alignment_array = np.clip(raw_strength / np.maximum(magnitudes, 1e-8), 0.0, 1.0)
    boundary_array = np.asarray(boundary_rows)[row_indices, peak_indices]
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
    residuals = np.abs(point_array @ line[:2] + line[2])
    # 连续覆盖率描述“沿边是否持续存在可观测梯度”。拟合时丢弃低分位异常点是
    # 稳健估计手段，不能反过来人为制造 28% 的断点，因此这里使用绝对观测门。
    kept_indices = np.flatnonzero(strength_array >= 0.012)
    longest_run = _longest_consecutive_run(kept_indices)
    diagnostics = {
        "residual_median": float(np.median(residuals[keep])),
        "residual_p95": float(np.percentile(residuals[keep], 95)),
        "continuous_coverage": float(longest_run / max(1, sample_count)),
        "normal_alignment": float(np.median(alignment_array[keep])),
        "boundary_consistency": float(np.median(boundary_array[keep]))
        if boundary is not None
        else 0.0,
    }
    return line, support, diagnostics


def _straight_edge_peak_indices(
    responses: np.ndarray,
    offsets: np.ndarray,
    band: int,
) -> np.ndarray:
    """在窄带内联合搜索整条直线，抑制纹理峰之间的逐点跳跃。

    平面矩形经针孔投影后每条边仍为直线，因此同时搜索起点与终点法向偏移，比逐点
    贪心或仅约束相邻步长更贴合成像模型。分数兼顾低分位响应和均值，只有持续可见的
    边才能胜过偶发的内部纹理；轻微零位移先验用于避免无证据时跳到相邻外框。
    """

    if responses.ndim != 2 or responses.shape[0] < 2 or responses.shape[1] != len(offsets):
        raise ValueError("边缘响应必须为 N×offsets 且至少包含两个采样点")
    finite = np.isfinite(responses)
    positive = responses[finite & (responses > 0)]
    if positive.size == 0:
        return np.zeros(responses.shape[0], dtype=np.int64)
    scale = max(float(np.percentile(positive, 90)), 0.012)
    emission = np.where(finite, np.clip(responses / scale, 0.0, 2.5), -2.5)
    rows, columns = emission.shape
    # 两像素粗搜将候选数控制在约 5k；随后每个采样点只在预测直线附近一像素内
    # 取峰，既保留亚像素前的局部适应，也不会跳到远处纹理。
    candidate_columns = np.arange(0, columns, 2, dtype=np.int32)
    if candidate_columns[-1] != columns - 1:
        candidate_columns = np.r_[candidate_columns, columns - 1]
    start = candidate_columns[:, None, None]
    end = candidate_columns[None, :, None]
    positions = np.linspace(0.0, 1.0, rows, dtype=np.float64)[None, None, :]
    paths = np.rint(start * (1.0 - positions) + end * positions).astype(np.int32)
    path_values = emission[np.arange(rows)[None, None, :], paths]
    scores = 0.65 * np.percentile(path_values, 25, axis=2) + 0.35 * np.mean(
        path_values, axis=2
    )
    normalized = np.abs(offsets[candidate_columns]) / max(1.0, float(band))
    scores -= 0.05 * (normalized[:, None] ** 2 + normalized[None, :] ** 2)
    best_start, best_end = np.unravel_index(int(np.argmax(scores)), scores.shape)
    result = paths[best_start, best_end].copy()
    for row in range(rows):
        left = max(0, int(result[row]) - 1)
        right = min(columns, int(result[row]) + 2)
        result[row] = left + int(np.argmax(responses[row, left:right]))
    return result.astype(np.int64)


def _longest_consecutive_run(indices: np.ndarray) -> int:
    if indices.size == 0:
        return 0
    differences = np.diff(indices)
    boundaries = np.flatnonzero(differences != 1)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries + 1, len(indices)]
    return int(np.max(ends - starts))


def _quad_aspect(corners: np.ndarray) -> float:
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    return (top + bottom) / max(left + right, 1e-6)


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
