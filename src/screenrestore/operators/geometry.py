"""平面屏幕四边形检测与透视恢复。

数学假设：本算子中的目标内容位于同一平面，可由单应矩阵描述。镜头畸变由上游
``lens_distortion`` 处理，弯曲银幕由下游 ``mesh_warp`` 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_rgb_float


class AspectRatioMode(StrEnum):
    """透视输出比例模式。"""

    AUTO = "auto"
    FREE = "free"
    ESTIMATED = "estimated"
    RATIO_16_9 = "16:9"
    RATIO_16_10 = "16:10"
    RATIO_4_3 = "4:3"
    RATIO_1_85 = "1.85:1"
    RATIO_2_39 = "2.39:1"
    CUSTOM = "custom"


class InterpolationMode(StrEnum):
    """OpenCV 透视插值模式。"""

    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS = "lanczos"


@dataclass
class GeometryParameters(ParameterModel):
    """几何算子参数；四角使用相对原图宽高的归一化坐标。"""

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


@dataclass(frozen=True, slots=True)
class QuadrilateralCandidate:
    """自动检测得到的有序四边形及可解释评分。"""

    corners: np.ndarray
    confidence: float
    scores: dict[str, float]


def order_corners(corners: np.ndarray) -> np.ndarray:
    """把四点稳定排序为左上、右上、右下、左下。"""

    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("角点包含非有限数值")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # 图像坐标 y 向下；角度排序通常从左上附近开始，再定位 x+y 最小点。
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # 期望 TL→TR→BR→BL；若第二点位于第四点左侧则反转绕行方向。
    if ordered[1, 0] < ordered[3, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    area = abs(float(cv2.contourArea(ordered)))
    if area < 1e-3:
        raise ValueError("角点退化，无法构成有效四边形")
    return ordered.astype(np.float32)


def detect_quadrilaterals(
    image_rgb: np.ndarray,
    max_candidates: int = 5,
    detection_max_edge: int = 1200,
) -> list[QuadrilateralCandidate]:
    """使用传统轮廓和多项可解释指标返回屏幕四边形候选。"""

    _validate_rgb(image_rgb)
    height, width = image_rgb.shape[:2]
    scale = min(1.0, detection_max_edge / max(height, width))
    if scale < 1.0:
        small_rgb = cv2.resize(
            image_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_rgb = image_rgb
    gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 7, 45, 45)
    median = float(np.median(filtered))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 20, 1.33 * median)))
    edges = cv2.Canny(filtered, lower, upper)
    kernel_size = max(3, int(round(min(small_rgb.shape[:2]) * 0.008)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 屏幕常贴近甚至超出照片边缘；补上图像边界可闭合这类轮廓，后续排除整幅外框。
    cv2.rectangle(closed, (0, 0), (closed.shape[1] - 1, closed.shape[0] - 1), 255, 2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(gray.shape[0] * gray.shape[1])
    image_center = np.array([gray.shape[1] / 2, gray.shape[0] / 2], dtype=np.float32)
    candidates: list[QuadrilateralCandidate] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < image_area * 0.04:
            break
        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.018, 0.025, 0.035):
            polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            corners_small = order_corners(polygon.reshape(4, 2))
            polygon_area = abs(float(cv2.contourArea(corners_small)))
            if polygon_area < image_area * 0.04:
                continue
            if polygon_area > image_area * 0.985:
                continue
            scores = _candidate_scores(corners_small, edges, polygon_area, image_area, image_center)
            confidence = (
                0.30 * scores["area"]
                + 0.22 * scores["rectangularity"]
                + 0.22 * scores["edge_strength"]
                + 0.14 * scores["center"]
                + 0.12 * scores["side_balance"]
            )
            corners_full = corners_small / scale
            candidate = QuadrilateralCandidate(corners_full, float(confidence), scores)
            if not _duplicates_existing(candidate, candidates, max(height, width) * 0.025):
                candidates.append(candidate)
            break

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    if not candidates:
        profile_corners = _profile_boundary_candidate(gray, edges)
        if profile_corners is not None:
            polygon_area = abs(float(cv2.contourArea(profile_corners)))
            scores = _candidate_scores(
                profile_corners,
                edges,
                polygon_area,
                image_area,
                image_center,
            )
            # 亮度剖面候选缺少完整闭合轮廓证据，因此置信度刻意保守。
            confidence = min(
                0.68,
                0.26 * scores["area"]
                + 0.18 * scores["rectangularity"]
                + 0.18 * scores["edge_strength"]
                + 0.14 * scores["center"]
                + 0.12 * scores["side_balance"]
                + 0.08,
            )
            candidates.append(
                QuadrilateralCandidate(profile_corners / scale, float(confidence), scores)
            )
    return candidates[:max_candidates]


def estimate_output_size(
    corners: np.ndarray,
    ratio_mode: AspectRatioMode = AspectRatioMode.AUTO,
    custom_ratio: float = 16 / 9,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """估计输出尺寸，并在可能时恢复矩形平面的真实纵横比。

    ``max(对边长度)`` 在强透视下会把纵向海报明显拉宽。AUTO 模式会在给出
    原图尺寸时，假设零倾斜像素、主点位于图像中心，用两组消失点的正交约束
    估计焦距，再从单应矩阵列向量恢复矩形宽高比。内参近似不成立、消失点位于
    无穷远或数值不稳定时，会安全回退到投影边长估计。
    """

    tl, tr, br, bl = order_corners(corners)
    estimated_width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    estimated_height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    estimated_width = max(1.0, float(estimated_width))
    estimated_height = max(1.0, float(estimated_height))
    estimated_ratio = estimated_width / estimated_height
    rectified_ratio = (
        estimate_rectified_aspect_ratio(corners, image_shape)
        if image_shape is not None and ratio_mode == AspectRatioMode.AUTO
        else None
    )
    target_ratio = _resolve_ratio(
        ratio_mode,
        estimated_ratio,
        custom_ratio,
        rectified_ratio=rectified_ratio,
    )
    if target_ratio is None:
        return max(1, round(estimated_width)), max(1, round(estimated_height))
    area = estimated_width * estimated_height
    height = max(1, round(np.sqrt(area / target_ratio)))
    width = max(1, round(height * target_ratio))
    return width, height


def warp_perspective(
    image_rgb: np.ndarray,
    corners: np.ndarray,
    ratio_mode: AspectRatioMode = AspectRatioMode.AUTO,
    custom_ratio: float = 16 / 9,
    interpolation: InterpolationMode = InterpolationMode.LANCZOS,
    rotation: int = 0,
    black_border: int = 0,
    auto_crop: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """执行单应性校正并返回 RGB 输出和 3×3 变换矩阵。"""

    _validate_rgb(image_rgb)
    source = order_corners(corners)
    width, height = estimate_output_size(
        source,
        ratio_mode,
        custom_ratio,
        image_shape=image_rgb.shape,
    )
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    flags = {
        InterpolationMode.LINEAR: cv2.INTER_LINEAR,
        InterpolationMode.CUBIC: cv2.INTER_CUBIC,
        InterpolationMode.LANCZOS: cv2.INTER_LANCZOS4,
    }[interpolation]
    warped = cv2.warpPerspective(
        image_rgb,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if auto_crop:
        valid_source = np.full(image_rgb.shape[:2], 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            valid_source,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_points = cv2.findNonZero((valid_mask >= 250).astype(np.uint8))
        if valid_points is not None:
            crop_x, crop_y, crop_width, crop_height = cv2.boundingRect(valid_points)
            if crop_width > 1 and crop_height > 1:
                warped = warped[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
                crop_matrix = np.array(
                    [[1, 0, -crop_x], [0, 1, -crop_y], [0, 0, 1]], dtype=np.float64
                )
                matrix = crop_matrix @ matrix
    if rotation:
        rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[rotation]
        warped = cv2.rotate(warped, rotate_code)
        rotated_height, rotated_width = warped.shape[:2]
        if rotation == 90:
            rotation_matrix = np.array(
                [[0, -1, rotated_width - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64
            )
        elif rotation == 180:
            rotation_matrix = np.array(
                [[-1, 0, rotated_width - 1], [0, -1, rotated_height - 1], [0, 0, 1]],
                dtype=np.float64,
            )
        else:
            rotation_matrix = np.array(
                [[0, 1, 0], [-1, 0, rotated_height - 1], [0, 0, 1]], dtype=np.float64
            )
        matrix = rotation_matrix @ matrix
    if black_border:
        warped = cv2.copyMakeBorder(
            warped,
            black_border,
            black_border,
            black_border,
            black_border,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        border_matrix = np.array(
            [[1, 0, black_border], [0, 1, black_border], [0, 0, 1]], dtype=np.float64
        )
        matrix = border_matrix @ matrix
    return warped, matrix


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


def _candidate_scores(
    corners: np.ndarray,
    edges: np.ndarray,
    polygon_area: float,
    image_area: float,
    image_center: np.ndarray,
) -> dict[str, float]:
    """计算面积、矩形程度、边缘、中心和边长平衡评分。"""

    rect = cv2.minAreaRect(corners)
    box_area = max(1.0, float(rect[1][0] * rect[1][1]))
    rectangularity = float(np.clip(polygon_area / box_area, 0.0, 1.0))
    area_score = float(np.clip(polygon_area / (image_area * 0.72), 0.0, 1.0))
    center = corners.mean(axis=0)
    diagonal = max(1.0, float(np.hypot(edges.shape[1], edges.shape[0])))
    center_score = float(np.clip(1.0 - np.linalg.norm(center - image_center) / (diagonal * 0.55), 0, 1))
    lengths = np.array(
        [np.linalg.norm(corners[(index + 1) % 4] - corners[index]) for index in range(4)]
    )
    opposite = min(lengths[0], lengths[2]) / max(1.0, max(lengths[0], lengths[2]))
    opposite *= min(lengths[1], lengths[3]) / max(1.0, max(lengths[1], lengths[3]))
    mask = np.zeros_like(edges)
    cv2.polylines(mask, [corners.astype(np.int32)], True, 255, thickness=3)
    edge_strength = float(edges[mask > 0].mean() / 255.0) if np.any(mask) else 0.0
    return {
        "area": area_score,
        "rectangularity": rectangularity,
        "edge_strength": edge_strength,
        "center": center_score,
        "side_balance": float(np.clip(opposite, 0.0, 1.0)),
    }


def _profile_boundary_candidate(gray: np.ndarray, edges: np.ndarray) -> np.ndarray | None:
    """在闭合轮廓缺失时，以行列亮度跃迁估计贴边矩形屏幕。"""

    height, width = gray.shape
    if min(height, width) < 40:
        return None
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 4.0)
    row_profile = blurred.mean(axis=1)
    column_profile = blurred.mean(axis=0)
    row_derivative = np.gradient(row_profile)
    column_derivative = np.gradient(column_profile)

    top_range = slice(max(1, round(height * 0.07)), max(2, round(height * 0.7)))
    top = int(np.argmax(row_derivative[top_range]) + (top_range.start or 0))
    bottom_start = max(top + round(height * 0.2), round(height * 0.35))
    bottom_end = max(bottom_start + 1, round(height * 0.96))
    if bottom_start >= bottom_end:
        return None
    bottom = int(np.argmin(row_derivative[bottom_start:bottom_end]) + bottom_start)
    row_noise = max(0.5, float(np.std(row_derivative)))
    if row_derivative[top] < row_noise * 2.5 or -row_derivative[bottom] < row_noise * 2.0:
        return None

    left_end = max(2, round(width * 0.42))
    left = int(np.argmax(column_derivative[1:left_end]) + 1)
    right_start = round(width * 0.58)
    right_end = max(right_start + 1, round(width * 0.98))
    right = int(np.argmin(column_derivative[right_start:right_end]) + right_start)
    column_noise = max(0.5, float(np.std(column_derivative)))
    if column_derivative[left] < column_noise * 2.0:
        left = 0
    if -column_derivative[right] < column_noise * 2.0:
        right = width - 1
    if right - left < width * 0.35 or bottom - top < height * 0.2:
        return None
    corners = _refine_profile_rectangle(gray, left, top, right, bottom)
    # 至少两条边需要有真实 Canny 支持，避免把普通明暗渐变误判成屏幕。
    support_mask = np.zeros_like(edges)
    cv2.polylines(support_mask, [corners.astype(np.int32)], True, 255, thickness=5)
    edge_support = float(edges[support_mask > 0].mean() / 255.0)
    if edge_support < 0.045:
        return None
    return corners


def _refine_profile_rectangle(
    gray: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> np.ndarray:
    """沿初始矩形法线采样梯度，以稳健直线交点细化四角。

    行列平均只能得到轴对齐包围框，会遗漏影院银幕常见的数像素倾斜。这里对
    每条边逐列/逐行找最强跃迁，使用迭代 MAD 剔除内容纹理，再以四条直线的
    交点恢复角点。贴住照片边界且缺少梯度的一侧会保留原边界。
    """

    height, width = gray.shape
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.4)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    radius = max(5, round(min(height, width) * 0.025))
    top_line = _fit_horizontal_edge(gradient_y, top, 1.0, left, right, radius)
    bottom_line = _fit_horizontal_edge(gradient_y, bottom, -1.0, left, right, radius)
    left_line = _fit_vertical_edge(gradient_x, left, 1.0, top, bottom, radius)
    right_line = _fit_vertical_edge(gradient_x, right, -1.0, top, bottom, radius)
    fallback_lines = (
        np.array([0.0, 1.0, -top]),
        np.array([1.0, 0.0, -right]),
        np.array([0.0, 1.0, -bottom]),
        np.array([1.0, 0.0, -left]),
    )
    top_line = fallback_lines[0] if top_line is None else top_line
    right_line = fallback_lines[1] if right_line is None else right_line
    bottom_line = fallback_lines[2] if bottom_line is None else bottom_line
    left_line = fallback_lines[3] if left_line is None else left_line
    intersections = np.array(
        [
            _line_intersection(top_line, left_line),
            _line_intersection(top_line, right_line),
            _line_intersection(bottom_line, right_line),
            _line_intersection(bottom_line, left_line),
        ],
        dtype=np.float32,
    )
    intersections[:, 0] = np.clip(intersections[:, 0], 0, width - 1)
    intersections[:, 1] = np.clip(intersections[:, 1], 0, height - 1)
    try:
        return order_corners(intersections)
    except ValueError:
        return np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )


def _fit_horizontal_edge(
    gradient_y: np.ndarray,
    center: int,
    sign: float,
    start_x: int,
    end_x: int,
    radius: int,
) -> np.ndarray | None:
    """拟合 ``y=m*x+b``，返回齐次直线 ``[-m,1,-b]``。"""

    start_x = max(0, start_x)
    end_x = min(gradient_y.shape[1] - 1, end_x)
    start_y = max(0, center - radius)
    end_y = min(gradient_y.shape[0], center + radius + 1)
    if end_x - start_x < 24 or end_y - start_y < 3:
        return None
    coordinates = np.arange(start_x, end_x + 1, dtype=np.float64)
    band = sign * gradient_y[start_y:end_y, start_x : end_x + 1]
    positions = np.argmax(band, axis=0).astype(np.float64) + start_y
    strengths = np.max(band, axis=0).astype(np.float64)
    coefficients = _robust_weighted_line(coordinates, positions, strengths)
    if coefficients is None:
        return None
    slope, intercept = coefficients
    return np.array([-slope, 1.0, -intercept], dtype=np.float64)


def _fit_vertical_edge(
    gradient_x: np.ndarray,
    center: int,
    sign: float,
    start_y: int,
    end_y: int,
    radius: int,
) -> np.ndarray | None:
    """拟合 ``x=m*y+b``，返回齐次直线 ``[1,-m,-b]``。"""

    start_y = max(0, start_y)
    end_y = min(gradient_x.shape[0] - 1, end_y)
    start_x = max(0, center - radius)
    end_x = min(gradient_x.shape[1], center + radius + 1)
    if end_y - start_y < 24 or end_x - start_x < 3:
        return None
    coordinates = np.arange(start_y, end_y + 1, dtype=np.float64)
    band = sign * gradient_x[start_y : end_y + 1, start_x:end_x]
    positions = np.argmax(band, axis=1).astype(np.float64) + start_x
    strengths = np.max(band, axis=1).astype(np.float64)
    coefficients = _robust_weighted_line(coordinates, positions, strengths)
    if coefficients is None:
        return None
    slope, intercept = coefficients
    return np.array([1.0, -slope, -intercept], dtype=np.float64)


def _robust_weighted_line(
    coordinates: np.ndarray,
    positions: np.ndarray,
    strengths: np.ndarray,
) -> tuple[float, float] | None:
    """按梯度加权拟合直线，并以 MAD 迭代排除内容边缘。"""

    finite = np.isfinite(strengths) & (strengths > 0)
    if np.count_nonzero(finite) < 24:
        return None
    global_threshold = max(12.0, float(np.percentile(np.abs(strengths[finite]), 35)))
    keep = finite & (strengths >= global_threshold)
    if np.count_nonzero(keep) < 24 or float(np.median(strengths[keep])) < 16.0:
        return None
    coefficients = np.array([0.0, float(np.median(positions[keep]))])
    for _ in range(3):
        coefficients = np.polyfit(
            coordinates[keep],
            positions[keep],
            1,
            w=np.sqrt(np.maximum(strengths[keep], 1.0)),
        )
        residual = positions - np.polyval(coefficients, coordinates)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center))) + 0.5
        keep &= np.abs(residual - center) <= max(2.5, 3.0 * mad)
        if np.count_nonzero(keep) < 24:
            return None
    return float(coefficients[0]), float(coefficients[1])


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """计算两条齐次直线交点；近似平行时返回有限退化点。"""

    point = np.cross(first, second)
    if abs(point[2]) < 1e-8:
        return np.array([0.0, 0.0], dtype=np.float64)
    return point[:2] / point[2]


def _duplicates_existing(
    candidate: QuadrilateralCandidate,
    existing: list[QuadrilateralCandidate],
    threshold: float,
) -> bool:
    return any(
        float(np.mean(np.linalg.norm(candidate.corners - item.corners, axis=1))) < threshold
        for item in existing
    )


def _resolve_ratio(
    mode: AspectRatioMode,
    estimated: float,
    custom: float,
    *,
    rectified_ratio: float | None = None,
) -> float | None:
    fixed = {
        AspectRatioMode.RATIO_16_9: 16 / 9,
        AspectRatioMode.RATIO_16_10: 16 / 10,
        AspectRatioMode.RATIO_4_3: 4 / 3,
        AspectRatioMode.RATIO_1_85: 1.85,
        AspectRatioMode.RATIO_2_39: 2.39,
        AspectRatioMode.CUSTOM: custom,
    }
    if mode in (AspectRatioMode.FREE, AspectRatioMode.ESTIMATED):
        return None
    if mode == AspectRatioMode.AUTO:
        # 同时考虑纵向显示/纸张；sqrt(2) 对应常见 A 系列纸张。
        landscape = (16 / 9, 16 / 10, 4 / 3, 1.85, 2.39, np.sqrt(2.0))
        common = landscape + tuple(1.0 / value for value in landscape)
        candidate = rectified_ratio if rectified_ratio is not None else estimated
        nearest = min(common, key=lambda value: abs(np.log(candidate / value)))
        tolerance = 0.12 if rectified_ratio is not None else 0.08
        return nearest if abs(candidate / nearest - 1.0) <= tolerance else candidate if rectified_ratio is not None else None
    target = fixed[mode]
    # “16:9/4:3”等表示纸面固有比例；纵向四角应自动使用其倒数，避免强制横置。
    if estimated < 1.0 < target:
        return 1.0 / target
    return target


def estimate_rectified_aspect_ratio(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
) -> float | None:
    """在弱相机标定假设下，从矩形四角估计其真实宽高比。

    数学依据：矩形的横纵消失方向正交，令主点为图像中心，可由
    ``f² = -(vx-c)·(vy-c)`` 求焦距。随后将单应矩阵前两列乘 ``K^-1``，
    两列范数之比即为矩形的宽高比。平行边、错误四角和广角畸变会使该估计
    不可靠，因此函数只返回通过有限范围与重投影一致性检查的结果。
    """

    if len(image_shape) < 2 or min(int(image_shape[0]), int(image_shape[1])) < 2:
        return None
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    ordered = order_corners(corners)
    homography = cv2.getPerspectiveTransform(source, ordered).astype(np.float64)
    # 必须复制：消失点归一化不能改写后续用于尺度估计的单应矩阵列。
    horizontal = homography[:, 0].copy()
    vertical = homography[:, 1].copy()
    # 至少一个方向近似平行时，有限消失点约束不足，使用边长回退更稳定。
    if abs(horizontal[2]) < 1e-7 or abs(vertical[2]) < 1e-7:
        return None
    horizontal /= horizontal[2]
    vertical /= vertical[2]
    center_x = (float(image_shape[1]) - 1.0) / 2.0
    center_y = (float(image_shape[0]) - 1.0) / 2.0
    focal_squared = -(
        (horizontal[0] - center_x) * (vertical[0] - center_x)
        + (horizontal[1] - center_y) * (vertical[1] - center_y)
    )
    longest_edge = float(max(image_shape[0], image_shape[1]))
    if not np.isfinite(focal_squared) or not (
        (longest_edge * 0.12) ** 2 <= focal_squared <= (longest_edge * 20.0) ** 2
    ):
        return None
    focal = float(np.sqrt(focal_squared))
    inverse_intrinsics = np.array(
        [
            [1.0 / focal, 0.0, -center_x / focal],
            [0.0, 1.0 / focal, -center_y / focal],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    width_vector = inverse_intrinsics @ homography[:, 0]
    height_vector = inverse_intrinsics @ homography[:, 1]
    ratio = float(np.linalg.norm(width_vector) / max(np.linalg.norm(height_vector), 1e-12))
    if not np.isfinite(ratio) or not 0.1 <= ratio <= 10.0:
        return None
    return ratio


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError("几何模块需要 H×W×3 RGB uint8/float32 图像")
