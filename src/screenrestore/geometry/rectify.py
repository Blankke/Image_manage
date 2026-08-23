"""平面四边形排序、画幅估计与单应变换。"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

from .types import AspectEstimate


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


def order_corners(corners: np.ndarray) -> np.ndarray:
    """把四点稳定排序为左上、右上、右下、左下。"""

    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("角点包含非有限数值")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[3, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    if abs(float(cv2.contourArea(ordered))) < 1e-3:
        raise ValueError("角点退化，无法构成有效四边形")
    return ordered.astype(np.float32)


def estimate_aspect(corners: np.ndarray, image_shape: tuple[int, ...]) -> AspectEstimate:
    """估计物理画幅，并显式返回证据置信度。

    metric 路径建立在中心主点、零像素倾斜和近似针孔相机假设上。置信度衡量的是
    这组假设的数值稳定性，不能替代一次真实相机标定。
    """

    ordered = order_corners(corners)
    tl, tr, br, bl = ordered
    projected_ratio = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(
        1e-6,
        max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)),
    )
    metric = _metric_aspect_details(ordered, image_shape)
    if metric is None:
        return AspectEstimate(float(projected_ratio), 0.15, "projected_edges")
    ratio, focal, orthogonality_error = metric
    longest = float(max(image_shape[:2]))
    focal_score = float(np.clip(1.0 - abs(np.log(max(focal / longest, 1e-6))) / 3.0, 0, 1))
    orthogonality_score = float(np.clip(1.0 - orthogonality_error / 0.08, 0, 1))
    # 投影边比与 metric 比可以相差很大；这里只排除极端不连续解，不惩罚正常透视。
    ratio_score = float(np.clip(1.0 - abs(np.log(max(ratio, 1e-6))) / np.log(10.0), 0, 1))
    confidence = 0.45 * focal_score + 0.45 * orthogonality_score + 0.10 * ratio_score
    return AspectEstimate(float(ratio), float(np.clip(confidence, 0.2, 0.9)), "metric_rectification")


def estimate_rectified_aspect_ratio(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
) -> float | None:
    """兼容数学调用场景，只返回通过约束的物理宽高比。"""

    details = _metric_aspect_details(corners, image_shape)
    return None if details is None else details[0]


def estimate_output_size(
    corners: np.ndarray,
    ratio_mode: AspectRatioMode = AspectRatioMode.AUTO,
    custom_ratio: float = 16 / 9,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """估计输出尺寸；AUTO 仅在 metric 证据可用时采用其比例。"""

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
    """执行单应校正并返回独立 RGB 输出与 3×3 变换矩阵。"""

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
                matrix = np.array(
                    [[1, 0, -crop_x], [0, 1, -crop_y], [0, 0, 1]],
                    dtype=np.float64,
                ) @ matrix
    if rotation:
        rotate_codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        if rotation not in rotate_codes:
            raise ValueError("旋转角度仅支持 0/90/180/270")
        warped = cv2.rotate(warped, rotate_codes[rotation])
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
        matrix = np.array(
            [[1, 0, black_border], [0, 1, black_border], [0, 0, 1]], dtype=np.float64
        ) @ matrix
    return warped, matrix


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
        landscape = (16 / 9, 16 / 10, 4 / 3, 1.85, 2.39, np.sqrt(2.0))
        common = landscape + tuple(1.0 / value for value in landscape)
        candidate = rectified_ratio if rectified_ratio is not None else estimated
        nearest = min(common, key=lambda value: abs(np.log(candidate / value)))
        tolerance = 0.12 if rectified_ratio is not None else 0.08
        if abs(candidate / nearest - 1.0) <= tolerance:
            return nearest
        return candidate if rectified_ratio is not None else None
    target = fixed[mode]
    return 1.0 / target if estimated < 1.0 < target else target


def _metric_aspect_details(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[float, float, float] | None:
    if len(image_shape) < 2 or min(int(image_shape[0]), int(image_shape[1])) < 2:
        return None
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    ordered = order_corners(corners)
    homography = cv2.getPerspectiveTransform(source, ordered).astype(np.float64)
    horizontal = homography[:, 0].copy()
    vertical = homography[:, 1].copy()
    if abs(horizontal[2]) < 1e-7 or abs(vertical[2]) < 1e-7:
        return None
    horizontal /= horizontal[2]
    vertical /= vertical[2]
    center_x = (float(image_shape[1]) - 1.0) / 2.0
    center_y = (float(image_shape[0]) - 1.0) / 2.0
    dot = (horizontal[0] - center_x) * (vertical[0] - center_x) + (
        horizontal[1] - center_y
    ) * (vertical[1] - center_y)
    focal_squared = -dot
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
    width_norm = float(np.linalg.norm(width_vector))
    height_norm = float(np.linalg.norm(height_vector))
    ratio = width_norm / max(height_norm, 1e-12)
    if not np.isfinite(ratio) or not 0.1 <= ratio <= 10.0:
        return None
    cosine = abs(float(np.dot(width_vector, height_vector))) / max(
        width_norm * height_norm,
        1e-12,
    )
    return float(ratio), focal, cosine


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError("几何模块需要 H×W×3 RGB uint8/float32 图像")
