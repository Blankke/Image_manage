"""无模型环境可运行的传统四边形候选基线。"""

from __future__ import annotations

import cv2
import numpy as np

from .edge_refine import EdgeRefineParameters, refine_quad_edges
from .rectify import order_corners
from .types import QuadrilateralCandidate, TargetLayer


def detect_classic_candidates(
    image_rgb: np.ndarray,
    max_candidates: int = 8,
    detection_max_edge: int = 1200,
) -> list[QuadrilateralCandidate]:
    """生成传统几何候选，供 fallback 和基线评估使用。

    该函数只回答“哪些边像四边形”，不声称理解画芯、卡纸、外框之间的语义层级。
    因此它的输出必须继续经过统一置信度策略，不能直接采用第一项归档。
    """

    _validate_rgb(image_rgb)
    if max_candidates < 1:
        raise ValueError("max_candidates 必须大于 0")
    height, width = image_rgb.shape[:2]
    scale = min(1.0, detection_max_edge / max(height, width))
    if scale < 1.0:
        small_rgb = cv2.resize(
            image_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_rgb = image_rgb.copy()
    gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 7, 45, 45)
    image_area = float(gray.shape[0] * gray.shape[1])
    image_center = np.array([gray.shape[1] / 2, gray.shape[0] / 2], dtype=np.float32)
    candidates: list[QuadrilateralCandidate] = []

    for lower, upper in _canny_thresholds(filtered):
        edges = cv2.Canny(filtered, lower, upper)
        kernel_size = max(3, int(round(min(small_rgb.shape[:2]) * 0.008)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.rectangle(closed, (0, 0), (closed.shape[1] - 1, closed.shape[0] - 1), 255, 2)
        contours, _hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < image_area * 0.025:
                break
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon_ratio in (0.012, 0.018, 0.025, 0.035):
                polygon = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                corners_small = order_corners(polygon.reshape(4, 2))
                polygon_area = abs(float(cv2.contourArea(corners_small)))
                if not image_area * 0.025 <= polygon_area <= image_area * 0.985:
                    continue
                scores = _candidate_scores(
                    corners_small,
                    edges,
                    polygon_area,
                    image_area,
                    image_center,
                )
                # 人工补入的照片边界只用于闭合贴边目标，本身不能成为候选。
                if polygon_area > image_area * 0.94 and scores["edge_strength"] < 0.05:
                    continue
                confidence = _combined_score(scores)
                candidate = QuadrilateralCandidate(
                    corners=corners_small / scale,
                    confidence=confidence,
                    scores=scores,
                    source="classic_contour",
                    layer=TargetLayer.UNKNOWN,
                )
                if not _duplicates_existing(candidate, candidates, max(height, width) * 0.018):
                    candidates.append(candidate)
                break

        # 轮廓断裂时保留 Hough 候选回路，但依旧只给传统基线置信度。
        for corners_small in _hough_candidates(edges):
            polygon_area = abs(float(cv2.contourArea(corners_small)))
            if not image_area * 0.025 <= polygon_area <= image_area * 0.985:
                continue
            scores = _candidate_scores(
                corners_small,
                edges,
                polygon_area,
                image_area,
                image_center,
            )
            confidence = min(0.82, _combined_score(scores) * 0.96)
            candidate = QuadrilateralCandidate(
                corners=corners_small / scale,
                confidence=confidence,
                scores=scores,
                source="classic_hough",
                layer=TargetLayer.UNKNOWN,
            )
            if not _duplicates_existing(candidate, candidates, max(height, width) * 0.018):
                candidates.append(candidate)

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    if not candidates:
        profile = _profile_boundary_candidate(gray)
        if profile is not None:
            refinement = refine_quad_edges(
                small_rgb,
                profile,
                params=EdgeRefineParameters(minimum_support=0.10),
            )
            corners_small = refinement.corners
            edges = cv2.Canny(filtered, *_canny_thresholds(filtered)[0])
            polygon_area = abs(float(cv2.contourArea(corners_small)))
            scores = _candidate_scores(
                corners_small,
                edges,
                polygon_area,
                image_area,
                image_center,
            )
            scores["profile_support"] = refinement.mean_support
            confidence = min(0.68, _combined_score(scores) * 0.90)
            candidates.append(
                QuadrilateralCandidate(
                    corners=corners_small / scale,
                    confidence=confidence,
                    scores=scores,
                    source="classic_profile",
                    layer=TargetLayer.UNKNOWN,
                )
            )
    return candidates[:max_candidates]


def _canny_thresholds(gray: np.ndarray) -> list[tuple[int, int]]:
    median = float(np.median(gray))
    adaptive = (
        int(max(0, 0.66 * median)),
        int(min(255, max(30, 1.33 * median))),
    )
    return [adaptive, (30, 100), (50, 150)]


def _candidate_scores(
    corners: np.ndarray,
    edges: np.ndarray,
    polygon_area: float,
    image_area: float,
    image_center: np.ndarray,
) -> dict[str, float]:
    rect = cv2.minAreaRect(corners)
    box_area = max(1.0, float(rect[1][0] * rect[1][1]))
    rectangularity = float(np.clip(polygon_area / box_area, 0.0, 1.0))
    area_score = float(np.clip(polygon_area / (image_area * 0.72), 0.0, 1.0))
    center = corners.mean(axis=0)
    diagonal = max(1.0, float(np.hypot(edges.shape[1], edges.shape[0])))
    center_score = float(
        np.clip(1.0 - np.linalg.norm(center - image_center) / (diagonal * 0.55), 0, 1)
    )
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


def _combined_score(scores: dict[str, float]) -> float:
    return float(
        np.clip(
            0.28 * scores["area"]
            + 0.18 * scores["rectangularity"]
            + 0.30 * scores["edge_strength"]
            + 0.12 * scores["center"]
            + 0.12 * scores["side_balance"],
            0.0,
            1.0,
        )
    )


def _hough_candidates(edges: np.ndarray) -> list[np.ndarray]:
    height, width = edges.shape
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(40, round(min(height, width) * 0.12)),
        minLineLength=max(24, round(min(height, width) * 0.18)),
        maxLineGap=max(8, round(max(height, width) * 0.035)),
    )
    if lines is None:
        return []
    horizontal: list[tuple[np.ndarray, float]] = []
    vertical: list[tuple[np.ndarray, float]] = []
    # OpenCV 4 返回 N×1×4，OpenCV 5 wheel 可能直接返回 N×4；统一展平处理。
    for raw in np.asarray(lines, dtype=np.float64).reshape(-1, 4):
        x1, y1, x2, y2 = raw
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))) % 180
        angle = min(angle, 180 - angle)
        if angle <= 35:
            horizontal.append((raw, (y1 + y2) / 2))
        elif angle >= 55:
            vertical.append((raw, (x1 + x2) / 2))
    if len(horizontal) < 2 or len(vertical) < 2:
        return []
    horizontal.sort(key=lambda item: item[1])
    vertical.sort(key=lambda item: item[1])
    top_pool = horizontal[: min(3, len(horizontal))]
    bottom_pool = horizontal[-min(3, len(horizontal)) :]
    left_pool = vertical[: min(3, len(vertical))]
    right_pool = vertical[-min(3, len(vertical)) :]
    output: list[np.ndarray] = []
    for top in top_pool:
        for bottom in bottom_pool:
            for left in left_pool:
                for right in right_pool:
                    points = [
                        _segment_line_intersection(top[0], left[0]),
                        _segment_line_intersection(top[0], right[0]),
                        _segment_line_intersection(bottom[0], right[0]),
                        _segment_line_intersection(bottom[0], left[0]),
                    ]
                    if any(point is None for point in points):
                        continue
                    try:
                        corners = order_corners(np.asarray(points, np.float32))
                    except ValueError:
                        continue
                    margin = max(height, width) * 0.05
                    if (
                        np.any(corners[:, 0] < -margin)
                        or np.any(corners[:, 0] > width - 1 + margin)
                        or np.any(corners[:, 1] < -margin)
                        or np.any(corners[:, 1] > height - 1 + margin)
                    ):
                        continue
                    output.append(corners)
                    if len(output) >= 12:
                        return output
    return output


def _segment_line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(float(denominator)) < 1e-9:
        return None
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return np.array([px, py], dtype=np.float32)


def _profile_boundary_candidate(gray: np.ndarray) -> np.ndarray | None:
    height, width = gray.shape
    if min(height, width) < 40:
        return None
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 4.0)
    row_derivative = np.gradient(blurred.mean(axis=1))
    column_derivative = np.gradient(blurred.mean(axis=0))
    top_start, top_end = max(1, round(height * 0.04)), max(2, round(height * 0.7))
    top = int(np.argmax(np.abs(row_derivative[top_start:top_end])) + top_start)
    bottom_start, bottom_end = max(top + round(height * 0.2), round(height * 0.35)), max(
        top + 2, round(height * 0.98)
    )
    if bottom_start >= bottom_end:
        return None
    bottom = int(np.argmax(np.abs(row_derivative[bottom_start:bottom_end])) + bottom_start)
    left_end = max(2, round(width * 0.48))
    right_start = round(width * 0.52)
    left = int(np.argmax(np.abs(column_derivative[1:left_end])) + 1)
    right = int(np.argmax(np.abs(column_derivative[right_start : width - 1])) + right_start)
    row_noise = max(0.5, float(np.std(row_derivative)))
    column_noise = max(0.5, float(np.std(column_derivative)))
    if abs(float(row_derivative[top])) < row_noise * 2.0:
        return None
    if abs(float(row_derivative[bottom])) < row_noise * 1.8:
        return None
    if abs(float(column_derivative[left])) < column_noise * 1.7:
        left = 0
    if abs(float(column_derivative[right])) < column_noise * 1.7:
        right = width - 1
    if right - left < width * 0.35 or bottom - top < height * 0.2:
        return None
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)


def _duplicates_existing(
    candidate: QuadrilateralCandidate,
    existing: list[QuadrilateralCandidate],
    threshold: float,
) -> bool:
    return any(
        float(np.mean(np.linalg.norm(candidate.corners - item.corners, axis=1))) < threshold
        for item in existing
    )


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError("传统检测器需要 H×W×3 RGB uint8/float32 图像")
