"""几何校正的数值和合成场景测试。"""

from __future__ import annotations

import cv2
import numpy as np
from tests.synthetic.generators import checkerboard, framed_screen

from screenrestore.geometry import (
    AspectRatioMode,
    ClassicQuadDetector,
    InterpolationMode,
    estimate_output_size,
    estimate_rectified_aspect_ratio,
    order_corners,
    warp_perspective,
)


def test_order_corners_is_stable_for_permutations() -> None:
    expected = np.array([[10, 20], [110, 10], [120, 80], [5, 90]], dtype=np.float32)
    for permutation in ([2, 0, 3, 1], [1, 3, 0, 2], [3, 2, 1, 0]):
        assert np.allclose(order_corners(expected[permutation]), expected)


def test_estimate_output_size_supports_fixed_cinema_ratio() -> None:
    corners = np.array([[0, 0], [400, 0], [400, 200], [0, 200]], np.float32)
    width, height = estimate_output_size(corners, AspectRatioMode.RATIO_2_39)
    assert abs(width / height - 2.39) < 0.01
    assert abs(width * height - 80_000) / 80_000 < 0.02


def test_fixed_ratio_preserves_portrait_orientation() -> None:
    corners = np.array([[20, 10], [180, 30], [210, 330], [0, 300]], np.float32)
    width, height = estimate_output_size(corners, AspectRatioMode.RATIO_4_3)
    assert height > width
    assert abs(width / height - 3 / 4) < 0.01


def test_metric_rectification_recovers_portrait_paper_ratio() -> None:
    """用合成针孔相机投影验证 AUTO 不会采用失真的投影边长比。"""

    image_shape = (1200, 1600, 3)
    physical_ratio = 1 / np.sqrt(2.0)
    object_points = np.array(
        [[0, 0, 0], [physical_ratio, 0, 0], [physical_ratio, 1, 0], [0, 1, 0]],
        np.float64,
    )
    rotation_vector = np.array([0.42, -0.58, 0.13], np.float64)
    translation = np.array([0.05, -0.08, 2.3], np.float64)
    intrinsics = np.array([[1100, 0, 800], [0, 1100, 600], [0, 0, 1]], np.float64)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        intrinsics,
        np.zeros(5),
    )
    corners = projected.reshape(4, 2).astype(np.float32)
    ratio = estimate_rectified_aspect_ratio(corners, image_shape)
    assert ratio is not None
    assert abs(ratio - physical_ratio) < 0.035
    width, height = estimate_output_size(
        corners,
        AspectRatioMode.AUTO,
        image_shape=image_shape,
    )
    assert abs(width / height - physical_ratio) < 0.01


def test_perspective_recovery_reduces_known_homography_error() -> None:
    clean = checkerboard()
    source = np.array([[0, 0], [319, 0], [319, 179], [0, 179]], np.float32)
    target = np.array([[52, 38], [382, 20], [360, 242], [34, 258]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    degraded = cv2.warpPerspective(clean, matrix, (430, 280))
    recovered, recovery_matrix = warp_perspective(
        degraded,
        target,
        AspectRatioMode.RATIO_16_9,
        interpolation=InterpolationMode.CUBIC,
    )
    resized = cv2.resize(recovered, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
    mae = float(np.mean(np.abs(resized.astype(np.float32) - clean.astype(np.float32))))
    combined = recovery_matrix @ matrix
    combined /= combined[2, 2]
    assert mae < 24.0
    assert np.allclose(combined[2, :2], 0, atol=0.02)


def test_detector_returns_candidate_near_artificial_screen() -> None:
    image, expected = framed_screen()
    candidates = ClassicQuadDetector().predict(image).candidates
    assert candidates
    error = min(
        float(np.mean(np.linalg.norm(candidate.corners - expected, axis=1)))
        for candidate in candidates
    )
    assert error < 15.0


def test_detector_handles_screen_clipped_by_image_edge() -> None:
    image = np.full((360, 600, 3), 18, np.uint8)
    image[92:310, 16:] = (145, 155, 165)
    cv2.line(image, (16, 92), (599, 92), (230, 230, 230), 3)
    cv2.line(image, (16, 92), (16, 309), (230, 230, 230), 3)
    cv2.line(image, (16, 309), (599, 309), (55, 55, 55), 2)
    candidates = ClassicQuadDetector().predict(image).candidates
    assert candidates
    expected = np.array([[16, 92], [599, 92], [599, 309], [16, 309]], np.float32)
    error = float(np.mean(np.linalg.norm(candidates[0].corners - expected, axis=1)))
    assert error < 12.0


def test_detector_refines_slanted_edges_when_screen_is_clipped() -> None:
    image = np.full((380, 620, 3), 16, np.uint8)
    expected = np.array([[28, 82], [619, 94], [619, 318], [11, 334]], np.float32)
    cv2.fillConvexPoly(image, expected.astype(np.int32), (150, 160, 170))
    cv2.line(image, tuple(expected[0].astype(int)), tuple(expected[1].astype(int)), (240,) * 3, 3)
    cv2.line(image, tuple(expected[0].astype(int)), tuple(expected[3].astype(int)), (240,) * 3, 3)
    cv2.line(image, tuple(expected[3].astype(int)), tuple(expected[2].astype(int)), (45,) * 3, 2)
    candidates = ClassicQuadDetector().predict(image).candidates
    assert candidates
    error = float(np.mean(np.linalg.norm(candidates[0].corners - expected, axis=1)))
    assert error < 11.0
