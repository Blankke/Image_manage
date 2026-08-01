"""镜头标定、畸变和弯曲银幕网格的数值质量测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.operators.lens_distortion import (
    LensCalibrationParameters,
    LensDistortionParameters,
    calibrate_lens,
    camera_matrix_for_shape,
    undistort_lens,
)
from screenrestore.operators.mesh_warp import (
    MeshWarpParameters,
    curved_control_grid,
    regular_control_grid,
    warp_mesh,
)


def _line_chart(width: int = 360, height: int = 240) -> np.ndarray:
    image = np.full((height, width, 3), 24, np.uint8)
    for x in range(20, width, 30):
        cv2.line(image, (x, 0), (x, height - 1), (235, 235, 235), 2)
    for y in range(20, height, 30):
        cv2.line(image, (0, y), (width - 1, y), (210, 210, 210), 2)
    cv2.circle(image, (width // 2, height // 2), 38, (30, 180, 250), 3)
    return image


def test_known_lens_parameters_reduce_radial_distortion_error() -> None:
    clean = _line_chart()
    params = LensDistortionParameters(focal_x=0.92, focal_y=1.38, k1=-0.24, k2=0.055)
    camera = camera_matrix_for_shape(params, clean.shape)
    distortion = np.array([params.k1, params.k2, 0.0, 0.0, 0.0], np.float64)
    height, width = clean.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    distorted_points = np.stack((xx, yy), axis=2).reshape(-1, 1, 2)
    ideal_points = cv2.undistortPoints(
        distorted_points,
        camera,
        distortion,
        P=camera,
    ).reshape(height, width, 2)
    distorted = cv2.remap(
        clean,
        ideal_points[..., 0],
        ideal_points[..., 1],
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )

    corrected, metadata = undistort_lens(distorted, params)
    margin = 24
    region = np.s_[margin:-margin, margin:-margin]
    before = float(np.mean(np.abs(distorted[region].astype(np.float32) - clean[region])))
    after = float(np.mean(np.abs(corrected[region].astype(np.float32) - clean[region])))

    assert after < before * 0.42
    assert metadata["model"] == "pinhole"


def test_curved_mesh_restores_synthetic_horizontal_bow() -> None:
    clean = _line_chart(350, 230)
    height, width = clean.shape[:2]
    curve = 0.085
    yy, xx = np.indices((height, width), dtype=np.float32)
    normalized_x = xx / (width - 1)
    normalized_y = yy / (height - 1)
    unit_x = normalized_x * 2.0 - 1.0
    # 求 ``source_y = ideal_y + a*u*(1-u²)`` 的逆函数，以合成已弯曲输入。
    ideal_y = normalized_y.copy()
    amplitude = curve * (1.0 - np.square(unit_x))
    for _ in range(7):
        unit_ideal_y = ideal_y * 2.0 - 1.0
        mapped = ideal_y + amplitude * unit_ideal_y * (1.0 - np.square(unit_ideal_y))
        derivative = 1.0 + 2.0 * amplitude * (1.0 - 3.0 * np.square(unit_ideal_y))
        ideal_y -= (mapped - normalized_y) / np.maximum(derivative, 0.2)
    distorted = cv2.remap(
        clean,
        xx,
        np.clip(ideal_y * (height - 1), 0, height - 1),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    grid = curved_control_grid(5, 7, horizontal_curve=curve)
    restored, diagnostics = warp_mesh(
        distorted,
        MeshWarpParameters(rows=5, columns=7, control_points=grid.tolist()),
    )

    before = float(np.mean(np.abs(distorted.astype(np.float32) - clean)))
    after = float(np.mean(np.abs(restored.astype(np.float32) - clean)))
    assert after < before * 0.38
    assert diagnostics["max_normalized_displacement"] > 0.02


def test_identity_mesh_is_pixel_stable() -> None:
    image = _line_chart(180, 120)
    grid = regular_control_grid(4, 6)
    restored, _ = warp_mesh(
        image,
        MeshWarpParameters(rows=4, columns=6, control_points=grid.tolist()),
    )
    assert float(np.mean(np.abs(restored.astype(np.int16) - image.astype(np.int16)))) < 0.02


def test_checkerboard_calibration_uses_multiple_detected_views() -> None:
    columns, rows = 7, 5
    square = 52
    board = np.full(((rows + 1) * square, (columns + 1) * square, 3), 255, np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                cv2.rectangle(
                    board,
                    (column * square, row * square),
                    ((column + 1) * square, (row + 1) * square),
                    (0, 0, 0),
                    -1,
                )
    source = np.array(
        [[0, 0], [board.shape[1] - 1, 0], [board.shape[1] - 1, board.shape[0] - 1], [0, board.shape[0] - 1]],
        np.float32,
    )
    camera = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])
    outer_board = np.array([[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0]], np.float32)
    poses = [
        ([0.08, -0.12, 0.02], [-4.0, -3.0, 12.0]),
        ([-0.14, 0.08, -0.04], [-3.8, -3.1, 12.5]),
        ([0.16, 0.13, 0.03], [-4.2, -2.9, 13.0]),
        ([-0.1, -0.16, 0.06], [-4.1, -3.2, 12.2]),
        ([0.05, 0.18, -0.08], [-3.9, -2.8, 12.8]),
    ]
    views = []
    for rotation, translation in poses:
        projected, _ = cv2.projectPoints(
            outer_board,
            np.asarray(rotation, np.float64),
            np.asarray(translation, np.float64),
            camera,
            np.zeros(5),
        )
        matrix = cv2.getPerspectiveTransform(source, projected.reshape(4, 2).astype(np.float32))
        views.append(
            cv2.warpPerspective(
                board,
                matrix,
                (640, 480),
                borderValue=(210, 210, 210),
            )
        )
    result = calibrate_lens(
        views,
        LensCalibrationParameters(
            board_columns=columns,
            board_rows=rows,
            min_views=5,
        ),
    )
    assert result.used_views == 5
    assert np.isfinite(result.rms_error)
    assert result.rms_error < 1.5
    assert len(result.per_view_errors) == 5
