"""参考图配对定位与客观指标测试。"""

from __future__ import annotations

import cv2
import numpy as np

from screenrestore.validation import (
    compare_images,
    extract_reference_region,
    register_reference,
)


def _textured_reference() -> np.ndarray:
    """生成带唯一局部纹理的合成参考图。"""

    random = np.random.default_rng(42)
    image = np.full((420, 620, 3), 224, np.uint8)
    for index in range(80):
        center = tuple(random.integers([15, 15], [605, 405]).tolist())
        radius = int(random.integers(3, 16))
        color = tuple(int(value) for value in random.integers(0, 220, size=3))
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index % 10),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def test_reference_registration_recovers_nested_content_corners() -> None:
    reference = _textured_reference()
    photo = np.full((720, 980, 3), 35, np.uint8)
    expected = np.array(
        [[170, 120], [830, 155], [870, 625], [125, 650]],
        dtype=np.float32,
    )
    source = np.array(
        [[0, 0], [619, 0], [619, 419], [0, 419]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, expected)
    rendered = cv2.warpPerspective(reference, matrix, (980, 720))
    mask = cv2.warpPerspective(np.full(reference.shape[:2], 255, np.uint8), matrix, (980, 720))
    photo[mask > 0] = rendered[mask > 0]

    registration = register_reference(photo, reference, max_edge=1000)
    error = np.linalg.norm(registration.corners_photo - expected, axis=1)
    assert float(error.mean()) < 2.0
    assert registration.inliers >= 30
    extracted = extract_reference_region(photo, registration, reference.shape)
    metrics = compare_images(extracted, reference)
    assert metrics["luminance_ssim"] > 0.96
    assert metrics["gradient_correlation"] > 0.94


def test_comparison_metrics_identical_image_are_ideal() -> None:
    reference = _textured_reference()
    metrics = compare_images(reference.copy(), reference)
    assert metrics["mae_255"] == 0.0
    assert metrics["luminance_ssim"] > 0.9999
    assert metrics["gradient_correlation"] > 0.9999
    assert abs(metrics["sharpness_ratio"] - 1.0) < 1e-6
