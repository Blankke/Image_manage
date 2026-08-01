"""用已知参考图定位实拍内容并进行客观对比。

这里的参考图只服务于测试集验收，不属于恢复流水线输入。这样既能得到可靠的内容
四角，又不会把原图像素泄漏到恢复结果中。实际用户没有参考图时仍使用自动检测或
手工四角。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from screenrestore.operators.geometry import (
    AspectRatioMode,
    InterpolationMode,
    order_corners,
    warp_perspective,
)


@dataclass(frozen=True, slots=True)
class ReferenceRegistration:
    """参考图到实拍图的稳健特征配准结果。"""

    homography_reference_to_photo: np.ndarray
    corners_photo: np.ndarray
    feature_matches: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error_px: float
    p95_reprojection_error_px: float

    def to_dict(self) -> dict[str, object]:
        """返回不包含图像和特征描述子的有限 JSON 诊断。"""

        return {
            "feature_matches": self.feature_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 6),
            "median_reprojection_error_px": round(
                self.median_reprojection_error_px, 6
            ),
            "p95_reprojection_error_px": round(self.p95_reprojection_error_px, 6),
            "corners_photo": np.round(self.corners_photo, 3).tolist(),
        }


def register_reference(
    photo_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    max_edge: int = 1400,
    ratio_threshold: float = 0.76,
) -> ReferenceRegistration:
    """以 SIFT+MAGSAC 求参考图到实拍内容的单应性。

    特征只用于定位测试图中的目标内容。函数会拒绝匹配数不足、外推过远或退化的
    单应矩阵，避免在评分阶段悄悄接受错误选区。
    """

    _require_rgb(photo_rgb, "实拍图")
    _require_rgb(reference_rgb, "参考图")
    if max_edge < 256:
        raise ValueError("特征配准 max_edge 不能小于 256")
    if not 0.5 <= ratio_threshold <= 0.95:
        raise ValueError("Lowe ratio 阈值必须位于 0.5..0.95")

    reference_proxy, reference_scale = _proxy(reference_rgb, max_edge)
    photo_proxy, photo_scale = _proxy(photo_rgb, max_edge)
    reference_gray = _feature_gray(reference_proxy)
    photo_gray = _feature_gray(photo_proxy)
    sift = cv2.SIFT_create(
        nfeatures=10_000,
        contrastThreshold=0.015,
        edgeThreshold=15,
    )
    reference_keys, reference_descriptors = sift.detectAndCompute(reference_gray, None)
    photo_keys, photo_descriptors = sift.detectAndCompute(photo_gray, None)
    if reference_descriptors is None or photo_descriptors is None:
        raise ValueError("参考配准失败：图像中没有足够稳定的局部特征")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        reference_descriptors,
        photo_descriptors,
        k=2,
    )
    good_matches = [
        first
        for first, second in matches
        if first.distance < ratio_threshold * second.distance
    ]
    if len(good_matches) < 12:
        raise ValueError(f"参考配准失败：可靠特征匹配仅 {len(good_matches)} 对")

    reference_points = np.float32(
        [reference_keys[item.queryIdx].pt for item in good_matches]
    )
    photo_points = np.float32([photo_keys[item.trainIdx].pt for item in good_matches])
    robust_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography_proxy, inlier_mask = cv2.findHomography(
        reference_points,
        photo_points,
        robust_method,
        3.0,
    )
    if homography_proxy is None or inlier_mask is None:
        raise ValueError("参考配准失败：无法估计稳定单应矩阵")
    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count < 10 or inlier_count / len(good_matches) < 0.2:
        raise ValueError("参考配准失败：单应矩阵内点不足")

    # 小图 H 的两侧分别还原到参考图和实拍图原尺寸坐标。
    reference_to_proxy = np.diag([reference_scale, reference_scale, 1.0])
    photo_from_proxy = np.diag([1.0 / photo_scale, 1.0 / photo_scale, 1.0])
    homography = photo_from_proxy @ homography_proxy @ reference_to_proxy
    homography /= homography[2, 2]
    reference_height, reference_width = reference_rgb.shape[:2]
    reference_corners = np.array(
        [
            [0.0, 0.0],
            [reference_width - 1.0, 0.0],
            [reference_width - 1.0, reference_height - 1.0],
            [0.0, reference_height - 1.0],
        ],
        dtype=np.float32,
    )
    corners = cv2.perspectiveTransform(reference_corners[None], homography)[0]
    corners = order_corners(corners)
    _validate_projected_corners(corners, photo_rgb.shape)

    predicted = cv2.perspectiveTransform(reference_points[None], homography_proxy)[0]
    errors = np.linalg.norm(predicted - photo_points, axis=1)[inlier_mask] / photo_scale
    return ReferenceRegistration(
        homography_reference_to_photo=homography.astype(np.float64),
        corners_photo=corners,
        feature_matches=len(good_matches),
        inliers=inlier_count,
        inlier_ratio=float(inlier_count / len(good_matches)),
        median_reprojection_error_px=float(np.median(errors)),
        p95_reprojection_error_px=float(np.percentile(errors, 95)),
    )


def extract_reference_region(
    photo_rgb: np.ndarray,
    registration: ReferenceRegistration,
    reference_shape: tuple[int, ...],
) -> np.ndarray:
    """只用参考图定位出的四角进行透视恢复，不读取参考图像素。"""

    _require_rgb(photo_rgb, "实拍图")
    reference_height, reference_width = reference_shape[:2]
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("参考尺寸无效")
    restored, _ = warp_perspective(
        photo_rgb,
        registration.corners_photo,
        ratio_mode=AspectRatioMode.CUSTOM,
        custom_ratio=reference_width / reference_height,
        interpolation=InterpolationMode.LANCZOS,
        auto_crop=False,
    )
    return restored


def align_for_comparison(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """把恢复结果的小量残余几何误差配准到参考图坐标，供分项评分。

    直接缩放指标仍应同时报告；这个二次配准指标用于区分“选区误差”和“恢复算法
    误差”，不能替代几何验收。
    """

    registration = register_reference(candidate_rgb, reference_rgb, max_edge=1200)
    inverse = np.linalg.inv(registration.homography_reference_to_photo)
    height, width = reference_rgb.shape[:2]
    aligned = cv2.warpPerspective(
        candidate_rgb,
        inverse,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.ascontiguousarray(aligned), registration.to_dict()


def compare_images(candidate_rgb: np.ndarray, reference_rgb: np.ndarray) -> dict[str, float]:
    """计算同尺寸下的色彩、结构、清晰度与周期伪影指标。"""

    _require_rgb(candidate_rgb, "候选图")
    _require_rgb(reference_rgb, "参考图")
    height, width = reference_rgb.shape[:2]
    if candidate_rgb.shape[:2] != (height, width):
        candidate_rgb = cv2.resize(
            candidate_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA
            if candidate_rgb.shape[0] >= height and candidate_rgb.shape[1] >= width
            else cv2.INTER_LANCZOS4,
        )
    candidate = _as_float(candidate_rgb)
    reference = _as_float(reference_rgb)
    difference = candidate - reference
    mse = float(np.mean(np.square(difference)))
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    candidate_gradient = _gradient_magnitude(candidate_gray)
    reference_gradient = _gradient_magnitude(reference_gray)
    gradient_correlation = _safe_correlation(candidate_gradient, reference_gradient)

    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_RGB2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    delta_e = np.linalg.norm(candidate_lab - reference_lab, axis=2)
    candidate_sharpness = float(cv2.Laplacian(candidate_gray, cv2.CV_32F).var())
    reference_sharpness = float(cv2.Laplacian(reference_gray, cv2.CV_32F).var())
    candidate_periodicity = _spectral_peak_ratio(candidate_gray)
    reference_periodicity = _spectral_peak_ratio(reference_gray)
    return {
        "mae_255": round(float(np.mean(np.abs(difference))) * 255.0, 6),
        "psnr_db": round(float(10.0 * np.log10(1.0 / max(mse, 1e-12))), 6),
        "luminance_ssim": round(_ssim(candidate_gray, reference_gray), 6),
        "gradient_correlation": round(gradient_correlation, 6),
        "delta_e_mean": round(float(delta_e.mean()), 6),
        "delta_e_p95": round(float(np.percentile(delta_e, 95)), 6),
        "luminance_bias": round(float(candidate_gray.mean() - reference_gray.mean()), 6),
        "contrast_ratio": round(
            float(candidate_gray.std() / max(reference_gray.std(), 1e-6)),
            6,
        ),
        "sharpness_ratio": round(candidate_sharpness / max(reference_sharpness, 1e-8), 6),
        "spectral_peak_ratio": round(candidate_periodicity, 6),
        "spectral_peak_excess_db": round(
            float(10.0 * np.log10(candidate_periodicity / max(reference_periodicity, 1e-8))),
            6,
        ),
        "black_clipping_ratio": round(float(np.mean(candidate_gray <= 2 / 255)), 6),
        "white_clipping_ratio": round(float(np.mean(candidate_gray >= 253 / 255)), 6),
    }


def difference_heatmap(candidate_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """把 CIELAB 色差映射为带颜色的相对热度图。"""

    _require_rgb(candidate_rgb, "候选图")
    _require_rgb(reference_rgb, "参考图")
    height, width = reference_rgb.shape[:2]
    if candidate_rgb.shape[:2] != (height, width):
        candidate_rgb = cv2.resize(
            candidate_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    candidate = _as_float(candidate_rgb)
    reference = _as_float(reference_rgb)
    delta_e = np.linalg.norm(
        cv2.cvtColor(candidate, cv2.COLOR_RGB2LAB)
        - cv2.cvtColor(reference, cv2.COLOR_RGB2LAB),
        axis=2,
    )
    scale = max(1.0, float(np.percentile(delta_e, 98)))
    normalized = np.clip(delta_e / scale * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _feature_gray(image_rgb: np.ndarray) -> np.ndarray:
    """局部均衡亮度以抵抗屏摄曝光和色温变化。"""

    gray = cv2.cvtColor(_as_uint8(image_rgb), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)


def _proxy(image_rgb: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale >= 1.0:
        return image_rgb, 1.0
    return (
        cv2.resize(
            image_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def _validate_projected_corners(corners: np.ndarray, photo_shape: tuple[int, ...]) -> None:
    height, width = photo_shape[:2]
    area = abs(float(cv2.contourArea(corners)))
    if area < width * height * 0.015:
        raise ValueError("参考配准失败：目标内容投影面积过小")
    margin = np.array([width, height], dtype=np.float32) * 0.3
    if np.any(corners < -margin) or np.any(
        corners > np.array([width - 1, height - 1], dtype=np.float32) + margin
    ):
        raise ValueError("参考配准失败：目标四角外推超出实拍图")


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gradient_x, gradient_y)


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.ravel().astype(np.float64)
    second_flat = second.ravel().astype(np.float64)
    first_flat -= first_flat.mean()
    second_flat -= second_flat.mean()
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    return float(np.dot(first_flat, second_flat) / denominator) if denominator > 1e-12 else 0.0


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first * mu_first
    sigma_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second * mu_second
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_first * mu_second
    constant_one = 0.01**2
    constant_two = 0.03**2
    numerator = (2 * mu_first * mu_second + constant_one) * (
        2 * covariance + constant_two
    )
    denominator = (mu_first**2 + mu_second**2 + constant_one) * (
        sigma_first + sigma_second + constant_two
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _spectral_peak_ratio(gray: np.ndarray) -> float:
    """测量去除慢变化后频谱尖峰相对背景的强度。"""

    scale = min(1.0, 768 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 2.5)
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(high_pass * window)))
    center_y, center_x = np.array(spectrum.shape) // 2
    radius = max(4, min(spectrum.shape) // 24)
    spectrum[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ] = 0.0
    positive = spectrum[spectrum > 0]
    if not positive.size:
        return 1.0
    return float(np.percentile(positive, 99.9) / max(np.percentile(positive, 75), 1e-8))


def _require_rgb(image: np.ndarray, label: str) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.float32):
        raise ValueError(f"{label}必须是 H×W×3 RGB uint8/float32 图像")


def _as_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
