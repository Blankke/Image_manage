"""完整的无版权测试卡和手机拍屏退化生成器。"""

from __future__ import annotations

import cv2
import numpy as np

from tests.synthetic.generators import add_banding, add_color_moire


def test_chart(width: int = 960, height: int = 540, seed: int = 21) -> np.ndarray:
    """生成彩条、灰阶、小字号文字、细线、渐变、棋盘格和随机纹理测试卡。"""

    image = np.full((height, width, 3), 24, np.uint8)
    bar_colors = np.array(
        [
            [255, 255, 255],
            [255, 230, 20],
            [30, 230, 240],
            [30, 220, 50],
            [230, 40, 220],
            [230, 40, 40],
            [30, 50, 230],
            [10, 10, 10],
        ],
        np.uint8,
    )
    bar_height = height // 5
    for index, color in enumerate(bar_colors):
        x0 = round(index * width / len(bar_colors))
        x1 = round((index + 1) * width / len(bar_colors))
        image[:bar_height, x0:x1] = color
    grayscale = np.linspace(0, 255, width, dtype=np.uint8)
    image[bar_height : bar_height * 2] = grayscale[None, :, None]
    gradient_x = np.linspace(0, 1, width, dtype=np.float32)
    gradient_y = np.linspace(0, 1, height - bar_height * 2, dtype=np.float32)[:, None]
    image[bar_height * 2 :, :, 0] = np.clip((0.2 + 0.8 * gradient_x) * 255, 0, 255)
    image[bar_height * 2 :, :, 1] = np.clip((0.15 + 0.7 * gradient_y) * 255, 0, 255)
    image[bar_height * 2 :, :, 2] = np.clip((0.7 - 0.5 * gradient_x) * 255, 0, 255)
    for thickness in range(1, 7):
        y = bar_height * 2 + thickness * 28
        cv2.line(image, (20, y), (width // 2 - 20, y), (245, 245, 245), thickness)
    for index, scale in enumerate((0.3, 0.4, 0.5, 0.7)):
        cv2.putText(
            image,
            f"ScreenRestore small text {index + 1}",
            (width // 2, bar_height * 2 + 35 + index * 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )
    generator = np.random.default_rng(seed)
    random_patch = generator.integers(0, 256, (height // 5, width // 4, 3), np.uint8)
    image[-random_patch.shape[0] :, -random_patch.shape[1] :] = random_patch
    yy, xx = np.indices((height // 5, width // 4))
    checker = (((xx // 5 + yy // 5) % 2) * 255).astype(np.uint8)
    image[-checker.shape[0] :, : checker.shape[1]] = checker[..., None]
    return image


def perspective_degradation(
    image_rgb: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把正面图投影为已知斜拍四边形。"""

    height, width = image_rgb.shape[:2]
    canvas_width, canvas_height = output_size or (round(width * 1.25), round(height * 1.3))
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    target = np.array(
        [
            [canvas_width * 0.12, canvas_height * 0.15],
            [canvas_width * 0.9, canvas_height * 0.06],
            [canvas_width * 0.84, canvas_height * 0.9],
            [canvas_width * 0.08, canvas_height * 0.82],
        ],
        np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image_rgb, matrix, (canvas_width, canvas_height)), target


def rotate_degradation(image_rgb: np.ndarray, angle: float) -> np.ndarray:
    """以反射边界模拟手机小角度旋转。"""

    height, width = image_rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image_rgb, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def tone_degradation(
    image_rgb: np.ndarray,
    gamma: float = 1.5,
    exposure_stops: float = -0.4,
    color_gain: tuple[float, float, float] = (1.18, 0.92, 0.78),
) -> np.ndarray:
    """叠加 Gamma、曝光与 RGB 色偏。"""

    value = image_rgb.astype(np.float32) / 255.0
    value = np.power(value, gamma) * (2.0**exposure_stops)
    value *= np.asarray(color_gain, np.float32).reshape(1, 1, 3)
    return np.clip(np.rint(value * 255), 0, 255).astype(np.uint8)


def gaussian_noise(image_rgb: np.ndarray, sigma: float = 12.0, seed: int = 5) -> np.ndarray:
    """叠加可复现高斯噪声。"""

    noise = np.random.default_rng(seed).normal(0, sigma, image_rgb.shape)
    return np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def poisson_noise(image_rgb: np.ndarray, peak: float = 35.0, seed: int = 6) -> np.ndarray:
    """叠加与亮度相关的泊松噪声。"""

    normalized = image_rgb.astype(np.float32) / 255.0
    noisy = np.random.default_rng(seed).poisson(normalized * peak) / peak
    return np.clip(np.rint(noisy * 255), 0, 255).astype(np.uint8)


def jpeg_degradation(image_rgb: np.ndarray, quality: int = 35) -> np.ndarray:
    """通过内存编码叠加 JPEG 块效应。"""

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("合成 JPEG 退化失败")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def motion_blur(image_rgb: np.ndarray, length: int = 13, angle: float = 12.0) -> np.ndarray:
    """用归一化直线 PSF 合成运动模糊。"""

    size = max(3, length | 1)
    kernel = np.zeros((size, size), np.float32)
    center = (size - 1) / 2
    radius = (length - 1) / 2
    radians = np.deg2rad(angle)
    dx, dy = radius * np.cos(radians), radius * np.sin(radians)
    cv2.line(
        kernel,
        (round(center - dx), round(center - dy)),
        (round(center + dx), round(center + dy)),
        1,
        1,
    )
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def defocus_blur(image_rgb: np.ndarray, radius: int = 5) -> np.ndarray:
    """用圆盘 PSF 合成失焦模糊。"""

    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    kernel /= float(kernel.sum())
    return cv2.filter2D(image_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def local_glow(image_rgb: np.ndarray, center: tuple[float, float] = (0.7, 0.3)) -> np.ndarray:
    """叠加局部低饱和光晕。"""

    height, width = image_rgb.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    distance = np.square((xx / width - center[0]) / 0.22) + np.square(
        (yy / height - center[1]) / 0.28
    )
    mask = np.exp(-0.5 * distance)[..., None]
    return np.clip(image_rgb.astype(np.float32) * (1 - mask * 0.2) + 255 * mask * 0.42, 0, 255).astype(
        np.uint8
    )


__all__ = [
    "add_banding",
    "add_color_moire",
    "defocus_blur",
    "gaussian_noise",
    "jpeg_degradation",
    "local_glow",
    "motion_blur",
    "perspective_degradation",
    "poisson_noise",
    "rotate_degradation",
    "test_chart",
    "tone_degradation",
]
