"""传统色彩去摩尔纹、热度图与实验性频域 Gaussian notch。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ImageOperator, ProcessingContext
from screenrestore.core.parameters import ParameterModel

from ._utils import clip_float, require_range, require_rgb_float


class DemoireMode(StrEnum):
    """传统去摩尔纹模式。"""

    CHROMA = "chroma"
    JOINT_EDGE_AWARE = "joint_edge_aware"
    FREQUENCY = "frequency_experimental"


@dataclass
class DemoireParameters(ParameterModel):
    """去摩尔纹参数；频域自动峰检测默认关闭。"""

    mode: DemoireMode = DemoireMode.CHROMA
    strength: float = 0.45
    chroma_radius: float = 2.2
    edge_protection: float = 0.8
    heat_threshold: float = 0.2
    auto_frequency: bool = False
    notch_radius: float = 7.0
    notch_depth: float = 0.75
    manual_notches: list[list[float]] = field(default_factory=list)
    luma_sigma_color: float = 0.06
    structural_edge_sigma: float = 1.35
    chroma_relative_strength: float = 0.7
    periodicity_threshold: float = 0.2
    periodicity_scale: float = 3.0
    minimum_filter_weight: float = 0.3

    def validate(self) -> None:
        require_range("strength", self.strength, 0.0, 1.0)
        require_range("chroma_radius", self.chroma_radius, 0.4, 12.0)
        require_range("edge_protection", self.edge_protection, 0.0, 1.0)
        require_range("heat_threshold", self.heat_threshold, 0.0, 0.9)
        require_range("notch_radius", self.notch_radius, 1.0, 50.0)
        require_range("notch_depth", self.notch_depth, 0.0, 1.0)
        require_range("luma_sigma_color", self.luma_sigma_color, 0.005, 0.25)
        require_range("structural_edge_sigma", self.structural_edge_sigma, 0.4, 8.0)
        require_range("chroma_relative_strength", self.chroma_relative_strength, 0.0, 1.0)
        require_range("periodicity_threshold", self.periodicity_threshold, 0.0, 0.95)
        require_range("periodicity_scale", self.periodicity_scale, 1.0, 12.0)
        require_range("minimum_filter_weight", self.minimum_filter_weight, 0.0, 1.0)
        if len(self.manual_notches) > 64:
            raise ValueError("手工陷波点不能超过 64 个")
        for point in self.manual_notches:
            if len(point) != 2 or not all(0.0 <= float(value) <= 1.0 for value in point):
                raise ValueError("陷波点必须是 [0,1] 范围的二维归一化坐标")


class DemoireOperator(ImageOperator[DemoireParameters]):
    """根据摩尔纹热度动态增强色度滤波，或执行显式频域陷波。"""

    id = "demoire"
    display_name = "去摩尔纹"
    parameter_type = DemoireParameters

    def default_parameters(self) -> DemoireParameters:
        return DemoireParameters()

    def apply(
        self,
        image: np.ndarray,
        params: DemoireParameters,
        context: ProcessingContext,
    ) -> np.ndarray:
        require_rgb_float(image)
        self.validate(params)
        if params.strength == 0:
            return image.copy()
        context.report(0.1, "分析摩尔纹结构")
        if params.mode == DemoireMode.JOINT_EDGE_AWARE:
            output, processing_mask = _joint_edge_aware_demoire(image, params)
            # 联合模式显示实际处理权重，比旧热度图更能解释哪些区域被平滑。
            context.metadata["moire_heatmap"] = processing_mask
            context.metadata["demoire"] = {
                "mode": params.mode.value,
                "mean_processing_weight": float(np.mean(processing_mask)),
                "p95_processing_weight": float(np.quantile(processing_mask, 0.95)),
            }
        elif params.mode == DemoireMode.FREQUENCY:
            context.metadata["moire_heatmap"] = moire_heatmap(image)
            output = _frequency_demoire(image, params, context)
        else:
            heat = moire_heatmap(image)
            context.metadata["moire_heatmap"] = heat
            output = _chroma_demoire(image, heat, params)
        context.report(1.0, "去摩尔纹完成")
        return output


def moire_heatmap(image_rgb: np.ndarray) -> np.ndarray:
    """综合局部色彩变化、高频能量、周期峰和边缘不一致度。"""

    require_rgb_float(image_rgb)
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[..., 0]
    chroma = ycrcb[..., 1:]
    luminance_low = cv2.GaussianBlur(luminance, (0, 0), 1.2)
    high_frequency = np.abs(luminance - luminance_low)
    chroma_low = cv2.GaussianBlur(chroma, (0, 0), 2.0)
    color_variation = np.sqrt(np.sum(np.square(chroma - chroma_low), axis=2))
    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gradient_x, gradient_y)
    edge = _robust_normalize(edge)
    color_texture_without_edge = color_variation * (1.0 - edge)
    local_periodic = cv2.GaussianBlur(high_frequency, (0, 0), 2.5)
    peak_score = _global_periodicity_score(luminance)
    heat = (
        0.42 * _robust_normalize(color_variation)
        + 0.23 * _robust_normalize(high_frequency)
        + 0.25 * _robust_normalize(color_texture_without_edge)
        + 0.10 * _robust_normalize(local_periodic) * peak_score
    )
    return np.clip(cv2.GaussianBlur(heat, (0, 0), 1.0), 0.0, 1.0).astype(np.float32)


def frequency_spectrum(image_rgb: np.ndarray, max_edge: int = 768) -> np.ndarray:
    """返回用于 UI 显示的对数亮度频谱 RGB uint8 图。"""

    require_rgb_float(image_rgb)
    height, width = image_rgb.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale < 1:
        image_rgb = cv2.resize(
            image_rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    luminance = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    window = np.outer(np.hanning(luminance.shape[0]), np.hanning(luminance.shape[1])).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(luminance * window))))
    normalized = (_robust_normalize(spectrum) * 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)


def detect_frequency_peaks(luminance: np.ndarray, max_peaks: int = 6) -> list[list[float]]:
    """在缩略亮度频谱中寻找远离中心的局部异常峰，返回归一化坐标。"""

    if luminance.ndim != 2:
        raise ValueError("峰检测需要二维亮度图")
    height, width = luminance.shape
    scale = min(1.0, 512 / max(height, width))
    if scale < 1:
        luminance = cv2.resize(
            luminance,
            (max(16, round(width * scale)), max(16, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    window = np.outer(np.hanning(luminance.shape[0]), np.hanning(luminance.shape[1]))
    log_spectrum = np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2(luminance.astype(np.float32) * window)))
    ).astype(np.float32)
    background = cv2.GaussianBlur(log_spectrum, (0, 0), 7.0)
    residual = log_spectrum - background
    yy, xx = np.indices(residual.shape)
    cy, cx = (np.array(residual.shape) - 1) / 2
    radius = np.hypot(xx - cx, yy - cy)
    residual[radius < min(residual.shape) * 0.08] = -np.inf
    # 坐标轴上的正常文字/栅格谐波风险较高，自动模式降低其优先级。
    residual[np.abs(xx - cx) < 3] *= 0.35
    residual[np.abs(yy - cy) < 3] *= 0.35
    local_max = residual == cv2.dilate(residual, np.ones((7, 7), np.uint8))
    finite_values = residual[np.isfinite(residual)]
    threshold = float(np.percentile(finite_values, 99.5)) if finite_values.size else np.inf
    ys, xs = np.where(local_max & (residual >= threshold))
    ranked = sorted(zip(ys, xs, strict=True), key=lambda point: residual[point], reverse=True)
    points: list[list[float]] = []
    for y, x in ranked:
        normalized = [float(x / max(1, residual.shape[1] - 1)), float(y / max(1, residual.shape[0] - 1))]
        if all(np.linalg.norm(np.asarray(normalized) - np.asarray(item)) > 0.035 for item in points):
            points.append(normalized)
        if len(points) >= max_peaks:
            break
    return points


def _chroma_demoire(
    image: np.ndarray,
    heat: np.ndarray,
    params: DemoireParameters,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    luminance, cr, cb = cv2.split(ycrcb)
    sigma = params.chroma_radius
    sigma_color = max(8.0, sigma * 18) / 255.0
    sigma_space = max(1.0, sigma * 2.5)
    cr_filtered = cv2.bilateralFilter(cr, 0, sigma_color, sigma_space)
    cb_filtered = cv2.bilateralFilter(cb, 0, sigma_color, sigma_space)
    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edge = _robust_normalize(np.hypot(gradient_x, gradient_y))
    active_heat = np.clip(
        (heat - params.heat_threshold) / max(0.05, 1.0 - params.heat_threshold), 0.0, 1.0
    )
    blend = params.strength * active_heat * (1.0 - params.edge_protection * edge)
    cr_mixed = cr * (1 - blend) + cr_filtered * blend
    cb_mixed = cb * (1 - blend) + cb_filtered * blend
    output = cv2.merge((luminance, cr_mixed, cb_mixed))
    return clip_float(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB))


def _joint_edge_aware_demoire(
    image: np.ndarray,
    params: DemoireParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """联合平滑亮度/色度小振荡，同时保护大尺度结构边缘。"""

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[..., 0]
    # 先在较大尺度求结构边缘，使 2～4 px 的屏幕栅格不会被误当成真实轮廓。
    structural = cv2.GaussianBlur(
        luminance,
        (0, 0),
        params.structural_edge_sigma,
    )
    gradient_x = cv2.Sobel(structural, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(structural, cv2.CV_32F, 0, 1, ksize=3)
    structural_edge = np.square(_percentile_normalize(np.hypot(gradient_x, gradient_y), 0, 95))

    # 结构张量衡量局部高频是否长期保持同一方向；规则屏幕栅格会得到高相干度，
    # 多方向自然纹理则只保留最低处理权重，避免把建筑、毛发和织物整体磨平。
    high_frequency = luminance - cv2.GaussianBlur(luminance, (0, 0), 1.2)
    high_x = cv2.Sobel(high_frequency, cv2.CV_32F, 1, 0, ksize=3)
    high_y = cv2.Sobel(high_frequency, cv2.CV_32F, 0, 1, ksize=3)
    tensor_xx = cv2.GaussianBlur(
        np.square(high_x),
        (0, 0),
        params.periodicity_scale,
    )
    tensor_yy = cv2.GaussianBlur(
        np.square(high_y),
        (0, 0),
        params.periodicity_scale,
    )
    tensor_xy = cv2.GaussianBlur(
        high_x * high_y,
        (0, 0),
        params.periodicity_scale,
    )
    coherence = np.sqrt(
        np.square(tensor_xx - tensor_yy) + 4.0 * np.square(tensor_xy)
    ) / (tensor_xx + tensor_yy + 1e-8)
    coherence = np.power(
        np.clip(
            (coherence - params.periodicity_threshold)
            / max(1e-6, 1.0 - params.periodicity_threshold),
            0.0,
            1.0,
        ),
        1.5,
    )
    periodic_energy = _percentile_normalize(
        np.sqrt(tensor_xx + tensor_yy),
        15,
        95,
    )
    periodicity = cv2.GaussianBlur(coherence * periodic_energy, (0, 0), 1.0)
    adaptive_weight = params.minimum_filter_weight + (
        1.0 - params.minimum_filter_weight
    ) * periodicity
    processing_mask = (
        params.strength
        * adaptive_weight
        * (1.0 - params.edge_protection * structural_edge)
    )

    filtered_luminance = cv2.bilateralFilter(
        luminance,
        0,
        params.luma_sigma_color,
        params.chroma_radius,
    )
    ycrcb[..., 0] = (
        luminance * (1.0 - processing_mask) + filtered_luminance * processing_mask
    )
    chroma_weight = processing_mask * params.chroma_relative_strength
    for channel in (1, 2):
        source = ycrcb[..., channel]
        filtered = cv2.bilateralFilter(
            source,
            0,
            max(0.03, params.luma_sigma_color * 1.4),
            params.chroma_radius * 1.25,
        )
        ycrcb[..., channel] = source * (1.0 - chroma_weight) + filtered * chroma_weight
    output = clip_float(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
    return output, np.ascontiguousarray(processing_mask.astype(np.float32))


def _percentile_normalize(
    values: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """用明确分位范围归一化局部能量，避免极少异常点支配掩膜。"""

    low, high = np.percentile(values, (low_percentile, high_percentile))
    return np.clip(
        (values - low) / max(1e-6, float(high - low)),
        0.0,
        1.0,
    ).astype(np.float32)


def _frequency_demoire(
    image: np.ndarray,
    params: DemoireParameters,
    context: ProcessingContext,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    points = [list(point) for point in params.manual_notches]
    if params.auto_frequency:
        points.extend(detect_frequency_peaks(ycrcb[..., 0]))
    context.metadata["frequency_notches"] = points
    if not points:
        return image.copy()
    context.report(0.45, "应用柔和频域陷波")
    filtered_channels = []
    for index in range(3):
        channel_strength = params.strength * (0.45 if index == 0 else 1.0)
        filtered = _gaussian_notch(
            ycrcb[..., index], points, params.notch_radius, params.notch_depth
        )
        filtered_channels.append(
            ycrcb[..., index] * (1.0 - channel_strength) + filtered * channel_strength
        )
    output = np.stack(filtered_channels, axis=2).astype(np.float32)
    return clip_float(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB))


def _gaussian_notch(
    channel: np.ndarray,
    points: list[list[float]],
    radius: float,
    depth: float,
) -> np.ndarray:
    height, width = channel.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    mask = np.ones((height, width), np.float32)
    for normalized_x, normalized_y in points:
        px = normalized_x * (width - 1)
        py = normalized_y * (height - 1)
        symmetric_x = width - 1 - px
        symmetric_y = height - 1 - py
        for center_x, center_y in ((px, py), (symmetric_x, symmetric_y)):
            distance_squared = np.square(xx - center_x) + np.square(yy - center_y)
            notch = 1.0 - depth * np.exp(-distance_squared / (2.0 * radius**2))
            mask *= notch.astype(np.float32)
    transformed = np.fft.fftshift(np.fft.fft2(channel))
    restored = np.real(np.fft.ifft2(np.fft.ifftshift(transformed * mask))).astype(np.float32)
    return np.clip(restored, 0.0, 1.0)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (5, 98))
    return np.clip((values - low) / max(1e-6, float(high - low)), 0.0, 1.0).astype(np.float32)


def _global_periodicity_score(luminance: np.ndarray) -> float:
    height, width = luminance.shape
    scale = min(1.0, 256 / max(height, width))
    if scale < 1:
        luminance = cv2.resize(
            luminance, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(luminance - float(luminance.mean()))))
    cy, cx = np.array(spectrum.shape) // 2
    radius = max(3, min(spectrum.shape) // 16)
    spectrum[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1] = 0
    peak = float(np.percentile(spectrum, 99.9))
    baseline = float(np.percentile(spectrum, 90)) + 1e-6
    return float(np.clip((peak / baseline - 1.0) / 8.0, 0.0, 1.0))
