"""DIV2K HR 的在线手机相机退化模拟。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.restoration.train \\
        --train-hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR" \\
        --validation-hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \\
        --output-directory "$SCREENRESTORE_RUN_ROOT/restoration/fidelity-smoke" --epochs 2

此模块只从调用方交给它的 clean HR 裁剪产生退化图，绝不将 augmentation 写成磁盘副本。
光学模糊、曝光、白平衡和照明在近似线性 RGB 中处理；JPEG 是唯一的 BGR/8 位边界。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CameraDegradationConfig:
    """受限相机退化分布；所有数值均服务于同尺寸 Fidelity 恢复。"""

    min_resize_scale: float = 0.55
    max_resize_scale: float = 0.95
    defocus_probability: float = 0.55
    motion_probability: float = 0.45
    jpeg_probability: float = 0.8
    ringing_probability: float = 0.25
    max_defocus_sigma: float = 2.2
    max_motion_length: int = 13
    max_noise_std: float = 0.018
    min_exposure_stops: float = -0.65
    max_exposure_stops: float = 0.45
    max_white_balance_shift: float = 0.18
    max_illumination_gradient: float = 0.22
    min_jpeg_quality: int = 52
    max_jpeg_quality: int = 94
    clean_probability: float = 0.04

    def __post_init__(self) -> None:
        if not 0.25 <= self.min_resize_scale <= self.max_resize_scale <= 1.0:
            raise ValueError("resize scale 必须位于 0.25..1")
        for name in (
            "defocus_probability",
            "motion_probability",
            "jpeg_probability",
            "ringing_probability",
            "clean_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")
        if not 0.25 <= self.max_defocus_sigma <= 6.0:
            raise ValueError("max_defocus_sigma 必须位于 0.25..6")
        if not 3 <= self.max_motion_length <= 31:
            raise ValueError("max_motion_length 必须位于 3..31")
        if not 0.0 <= self.max_noise_std <= 0.08:
            raise ValueError("max_noise_std 必须位于 0..0.08")
        if not -2.0 <= self.min_exposure_stops <= self.max_exposure_stops <= 2.0:
            raise ValueError("曝光范围必须位于 -2..2 stops")
        if not 0.0 <= self.max_white_balance_shift <= 0.45:
            raise ValueError("max_white_balance_shift 必须位于 0..0.45")
        if not 0.0 <= self.max_illumination_gradient <= 0.5:
            raise ValueError("max_illumination_gradient 必须位于 0..0.5")
        if not 15 <= self.min_jpeg_quality <= self.max_jpeg_quality <= 100:
            raise ValueError("JPEG quality 必须位于 15..100")


@dataclass(frozen=True, slots=True)
class CameraDegradationSample:
    """在线退化结果及不含图像内容的参数记录。"""

    degraded_rgb: np.ndarray
    parameters: dict[str, float | int | bool]


def degrade_camera_image(
    clean_rgb: np.ndarray,
    rng: np.random.Generator,
    config: CameraDegradationConfig | None = None,
) -> CameraDegradationSample:
    """将 clean RGB 退化为同尺寸 float32 输入，原图不会被改写。

    退化顺序近似手机成像链：有效采样/PSF、线性光光度变化、传感器噪声、sRGB 编码、JPEG
    与轻微振铃。它不引入纹理生成，也不会把目标图混入输入。
    """

    settings = config or CameraDegradationConfig()
    image = _as_float_rgb(clean_rgb)
    if rng.random() < settings.clean_probability:
        return CameraDegradationSample(
            degraded_rgb=image.copy(),
            parameters={"identity": True, **_config_marker(settings)},
        )

    height, width = image.shape[:2]
    linear = _srgb_to_linear(image)
    parameters: dict[str, float | int | bool] = {"identity": False}

    resize_scale = float(rng.uniform(settings.min_resize_scale, settings.max_resize_scale))
    # 先欠采样再插值回原坐标，模拟有限有效分辨率，同时保持训练目标同尺寸。
    small_size = (max(2, round(width * resize_scale)), max(2, round(height * resize_scale)))
    linear = cv2.resize(linear, small_size, interpolation=cv2.INTER_AREA)
    linear = cv2.resize(linear, (width, height), interpolation=cv2.INTER_CUBIC)
    parameters["resize_scale"] = resize_scale

    defocus_sigma = 0.0
    if rng.random() < settings.defocus_probability:
        defocus_sigma = float(rng.uniform(0.25, settings.max_defocus_sigma))
        linear = cv2.GaussianBlur(linear, (0, 0), defocus_sigma, borderType=cv2.BORDER_REFLECT_101)
    parameters["defocus_sigma"] = defocus_sigma

    motion_length = 0
    if rng.random() < settings.motion_probability:
        motion_length = int(rng.integers(3, settings.max_motion_length + 1))
        linear = cv2.filter2D(
            linear,
            -1,
            _motion_kernel(motion_length, float(rng.uniform(0.0, 180.0))),
            borderType=cv2.BORDER_REFLECT_101,
        )
    parameters["motion_length"] = motion_length

    exposure_stops = float(rng.uniform(settings.min_exposure_stops, settings.max_exposure_stops))
    wb_delta = rng.uniform(
        -settings.max_white_balance_shift,
        settings.max_white_balance_shift,
        size=3,
    ).astype(np.float32)
    white_balance = np.clip(1.0 + wb_delta - wb_delta.mean(), 0.65, 1.45)
    linear *= white_balance.reshape(1, 1, 3) * float(2.0**exposure_stops)
    illumination = _illumination_field(height, width, rng, settings.max_illumination_gradient)
    linear *= illumination[..., None]
    parameters.update(
        {
            "exposure_stops": exposure_stops,
            "white_balance_red": float(white_balance[0]),
            "white_balance_green": float(white_balance[1]),
            "white_balance_blue": float(white_balance[2]),
            "illumination_gradient": float(illumination.max() - illumination.min()),
        }
    )

    noise_std = float(rng.uniform(0.0, settings.max_noise_std))
    # 噪声在暗部以加性读出噪声为主，在亮部保留弱泊松近似项。
    shot_std = noise_std * (0.35 + 0.65 * np.sqrt(np.clip(linear, 0.0, 1.0)))
    linear += rng.normal(0.0, 1.0, linear.shape).astype(np.float32) * shot_std
    linear = np.clip(linear, 0.0, 1.0)
    parameters["noise_std"] = noise_std

    degraded = _linear_to_srgb(linear)
    jpeg_quality = 100
    if rng.random() < settings.jpeg_probability:
        jpeg_quality = int(rng.integers(settings.min_jpeg_quality, settings.max_jpeg_quality + 1))
        degraded = _jpeg_round_trip_rgb(degraded, jpeg_quality)
    parameters["jpeg_quality"] = jpeg_quality

    ringing_amount = 0.0
    if rng.random() < settings.ringing_probability:
        ringing_amount = float(rng.uniform(0.015, 0.09))
        blurred = cv2.GaussianBlur(degraded, (0, 0), 0.8, borderType=cv2.BORDER_REFLECT_101)
        degraded = np.clip(degraded + ringing_amount * (degraded - blurred), 0.0, 1.0)
    parameters["ringing_amount"] = ringing_amount
    parameters.update(_config_marker(settings))
    return CameraDegradationSample(
        degraded_rgb=np.ascontiguousarray(degraded.astype(np.float32)),
        parameters=parameters,
    )


def _as_float_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("相机退化需要 H×W×3 RGB 图像")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image.astype(np.float32) / 255.0)
    if image.dtype != np.float32:
        raise ValueError("相机退化只接受 uint8 或 float32 RGB 图像")
    if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError("float32 RGB 图像必须位于 [0,1]")
    return np.ascontiguousarray(image.copy())


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def _motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    size = max(3, length | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    radius = (length - 1) / 2.0
    radians = np.deg2rad(angle_degrees)
    dx, dy = radius * np.cos(radians), radius * np.sin(radians)
    cv2.line(
        kernel,
        (round(center - dx), round(center - dy)),
        (round(center + dx), round(center + dy)),
        1.0,
        1,
    )
    return kernel / max(float(kernel.sum()), 1e-6)


def _illumination_field(
    height: int,
    width: int,
    rng: np.random.Generator,
    max_gradient: float,
) -> np.ndarray:
    if max_gradient == 0.0:
        return np.ones((height, width), dtype=np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    xx = xx / max(1, width - 1) - 0.5
    yy = yy / max(1, height - 1) - 0.5
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    amplitude = float(rng.uniform(0.0, max_gradient))
    field = 1.0 + amplitude * (xx * np.cos(angle) + yy * np.sin(angle))
    return np.clip(field, 0.7, 1.3).astype(np.float32)


def _jpeg_round_trip_rgb(image: np.ndarray, quality: int) -> np.ndarray:
    # OpenCV JPEG 边界明确转换 RGB→BGR→RGB，训练内部其余步骤保持 RGB。
    bgr = cv2.cvtColor(np.rint(image * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("在线 JPEG 相机退化编码失败")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("在线 JPEG 相机退化解码失败")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _config_marker(config: CameraDegradationConfig) -> dict[str, float | int | bool]:
    """保留分布版本化线索，便于训练记录而不序列化任何图像内容。"""

    return {f"config_{key}": value for key, value in asdict(config).items()}
