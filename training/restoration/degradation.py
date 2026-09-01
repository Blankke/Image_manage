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
from typing import Literal

import cv2
import numpy as np

DEGRADATION_GRAPH_VERSION = "factorized-c-p-a-o-i-v1"
TargetStage = Literal["canonical", "photometric", "artifact", "optical", "observation"]


@dataclass(frozen=True, slots=True)
class DegradationStep:
    """一项可审计退化；参数中不得包含像素或文件路径。"""

    name: str
    stage: Literal["photometric", "artifact", "optical", "sensor"]
    active: bool
    severity: float
    parameters: dict[str, float | int | bool | str | list[float]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DegradationTrace:
    """C/P/A/O/I 图的逻辑追踪，不保存 augmentation 图片。"""

    seed: int
    target_stage: TargetStage
    identity: bool
    steps: tuple[DegradationStep, ...]
    version: str = DEGRADATION_GRAPH_VERSION

    def to_dict(self) -> dict[str, object]:
        artifacts = {step.name: step.active for step in self.steps}
        severity = {step.name: step.severity for step in self.steps}
        return {
            "version": self.version,
            "seed": self.seed,
            "target_stage": self.target_stage,
            "identity": self.identity,
            "artifacts": artifacts,
            "severity": severity,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class FactorizedDegradationSample:
    """逻辑中间状态只存在于当前样本内存，供不同 task 选择监督 target。"""

    canonical_rgb: np.ndarray
    photometric_rgb: np.ndarray
    artifact_rgb: np.ndarray
    optical_rgb: np.ndarray
    observation_rgb: np.ndarray
    input_rgb: np.ndarray
    target_rgb: np.ndarray
    trace: DegradationTrace
    artifact_mask: np.ndarray | None = None


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
    max_ccm_residual: float = 0.06
    max_tone_gamma_delta: float = 0.18
    max_vignette_strength: float = 0.22
    min_jpeg_quality: int = 52
    max_jpeg_quality: int = 94
    clean_probability: float = 0.04
    apply_photometric: bool = True

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
        if not 0.0 <= self.max_ccm_residual <= 0.15:
            raise ValueError("max_ccm_residual 必须位于 0..0.15")
        if not 0.0 <= self.max_tone_gamma_delta <= 0.35:
            raise ValueError("max_tone_gamma_delta 必须位于 0..0.35")
        if not 0.0 <= self.max_vignette_strength <= 0.45:
            raise ValueError("max_vignette_strength 必须位于 0..0.45")
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

    exposure_stops = (
        float(rng.uniform(settings.min_exposure_stops, settings.max_exposure_stops))
        if settings.apply_photometric
        else 0.0
    )
    wb_delta = (
        rng.uniform(-settings.max_white_balance_shift, settings.max_white_balance_shift, size=3)
        .astype(np.float32)
        if settings.apply_photometric
        else np.zeros(3, np.float32)
    )
    white_balance = np.clip(1.0 + wb_delta - wb_delta.mean(), 0.65, 1.45)
    illumination = (
        _illumination_field(height, width, rng, settings.max_illumination_gradient)
        if settings.apply_photometric
        else np.ones((height, width), np.float32)
    )
    linear *= white_balance.reshape(1, 1, 3) * float(2.0**exposure_stops)
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


def factorized_degradation_graph(
    clean_rgb: np.ndarray,
    *,
    task: Literal["fidelity", "photometric", "demoire", "reflection", "identity"],
    seed: int,
    config: CameraDegradationConfig | None = None,
) -> FactorizedDegradationSample:
    """按 C→P→A→O→I 生成在线监督对，并记录完整参数追踪。

    Fidelity 的 target 为 A（没有专项 artifact 时 A=P），因此曝光、WB 与 illumination
    会同时保留在输入和 target；Photometric、Demoire、Reflection 分别使用 C、P、P。
    """

    if seed < 0:
        raise ValueError("退化 seed 不能为负数")
    settings = config or CameraDegradationConfig()
    rng = np.random.default_rng(seed)
    canonical = _as_float_rgb(clean_rgb)
    if task == "identity":
        trace = DegradationTrace(seed, "canonical", True, ())
        return FactorizedDegradationSample(
            canonical,
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            trace,
        )
    if rng.random() < settings.clean_probability:
        target_stage: TargetStage = {
            "fidelity": "artifact",
            "photometric": "canonical",
            "demoire": "photometric",
            "reflection": "photometric",
        }[task]
        trace = DegradationTrace(seed, target_stage, True, ())
        return FactorizedDegradationSample(
            canonical,
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            canonical.copy(),
            trace,
        )

    photometric, photo_steps = _photometric_stage(canonical, rng, settings)
    artifact = photometric.copy()
    artifact_mask: np.ndarray | None = None
    artifact_steps: list[DegradationStep] = []
    if task == "demoire":
        from training.p3.degradations import synthetic_screen_recapture

        pair = synthetic_screen_recapture(photometric, rng)
        artifact = pair.input_rgb
        artifact_steps.append(
            DegradationStep(
                "moire",
                "artifact",
                True,
                float(pair.trace.get("severity", 0.5)),
                _trace_parameters(pair.trace),
            )
        )
    elif task == "reflection":
        from training.p3.degradations import synthetic_reflection

        pair = synthetic_reflection(photometric, rng)
        artifact = pair.input_rgb
        artifact_mask = pair.mask
        artifact_steps.append(
            DegradationStep(
                "reflection",
                "artifact",
                True,
                float(pair.trace.get("severity", 0.5)),
                _trace_parameters(pair.trace),
            )
        )
    else:
        artifact_steps.extend(
            (
                DegradationStep("moire", "artifact", False, 0.0, {}),
                DegradationStep("reflection", "artifact", False, 0.0, {}),
            )
        )

    if task in {"photometric", "demoire", "reflection"}:
        optical = artifact.copy()
        observation = artifact.copy()
        optical_steps: list[DegradationStep] = []
        sensor_steps: list[DegradationStep] = []
    else:
        optical, optical_steps = _optical_stage(artifact, rng, settings)
        observation, sensor_steps = _sensor_stage(optical, rng, settings)

    target_stage: TargetStage
    if task == "photometric":
        input_rgb, target_rgb, target_stage = photometric, canonical, "canonical"
    elif task in {"demoire", "reflection"}:
        input_rgb, target_rgb, target_stage = artifact, photometric, "photometric"
    else:
        input_rgb, target_rgb, target_stage = observation, artifact, "artifact"
    steps = tuple((*photo_steps, *artifact_steps, *optical_steps, *sensor_steps))
    identity = not any(step.active for step in steps)
    return FactorizedDegradationSample(
        canonical.copy(),
        photometric,
        artifact,
        optical,
        observation,
        np.ascontiguousarray(input_rgb),
        np.ascontiguousarray(target_rgb),
        DegradationTrace(seed, target_stage, identity, steps),
        artifact_mask,
    )


def _photometric_stage(
    image: np.ndarray,
    rng: np.random.Generator,
    config: CameraDegradationConfig,
) -> tuple[np.ndarray, list[DegradationStep]]:
    if not config.apply_photometric:
        inactive = [
            DegradationStep("exposure", "photometric", False, 0.0, {}),
            DegradationStep("white_balance", "photometric", False, 0.0, {}),
            DegradationStep("ccm", "photometric", False, 0.0, {}),
            DegradationStep("tone", "photometric", False, 0.0, {}),
            DegradationStep("illumination", "photometric", False, 0.0, {}),
            DegradationStep("vignette", "photometric", False, 0.0, {}),
        ]
        return image.copy(), inactive
    height, width = image.shape[:2]
    linear = _srgb_to_linear(image)
    ev = float(rng.uniform(config.min_exposure_stops, config.max_exposure_stops))
    delta = rng.uniform(-config.max_white_balance_shift, config.max_white_balance_shift, 3)
    gains = np.clip(1.0 + delta - delta.mean(), 0.65, 1.45).astype(np.float32)
    field = _illumination_field(height, width, rng, config.max_illumination_gradient)
    ccm_delta = rng.uniform(-config.max_ccm_residual, config.max_ccm_residual, (3, 3)).astype(
        np.float32
    )
    # 对角占主导且每行总增益接近 1，避免 synthetic 产生不可信的强色彩混合。
    ccm_delta *= np.array(
        [[0.45, 0.2, 0.2], [0.2, 0.45, 0.2], [0.2, 0.2, 0.45]], np.float32
    )
    ccm = np.eye(3, dtype=np.float32) + ccm_delta
    ccm /= np.maximum(ccm.sum(axis=1, keepdims=True), 1e-4)
    linear = np.einsum("ij,hwj->hwi", ccm, linear)
    vignette_strength = float(rng.uniform(0.0, config.max_vignette_strength))
    center_x, center_y = rng.uniform(-0.08, 0.08, 2)
    yy, xx = np.indices((height, width), dtype=np.float32)
    normalized_x = xx / max(1, width - 1) * 2.0 - 1.0 - center_x
    normalized_y = yy / max(1, height - 1) * 2.0 - 1.0 - center_y
    radius2 = np.clip((normalized_x * normalized_x + normalized_y * normalized_y) / 2.0, 0.0, 1.0)
    vignette = np.clip(1.0 - vignette_strength * radius2, 0.65, 1.0).astype(np.float32)
    linear *= (2.0**ev) * gains[None, None] * field[..., None] * vignette[..., None]
    output = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    gamma = rng.uniform(
        1.0 - config.max_tone_gamma_delta,
        1.0 + config.max_tone_gamma_delta,
        3,
    ).astype(np.float32)
    output = np.power(np.clip(output, 0.0, 1.0), gamma[None, None]).astype(np.float32)
    steps = [
        DegradationStep("exposure", "photometric", abs(ev) > 1e-6, min(1.0, abs(ev) / 1.0), {"ev": ev}),
        DegradationStep(
            "white_balance",
            "photometric",
            bool(np.max(np.abs(gains - 1.0)) > 1e-6),
            min(1.0, float(np.max(np.abs(gains - 1.0))) / 0.25),
            {"gains": gains.astype(float).tolist()},
        ),
        DegradationStep(
            "ccm",
            "photometric",
            bool(np.max(np.abs(ccm - np.eye(3, dtype=np.float32))) > 1e-6),
            min(1.0, float(np.max(np.abs(ccm - np.eye(3, dtype=np.float32)))) / 0.08),
            {"matrix": ccm.astype(float).reshape(-1).tolist()},
        ),
        DegradationStep(
            "tone",
            "photometric",
            bool(np.max(np.abs(gamma - 1.0)) > 1e-6),
            min(1.0, float(np.max(np.abs(gamma - 1.0))) / max(config.max_tone_gamma_delta, 1e-6)),
            {"gamma": gamma.astype(float).tolist()},
        ),
        DegradationStep(
            "illumination",
            "photometric",
            bool(float(field.max() - field.min()) > 1e-6),
            min(1.0, float(field.max() - field.min()) / 0.3),
            {"range": float(field.max() - field.min())},
        ),
        DegradationStep(
            "vignette",
            "photometric",
            vignette_strength > 1e-6,
            min(1.0, vignette_strength / max(config.max_vignette_strength, 1e-6)),
            {
                "strength": vignette_strength,
                "center_offset": [float(center_x), float(center_y)],
            },
        ),
    ]
    return np.ascontiguousarray(output), steps


def _optical_stage(
    image: np.ndarray,
    rng: np.random.Generator,
    config: CameraDegradationConfig,
) -> tuple[np.ndarray, list[DegradationStep]]:
    height, width = image.shape[:2]
    output = image.copy()
    scale = float(rng.uniform(config.min_resize_scale, config.max_resize_scale))
    small = (max(2, round(width * scale)), max(2, round(height * scale)))
    output = cv2.resize(output, small, interpolation=cv2.INTER_AREA)
    output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
    defocus = float(rng.uniform(0.25, config.max_defocus_sigma)) if rng.random() < config.defocus_probability else 0.0
    if defocus:
        output = cv2.GaussianBlur(output, (0, 0), defocus, borderType=cv2.BORDER_REFLECT_101)
    motion = int(rng.integers(3, config.max_motion_length + 1)) if rng.random() < config.motion_probability else 0
    motion_angle = float(rng.uniform(0.0, 180.0))
    if motion:
        output = cv2.filter2D(output, -1, _motion_kernel(motion, motion_angle), borderType=cv2.BORDER_REFLECT_101)
    ringing = float(rng.uniform(0.015, 0.09)) if rng.random() < config.ringing_probability else 0.0
    if ringing:
        blurred = cv2.GaussianBlur(output, (0, 0), 0.8)
        output = np.clip(output + ringing * (output - blurred), 0.0, 1.0)
    return np.ascontiguousarray(output.astype(np.float32)), [
        DegradationStep("resize", "optical", scale < 0.999, 1.0 - scale, {"scale": scale}),
        DegradationStep("defocus", "optical", defocus > 0.0, min(1.0, defocus / config.max_defocus_sigma), {"sigma": defocus}),
        DegradationStep("motion", "optical", motion > 0, min(1.0, motion / config.max_motion_length), {"length": motion, "angle_degrees": motion_angle}),
        DegradationStep("ringing", "optical", ringing > 0.0, min(1.0, ringing / 0.09), {"amount": ringing}),
    ]


def _sensor_stage(
    image: np.ndarray,
    rng: np.random.Generator,
    config: CameraDegradationConfig,
) -> tuple[np.ndarray, list[DegradationStep]]:
    noise = float(rng.uniform(0.0, config.max_noise_std))
    output = np.clip(image + rng.normal(0.0, noise, image.shape).astype(np.float32), 0.0, 1.0)
    quality = int(rng.integers(config.min_jpeg_quality, config.max_jpeg_quality + 1)) if rng.random() < config.jpeg_probability else 100
    if quality < 100:
        output = _jpeg_round_trip_rgb(output, quality)
    return np.ascontiguousarray(output), [
        DegradationStep("noise", "sensor", noise > 1e-6, min(1.0, noise / max(config.max_noise_std, 1e-6)), {"std": noise}),
        DegradationStep("jpeg", "sensor", quality < 100, (100 - quality) / max(1, 100 - config.min_jpeg_quality), {"quality": quality}),
    ]


def _trace_parameters(trace: dict[str, object]) -> dict[str, float | int | bool | str | list[float]]:
    allowed = (float, int, bool, str)
    return {
        str(key): value
        for key, value in trace.items()
        if isinstance(value, allowed)
        or (isinstance(value, list) and all(isinstance(item, allowed) for item in value))
    }  # type: ignore[return-value]


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
