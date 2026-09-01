"""P3 在线合成退化；所有增强只存在于内存并返回参数追踪。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DegradationPair:
    input_rgb: np.ndarray
    target_rgb: np.ndarray
    trace: dict[str, object]
    mask: np.ndarray | None = None
    unresolved_mask: np.ndarray | None = None


def mild_dewarp_grid(
    rows: int,
    columns: int,
    rng: np.random.Generator,
    kind: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """生成 TPS/mesh/curl/bend/lift/sinusoid 风格轻度逆位移网格。"""

    selected = kind or str(rng.choice(("tps", "mesh", "curl", "bend", "lift", "sinusoid")))
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        np.linspace(-1.0, 1.0, columns, dtype=np.float32),
        indexing="ij",
    )
    amplitude = float(rng.uniform(0.004, 0.035))
    displacement = np.zeros((rows, columns, 2), np.float32)
    if selected == "curl":
        displacement[..., 0] = amplitude * yy * (1.0 - xx * xx)
    elif selected == "bend":
        displacement[..., 1] = amplitude * xx * xx * np.sign(yy + 1e-6)
    elif selected == "lift":
        displacement[..., 0] = amplitude * xx * yy
        displacement[..., 1] = -amplitude * (1.0 - xx * xx) * (1.0 - yy * yy)
    elif selected == "sinusoid":
        displacement[..., 1] = amplitude * np.sin(np.pi * xx) * (1.0 - yy * yy)
    elif selected == "mesh":
        random_grid = rng.normal(0.0, amplitude, displacement.shape).astype(np.float32)
        displacement = cv2.GaussianBlur(random_grid, (0, 0), 1.2)
    elif selected == "tps":
        centers = rng.uniform(-0.8, 0.8, (4, 2))
        for center_x, center_y in centers:
            radius2 = np.square(xx - center_x) + np.square(yy - center_y)
            basis = radius2 * np.log(np.maximum(radius2, 1e-5))
            displacement[..., 0] += float(rng.normal(0, amplitude / 6)) * basis
            displacement[..., 1] += float(rng.normal(0, amplitude / 6)) * basis
    else:
        raise ValueError(f"未知 dewarp 合成类型：{selected}")
    displacement = np.clip(displacement, -0.06, 0.06).astype(np.float32)
    return displacement, {"kind": selected, "amplitude": amplitude, "rows": rows, "columns": columns}


def shared_photometric_nuisance(
    input_rgb: np.ndarray,
    target_rgb: np.ndarray,
    rng: np.random.Generator,
) -> DegradationPair:
    """向 Fidelity input/target 同时施加同一摄影 nuisance，防止网络改色调。"""

    ev = float(rng.uniform(-0.7, 0.5))
    gains = rng.uniform(0.82, 1.18, 3).astype(np.float32)
    height, width = input_rgb.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    field = 1.0 + float(rng.uniform(-0.18, 0.18)) * (xx / max(1, width - 1) - 0.5)

    def apply(value: np.ndarray) -> np.ndarray:
        return np.clip(
            value.astype(np.float32) * (2.0**ev) * gains[None, None] * field[..., None],
            0.0,
            1.0,
        ).astype(np.float32)

    return DegradationPair(
        apply(input_rgb),
        apply(target_rgb),
        {"type": "shared_photometric", "ev": ev, "gains": gains.astype(float).tolist()},
    )


def synthetic_screen_recapture(clean_rgb: np.ndarray, rng: np.random.Generator) -> DegradationPair:
    """用多种子像素布局、光学/CFA 采样和 JPEG 生成屏摄配对。"""

    clean = np.asarray(clean_rgb, np.float32)
    height, width = clean.shape[:2]
    scale = int(rng.integers(2, 6))
    panel = cv2.resize(clean, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.indices(panel.shape[:2])
    phase_offset = int(rng.integers(0, 3))
    phase = ((xx + phase_offset) % 3).astype(np.int32)
    layout = str(rng.choice(("rgb_stripe", "bgr_stripe", "pentile_like")))
    subpixel = np.ones_like(panel)
    order = (0, 1, 2) if layout == "rgb_stripe" else (2, 1, 0)
    for phase_index, channel in enumerate(order):
        subpixel[..., channel] *= np.where(phase == phase_index, 1.0, 0.76)
    if layout == "pentile_like":
        subpixel[..., 1] *= np.where((yy + phase_offset) % 2 == 0, 1.0, 0.82)
    panel *= subpixel
    angle = float(rng.uniform(-2.5, 2.5))
    matrix = cv2.getRotationMatrix2D((panel.shape[1] / 2, panel.shape[0] / 2), angle, 1.0)
    panel = cv2.warpAffine(panel, matrix, (panel.shape[1], panel.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    psf_sigma = float(rng.uniform(0.2, 1.25))
    panel = cv2.GaussianBlur(panel, (0, 0), psf_sigma, borderType=cv2.BORDER_REFLECT_101)
    recaptured = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
    # 轻量 Bayer/CFA 相位近似：不同像素位置保留不同通道响应，再做局部重建。
    cfa_phase = int(rng.integers(0, 4))
    cfa = np.ones_like(recaptured)
    parity = ((np.indices((height, width)).sum(axis=0) + cfa_phase) % 4)
    cfa[..., 0] *= np.where(parity == 0, 1.0, 0.96)
    cfa[..., 1] *= np.where((parity == 1) | (parity == 2), 1.0, 0.97)
    cfa[..., 2] *= np.where(parity == 3, 1.0, 0.96)
    color_gains = rng.uniform(0.94, 1.06, 3).astype(np.float32)
    recaptured = np.clip(recaptured * cfa * color_gains[None, None], 0.0, 1.0)
    jpeg_quality = int(rng.integers(55, 96))
    bgr = cv2.cvtColor(np.rint(recaptured * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Demoire synthetic JPEG 编码失败")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    recaptured = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    severity = float(np.clip(0.22 + 0.12 * scale + abs(angle) / 10.0, 0.0, 1.0))
    return DegradationPair(
        np.clip(recaptured, 0.0, 1.0).astype(np.float32),
        clean.copy(),
        {
            "type": "screen_recapture",
            "severity": severity,
            "layout": layout,
            "panel_scale": scale,
            "subpixel_phase": phase_offset,
            "angle_degrees": angle,
            "psf_sigma": psf_sigma,
            "cfa_phase": cfa_phase,
            "jpeg_quality": jpeg_quality,
            "color_gains": color_gains.astype(float).tolist(),
        },
    )


def synthetic_reflection(clean_rgb: np.ndarray, rng: np.random.Generator) -> DegradationPair:
    """合成软反射、ghost、色偏与局部眩光，并显式标记不可见区域。"""

    clean = np.asarray(clean_rgb, np.float32)
    height, width = clean.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    center_x, center_y = rng.uniform(0.2, 0.8, 2)
    sigma_x, sigma_y = rng.uniform(0.08, 0.28, 2)
    mask = np.exp(
        -0.5
        * (
            np.square((xx / max(1, width - 1) - center_x) / sigma_x)
            + np.square((yy / max(1, height - 1) - center_y) / sigma_y)
        )
    ).astype(np.float32)
    strength = float(rng.uniform(0.15, 0.65))
    ghost_offset = (int(rng.integers(-9, 10)), int(rng.integers(-9, 10)))
    reflection_layer = np.roll(np.flip(clean, axis=1), ghost_offset, axis=(0, 1))
    blur_sigma = float(rng.uniform(1.0, 5.0))
    reflection_layer = cv2.GaussianBlur(reflection_layer, (0, 0), blur_sigma)
    tint = rng.uniform(0.82, 1.18, 3).astype(np.float32)
    reflection_layer = np.clip(reflection_layer * tint[None, None], 0.0, 1.0)
    alpha = np.clip(mask * strength, 0.0, 0.92)
    reflected = clean * (1.0 - alpha[..., None] * 0.25) + reflection_layer * alpha[..., None]
    glare_strength = float(rng.uniform(0.0, 0.45))
    reflected += np.power(mask[..., None], 2.0) * glare_strength
    reflected = np.clip(reflected, 0.0, 1.0)
    unresolved = (reflected.max(axis=2) >= 0.995) & (mask >= 0.5)
    severity = float(np.clip(strength + 0.5 * glare_strength, 0.0, 1.0))
    return DegradationPair(
        reflected.astype(np.float32),
        clean.copy(),
        {
            "type": "reflection",
            "severity": severity,
            "strength": strength,
            "blur_sigma": blur_sigma,
            "ghost_offset": list(ghost_offset),
            "tint": tint.astype(float).tolist(),
            "glare_strength": glare_strength,
            "unresolved_fraction": float(unresolved.mean()),
        },
        mask,
        unresolved.astype(np.float32),
    )


__all__ = [
    "DegradationPair",
    "mild_dewarp_grid",
    "shared_photometric_nuisance",
    "synthetic_reflection",
    "synthetic_screen_recapture",
]
