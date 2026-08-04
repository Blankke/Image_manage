"""经典流水线构造和场景预设。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from screenrestore.operators import (
    BandingOperator,
    ClaheOperator,
    DeblurOperator,
    DehaloOperator,
    DemoireOperator,
    DenoiseOperator,
    EnhancementModelOperator,
    ExposureOperator,
    GeometryOperator,
    IlluminationOperator,
    LensDistortionOperator,
    MeshWarpOperator,
    OrientationOperator,
    ReflectionOperator,
    ResizeOperator,
    RestorationModelOperator,
    SharpenOperator,
    WhiteBalanceOperator,
)
from screenrestore.operators.banding import BandingParameters
from screenrestore.operators.dehalo import DehaloParameters
from screenrestore.operators.demoire import DemoireMode, DemoireParameters
from screenrestore.operators.denoise import DenoiseMode, DenoiseParameters
from screenrestore.operators.exposure import ExposureParameters
from screenrestore.operators.illumination import IlluminationParameters
from screenrestore.operators.local_contrast import ClaheParameters
from screenrestore.operators.reflection import ReflectionMode, ReflectionParameters
from screenrestore.operators.sharpen import SharpenParameters
from screenrestore.operators.white_balance import WhiteBalanceMode, WhiteBalanceParameters

from .pipeline import ImagePipeline, OperatorRegistry, OperatorState


class PresetId(StrEnum):
    """内置处理预设标识。"""

    DISPLAY = "display"
    ELECTRONIC_POSTER = "electronic_poster"
    CINEMA = "cinema"
    LED = "led"
    DOCUMENT = "document"
    ARTWORK = "artwork"
    GLOSSY_ARTWORK = "glossy_artwork"
    CUSTOM = "custom"


class ProcessingMode(StrEnum):
    """区分观测忠实恢复与允许生成统计细节的增强。"""

    FIDELITY = "fidelity"
    AI_ENHANCED = "ai_enhanced"


PRESET_NAMES = {
    PresetId.DISPLAY: "显示器",
    PresetId.ELECTRONIC_POSTER: "电子海报",
    PresetId.CINEMA: "电影院/投影（忠实）",
    PresetId.LED: "LED 大屏",
    PresetId.DOCUMENT: "文档/PPT",
    PresetId.ARTWORK: "艺术品/画作（色彩忠实）",
    PresetId.GLOSSY_ARTWORK: "覆膜/玻璃反光",
    PresetId.CUSTOM: "自定义",
}


def build_registry() -> OperatorRegistry:
    """构建不含可选模型依赖的核心算子注册表。"""

    return OperatorRegistry(
        [
            OrientationOperator(),
            LensDistortionOperator(),
            GeometryOperator(),
            MeshWarpOperator(),
            BandingOperator(),
            DemoireOperator(),
            RestorationModelOperator(),
            DenoiseOperator(),
            WhiteBalanceOperator(),
            ExposureOperator(),
            ClaheOperator(),
            IlluminationOperator(),
            ReflectionOperator(),
            DehaloOperator(),
            DeblurOperator(),
            SharpenOperator(),
            EnhancementModelOperator(),
            ResizeOperator(),
        ]
    )


def build_default_pipeline(registry: OperatorRegistry | None = None) -> ImagePipeline:
    """按安全默认顺序创建显示器预设流水线。"""

    active_registry = registry or build_registry()
    states = []
    disabled_by_default = {
        "lens_distortion",
        "mesh_warp",
        "reflection",
        "deblur",
        "restoration_model",
        "enhancement_model",
        "resize",
    }
    for operator_id in active_registry.ids:
        operator = active_registry.get(operator_id)
        states.append(
            OperatorState(
                operator=operator,
                enabled=operator_id not in disabled_by_default,
                params=operator.default_parameters(),
            )
        )
    pipeline = ImagePipeline(states)
    apply_preset(pipeline, PresetId.DISPLAY)
    apply_processing_mode(pipeline, ProcessingMode.FIDELITY)
    return pipeline


def apply_preset(pipeline: ImagePipeline, preset: PresetId) -> None:
    """应用场景参数；不覆盖几何、方向和输出尺寸等用户契约。"""

    preset = PresetId(preset)
    if preset == PresetId.CUSTOM:
        return
    definitions = _preset_definitions()[preset]
    for operator_id, (enabled, params) in definitions.items():
        state = pipeline.state(operator_id)
        state.enabled = enabled
        state.params = state.operator.parameter_type.from_dict(params)
    pipeline.cache.clear()


def apply_processing_mode(pipeline: ImagePipeline, mode: ProcessingMode) -> None:
    """应用工作模式的强制语义，不隐式启用尚未配置的模型。"""

    mode = ProcessingMode(mode)
    if mode == ProcessingMode.FIDELITY:
        pipeline.state("enhancement_model").enabled = False
    elif pipeline.state("enhancement_model").enabled:
        # 感知模型本身会增强高频，避免把预先锐化的 halo 固化成“纹理”。
        pipeline.state("sharpen").enabled = False
    pipeline.cache.clear()


def _preset_definitions() -> dict[PresetId, dict[str, tuple[bool, dict[str, Any]]]]:
    """集中声明预设差异，所有值仍可由用户覆盖。"""

    display = {
        "banding": (False, BandingParameters(strength=0.2).to_dict()),
        "demoire": (
            True,
            DemoireParameters(
                mode=DemoireMode.JOINT_EDGE_AWARE,
                strength=1.0,
                chroma_radius=2.5,
                edge_protection=0.65,
                luma_sigma_color=0.06,
                structural_edge_sigma=1.35,
                chroma_relative_strength=0.7,
            ).to_dict(),
        ),
        "denoise": (True, DenoiseParameters(strength=0.22).to_dict()),
        "white_balance": (
            True,
            WhiteBalanceParameters(
                mode=WhiteBalanceMode.WHITE_PATCH,
                max_gain=1.15,
                strength=0.2,
            ).to_dict(),
        ),
        "exposure": (
            True,
            ExposureParameters(
                auto_white_background_strength=1.0,
                white_background_min_area=0.65,
                white_background_saturation_ceiling=0.12,
                white_background_luminance_floor=0.55,
                white_background_quantile=0.9,
                target_white_background=0.985,
                max_white_background_gain=1.35,
            ).to_dict(),
        ),
        "clahe": (False, ClaheParameters(strength=0.1).to_dict()),
        "illumination": (False, IlluminationParameters(strength=0.08).to_dict()),
        "reflection": (False, ReflectionOperator().default_parameters().to_dict()),
        "dehalo": (
            True,
            DehaloParameters(
                strength=0.25,
                highlight_threshold=0.65,
                core_radius=0.7,
                halo_radius=2.2,
                max_correction=0.06,
                auto_gate=True,
                max_scene_median=0.32,
                min_highlight_area=0.02,
                max_highlight_area=0.25,
            ).to_dict(),
        ),
        "deblur": (False, DeblurOperator().default_parameters().to_dict()),
        "sharpen": (False, SharpenParameters(amount=0.15).to_dict()),
        "restoration_model": (
            False,
            RestorationModelOperator().default_parameters().to_dict(),
        ),
        "enhancement_model": (
            False,
            EnhancementModelOperator().default_parameters().to_dict(),
        ),
    }
    return {
        PresetId.DISPLAY: display,
        PresetId.ELECTRONIC_POSTER: {
            **display,
            "banding": (True, BandingParameters(strength=0.72).to_dict()),
            "clahe": (True, ClaheParameters(strength=0.42, clip_limit=1.8).to_dict()),
            "illumination": (True, IlluminationParameters(strength=0.38).to_dict()),
        },
        PresetId.CINEMA: {
            **display,
            "banding": (
                True,
                BandingParameters(
                    strength=0.08,
                    broad_haze_strength=0.4,
                    broad_haze_scale=95.0,
                    black_level_quantile=0.03,
                    max_haze_correction=0.045,
                ).to_dict(),
            ),
            "demoire": (
                True,
                DemoireParameters(mode=DemoireMode.CHROMA, strength=0.1).to_dict(),
            ),
            "denoise": (
                True,
                DenoiseParameters(
                    mode=DenoiseMode.LUMA_CHROMA,
                    strength=0.08,
                    luma_strength=1.6,
                    chroma_strength=4.0,
                ).to_dict(),
            ),
            "white_balance": (
                False,
                WhiteBalanceParameters(mode=WhiteBalanceMode.OFF).to_dict(),
            ),
            "exposure": (
                True,
                ExposureParameters(
                    auto_black_level_strength=1.0,
                    black_level_quantile=0.01,
                    target_black_level=0.002,
                    max_black_level_correction=0.025,
                    black_level_activation_ceiling=0.06,
                    auto_black_contrast=0.06,
                ).to_dict(),
            ),
            "clahe": (False, ClaheParameters(strength=0.06, clip_limit=1.2).to_dict()),
            "illumination": (False, IlluminationParameters(strength=0.05).to_dict()),
            "reflection": (
                False,
                ReflectionParameters(
                    mode=ReflectionMode.GRADIENT_DCT,
                    strength=0.2,
                    gradient_threshold=0.015,
                ).to_dict(),
            ),
            "dehalo": (False, DehaloParameters().to_dict()),
            "sharpen": (
                True,
                SharpenParameters(
                    radius=0.9,
                    amount=0.08,
                    threshold=0.012,
                    highlight_protection=0.45,
                    shadow_protection=0.4,
                ).to_dict(),
            ),
        },
        PresetId.LED: {
            **display,
            "banding": (True, BandingParameters(strength=0.8).to_dict()),
            "demoire": (
                True,
                DemoireParameters(
                    mode=DemoireMode.JOINT_EDGE_AWARE,
                    strength=1.0,
                    chroma_radius=3.0,
                    edge_protection=0.6,
                    luma_sigma_color=0.075,
                ).to_dict(),
            ),
            "dehalo": (False, DehaloParameters().to_dict()),
            "denoise": (True, DenoiseParameters(strength=0.3).to_dict()),
            "sharpen": (True, SharpenParameters(amount=0.42).to_dict()),
        },
        PresetId.DOCUMENT: {
            **display,
            "banding": (True, BandingParameters(strength=0.4).to_dict()),
            "demoire": (
                True,
                DemoireParameters(mode=DemoireMode.CHROMA, strength=0.2).to_dict(),
            ),
            "dehalo": (False, DehaloParameters().to_dict()),
            "denoise": (True, DenoiseParameters(strength=0.15).to_dict()),
            "white_balance": (True, WhiteBalanceParameters().to_dict()),
            "clahe": (True, ClaheParameters(strength=0.34, clip_limit=1.7).to_dict()),
            "illumination": (True, IlluminationParameters(strength=0.4).to_dict()),
            "sharpen": (True, SharpenParameters(amount=0.46).to_dict()),
        },
        # ── 场景 1：艺术品斜拍 ──
        # 目标：摄影式复现（photographic/colorimetric reproduction）
        # 约束：色彩忠实度优先于锐度；不自动修改构图比例
        # 油画、摄影作品的暗部、低对比度、暖色调是创作意图，不得被"修正"
        PresetId.ARTWORK: {
            **display,
            # 镜头畸变在画作拍摄中常见，默认启用
            "lens_distortion": (
                True,
                LensDistortionOperator().default_parameters().to_dict(),
            ),
            # 关闭所有可能破坏原作意图的算子
            "banding": (False, BandingParameters(strength=0.0).to_dict()),
            "demoire": (False, DemoireParameters(mode=DemoireMode.CHROMA, strength=0.0).to_dict()),
            "denoise": (
                True,
                DenoiseParameters(
                    mode=DenoiseMode.LUMA_CHROMA,
                    strength=0.06,
                    luma_strength=0.8,
                    chroma_strength=1.2,
                ).to_dict(),
            ),
            "white_balance": (
                True,
                WhiteBalanceParameters(
                    mode=WhiteBalanceMode.WHITE_PATCH,
                    max_gain=1.10,
                    strength=0.15,
                ).to_dict(),
            ),
            "exposure": (
                True,
                ExposureParameters(
                    exposure=0.0,
                    contrast=0.02,
                    auto_black_level_strength=0.0,
                    auto_white_background_strength=0.0,
                ).to_dict(),
            ),
            "clahe": (False, ClaheParameters(strength=0.0).to_dict()),
            "illumination": (
                True,
                IlluminationParameters(strength=0.08).to_dict(),
            ),
            "reflection": (
                False,
                ReflectionParameters(
                    mode=ReflectionMode.HIGHLIGHT_MASK,
                    strength=0.0,
                ).to_dict(),
            ),
            "dehalo": (False, DehaloParameters().to_dict()),
            "deblur": (False, DeblurOperator().default_parameters().to_dict()),
            "sharpen": (
                True,
                SharpenParameters(
                    radius=0.6,
                    amount=0.10,
                    threshold=0.015,
                    highlight_protection=0.6,
                    shadow_protection=0.5,
                ).to_dict(),
            ),
        },
        # ── 场景 4：透明覆盖层反光 ──
        # 退化模型 I = αT + βR（透射 + 反射）
        # 按反光强度分层：轻微(HSV高光压缩) → 中等(Gradient-domain separation) → 强烈(inpainting ≤8%)
        # Reflection mask 在原始 radiometric image 上检测，CLAHE 永远关闭
        PresetId.GLOSSY_ARTWORK: {
            **display,
            "lens_distortion": (
                True,
                LensDistortionOperator().default_parameters().to_dict(),
            ),
            "banding": (False, BandingParameters(strength=0.0).to_dict()),
            "demoire": (False, DemoireParameters(mode=DemoireMode.CHROMA, strength=0.0).to_dict()),
            "denoise": (
                True,
                DenoiseParameters(
                    mode=DenoiseMode.LUMA_CHROMA,
                    strength=0.08,
                    luma_strength=1.0,
                    chroma_strength=1.5,
                ).to_dict(),
            ),
            "white_balance": (
                True,
                WhiteBalanceParameters(
                    mode=WhiteBalanceMode.WHITE_PATCH,
                    max_gain=1.12,
                    strength=0.15,
                ).to_dict(),
            ),
            "exposure": (
                True,
                ExposureParameters(
                    exposure=0.0,
                    contrast=0.03,
                    auto_black_level_strength=0.0,
                    auto_white_background_strength=0.0,
                ).to_dict(),
            ),
            # CLAHE 会放大反光 → 永远关闭
            "clahe": (False, ClaheParameters(strength=0.0).to_dict()),
            "illumination": (
                True,
                IlluminationParameters(strength=0.12).to_dict(),
            ),
            # 反射检测和抑制
            "reflection": (
                True,
                ReflectionParameters(
                    mode=ReflectionMode.HIGHLIGHT_MASK,
                    bright_threshold=0.85,
                    low_saturation_threshold=0.15,
                    strength=0.35,
                    feather_radius=8.0,
                ).to_dict(),
            ),
            "dehalo": (False, DehaloParameters().to_dict()),
            "deblur": (False, DeblurOperator().default_parameters().to_dict()),
            "sharpen": (
                True,
                SharpenParameters(
                    radius=0.6,
                    amount=0.10,
                    threshold=0.015,
                    highlight_protection=0.6,
                    shadow_protection=0.5,
                ).to_dict(),
            ),
            # 多帧融合提示：UI 应建议用户提供多角度照片
        },
    }
