"""经典流水线构造和场景预设。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from screenrestore.operators import (
    BandingOperator,
    ClaheOperator,
    DeblurOperator,
    DemoireOperator,
    DenoiseOperator,
    ExposureOperator,
    GeometryOperator,
    IlluminationOperator,
    LensDistortionOperator,
    MeshWarpOperator,
    ModelPluginOperator,
    OrientationOperator,
    ReflectionOperator,
    ResizeOperator,
    SharpenOperator,
    WhiteBalanceOperator,
)
from screenrestore.operators.banding import BandingParameters
from screenrestore.operators.demoire import DemoireParameters
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
    CUSTOM = "custom"


PRESET_NAMES = {
    PresetId.DISPLAY: "显示器",
    PresetId.ELECTRONIC_POSTER: "电子海报",
    PresetId.CINEMA: "电影院/投影",
    PresetId.LED: "LED 大屏",
    PresetId.DOCUMENT: "文档/PPT",
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
            DenoiseOperator(),
            WhiteBalanceOperator(),
            ExposureOperator(),
            ClaheOperator(),
            IlluminationOperator(),
            ReflectionOperator(),
            DeblurOperator(),
            SharpenOperator(),
            ModelPluginOperator(),
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
        "model_plugin",
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


def _preset_definitions() -> dict[PresetId, dict[str, tuple[bool, dict[str, Any]]]]:
    """集中声明预设差异，所有值仍可由用户覆盖。"""

    display = {
        "banding": (True, BandingParameters(strength=0.65).to_dict()),
        "demoire": (True, DemoireParameters(strength=0.42).to_dict()),
        "denoise": (True, DenoiseParameters(strength=0.22).to_dict()),
        "white_balance": (True, WhiteBalanceParameters().to_dict()),
        "clahe": (True, ClaheParameters(strength=0.28).to_dict()),
        "illumination": (True, IlluminationParameters(strength=0.12).to_dict()),
        "reflection": (False, ReflectionOperator().default_parameters().to_dict()),
        "deblur": (False, DeblurOperator().default_parameters().to_dict()),
        "sharpen": (True, SharpenParameters(amount=0.55).to_dict()),
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
                    strength=0.0,
                    broad_haze_strength=1.8,
                    broad_haze_scale=95.0,
                    black_level_quantile=0.03,
                    max_haze_correction=0.1,
                ).to_dict(),
            ),
            "demoire": (True, DemoireParameters(strength=0.18).to_dict()),
            "denoise": (
                True,
                DenoiseParameters(
                    mode=DenoiseMode.LUMA_CHROMA,
                    strength=0.1,
                    luma_strength=1.6,
                    chroma_strength=4.0,
                ).to_dict(),
            ),
            "white_balance": (
                True,
                WhiteBalanceParameters(
                    mode=WhiteBalanceMode.GRAY_WORLD,
                    max_gain=1.25,
                    strength=0.18,
                ).to_dict(),
            ),
            "exposure": (
                True,
                ExposureParameters(
                    contrast=0.1,
                    shadows=0.04,
                    blacks=-0.04,
                    saturation=0.08,
                    temperature=-0.025,
                ).to_dict(),
            ),
            "clahe": (True, ClaheParameters(strength=0.18, clip_limit=1.25).to_dict()),
            "illumination": (False, IlluminationParameters(strength=0.05).to_dict()),
            "reflection": (
                True,
                ReflectionParameters(
                    mode=ReflectionMode.GRADIENT_DCT,
                    strength=0.72,
                    gradient_threshold=0.015,
                ).to_dict(),
            ),
            "sharpen": (
                True,
                SharpenParameters(
                    radius=0.9,
                    amount=0.55,
                    threshold=0.012,
                    highlight_protection=0.45,
                    shadow_protection=0.4,
                ).to_dict(),
            ),
        },
        PresetId.LED: {
            **display,
            "banding": (True, BandingParameters(strength=0.8).to_dict()),
            "demoire": (True, DemoireParameters(strength=0.68, chroma_radius=3.0).to_dict()),
            "denoise": (True, DenoiseParameters(strength=0.3).to_dict()),
            "sharpen": (True, SharpenParameters(amount=0.42).to_dict()),
        },
        PresetId.DOCUMENT: {
            **display,
            "banding": (True, BandingParameters(strength=0.4).to_dict()),
            "demoire": (True, DemoireParameters(strength=0.2).to_dict()),
            "denoise": (True, DenoiseParameters(strength=0.15).to_dict()),
            "white_balance": (True, WhiteBalanceParameters().to_dict()),
            "clahe": (True, ClaheParameters(strength=0.34, clip_limit=1.7).to_dict()),
            "illumination": (True, IlluminationParameters(strength=0.4).to_dict()),
            "sharpen": (True, SharpenParameters(amount=0.46).to_dict()),
        },
    }
