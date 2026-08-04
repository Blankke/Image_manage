"""预设顺序和覆盖行为测试。"""

from __future__ import annotations

from screenrestore.core.presets import (
    PresetId,
    ProcessingMode,
    apply_preset,
    apply_processing_mode,
    build_default_pipeline,
)


def test_default_pipeline_has_safe_fixed_boundaries() -> None:
    pipeline = build_default_pipeline()
    ids = [state.operator.id for state in pipeline.states]
    assert ids[:4] == ["orientation", "lens_distortion", "geometry", "mesh_warp"]
    assert ids[-1] == "resize"
    assert not pipeline.state("geometry").operator.reorderable
    assert not pipeline.state("lens_distortion").operator.reorderable
    assert not pipeline.state("mesh_warp").operator.reorderable
    assert not pipeline.state("lens_distortion").enabled
    assert not pipeline.state("mesh_warp").enabled
    assert pipeline.state("dehalo").enabled
    assert not pipeline.state("resize").operator.reorderable


def test_cinema_fidelity_only_enables_evidence_gated_black_level_tone() -> None:
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.CINEMA)
    assert not pipeline.state("illumination").enabled
    assert not pipeline.state("clahe").enabled
    assert not pipeline.state("white_balance").enabled
    exposure = pipeline.state("exposure")
    assert exposure.enabled
    exposure_values = exposure.params.to_dict()
    assert exposure_values["exposure"] == 0
    assert exposure_values["gamma"] == 1
    assert exposure_values["auto_black_level_strength"] > 0
    assert exposure_values["auto_black_contrast"] > 0
    assert pipeline.state("banding").params.to_dict()["broad_haze_strength"] > 0
    reflection = pipeline.state("reflection")
    assert not reflection.enabled
    assert reflection.params.to_dict()["mode"] == "gradient_dct_experimental"


def test_display_preset_enables_only_evidence_gated_white_background() -> None:
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.DISPLAY)
    exposure = pipeline.state("exposure")
    assert exposure.enabled
    values = exposure.params.to_dict()
    assert values["exposure"] == 0
    assert values["auto_black_level_strength"] == 0
    assert values["auto_white_background_strength"] == 1


def test_ai_enhanced_mode_disables_classic_sharpen_when_model_is_enabled() -> None:
    pipeline = build_default_pipeline()
    pipeline.state("enhancement_model").enabled = True
    pipeline.state("sharpen").enabled = True
    apply_processing_mode(pipeline, ProcessingMode.AI_ENHANCED)
    assert not pipeline.state("sharpen").enabled
    apply_processing_mode(pipeline, ProcessingMode.FIDELITY)
    assert not pipeline.state("enhancement_model").enabled


# ── 新增 ARTWORK / GLOSSY_ARTWORK 测试 ──


def test_artwork_preset_disables_destructive_operators() -> None:
    """艺术品预设：关闭 CLAHE、Dehalo、Demoire，保护原作意图。"""
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.ARTWORK)

    assert not pipeline.state("clahe").enabled, "CLAHE 会改变画作局部对比度"
    assert not pipeline.state("dehalo").enabled, "Dehalo 可能误伤画作原有光晕"
    assert not pipeline.state("demoire").enabled, "画作不存在摩尔纹"
    assert not pipeline.state("reflection").enabled or pipeline.state("reflection").params.to_dict()["strength"] == 0.0
    assert not pipeline.state("deblur").enabled


def test_artwork_preset_prioritizes_color_fidelity() -> None:
    """艺术品预设：色彩忠实度优先于锐度。"""
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.ARTWORK)

    wb = pipeline.state("white_balance")
    assert wb.enabled
    wb_values = wb.params.to_dict()
    assert wb_values["max_gain"] <= 1.15, "色彩增益应保守"

    sharpen = pipeline.state("sharpen")
    sharpen_values = sharpen.params.to_dict()
    assert sharpen_values["amount"] <= 0.15, "锐化应极轻"
    assert sharpen_values["highlight_protection"] >= 0.5
    assert sharpen_values["shadow_protection"] >= 0.4

    exposure = pipeline.state("exposure")
    exp_values = exposure.params.to_dict()
    assert exp_values["auto_black_level_strength"] == 0, "不自动修改黑位"


def test_glossy_artwork_preset_enables_reflection() -> None:
    """覆膜反光预设：启用反光检测，关闭 CLAHE。"""
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.GLOSSY_ARTWORK)

    reflection = pipeline.state("reflection")
    assert reflection.enabled, "反光检测应启用"
    ref_values = reflection.params.to_dict()
    assert ref_values["mode"] == "highlight_mask"
    assert ref_values["strength"] > 0


def test_glossy_artwork_preset_disables_clahe() -> None:
    """覆膜反光预设：CLAHE 必须先于反光检测，否则放大反光。"""
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.GLOSSY_ARTWORK)

    assert not pipeline.state("clahe").enabled, (
        "CLAHE 会放大反光，必须在反光检测之前关闭"
    )


def test_glossy_artwork_preset_keeps_lens_distortion() -> None:
    """覆膜反光预设：塑料膜/玻璃常与镜头畸变叠加，应启用镜头校正。"""
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.GLOSSY_ARTWORK)

    assert pipeline.state("lens_distortion").enabled


def test_both_new_presets_exist_in_registry() -> None:
    """验证两个新 preset 已注册且可应用。"""
    from screenrestore.core.presets import PRESET_NAMES

    assert PresetId.ARTWORK in PRESET_NAMES
    assert PresetId.GLOSSY_ARTWORK in PRESET_NAMES
    assert "画作" in PRESET_NAMES[PresetId.ARTWORK]
    assert "反光" in PRESET_NAMES[PresetId.GLOSSY_ARTWORK]
