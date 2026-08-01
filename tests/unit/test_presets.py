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
