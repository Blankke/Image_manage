"""预设顺序和覆盖行为测试。"""

from __future__ import annotations

from screenrestore.core.presets import PresetId, apply_preset, build_default_pipeline


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
    assert not pipeline.state("resize").operator.reorderable


def test_cinema_preset_uses_guarded_tonal_recovery() -> None:
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PresetId.CINEMA)
    assert not pipeline.state("illumination").enabled
    assert pipeline.state("clahe").params.to_dict()["strength"] <= 0.2
    assert pipeline.state("white_balance").params.to_dict()["strength"] <= 0.2
    assert pipeline.state("banding").params.to_dict()["broad_haze_strength"] > 0
    reflection = pipeline.state("reflection")
    assert reflection.enabled
    assert reflection.params.to_dict()["mode"] == "gradient_dct_experimental"
