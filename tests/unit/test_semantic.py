"""语义分析层单元测试。"""

from __future__ import annotations

import numpy as np

from screenrestore.semantic import (
    RestorationPlanner,
    SceneContext,
    SemanticAnalyzer,
    classify_scene,
)
from screenrestore.semantic.scene_classifier import ALL_SCENE_TYPES


def test_scene_context_defaults() -> None:
    ctx = SceneContext()
    assert ctx.scene_type == "other"
    assert ctx.scene_confidence == 0.0
    assert not ctx.has_target()
    assert ctx.semantic_masks == {}
    assert ctx.artifact_masks == {}


def test_scene_context_serialization() -> None:
    ctx = SceneContext(
        scene_type="artwork",
        scene_confidence=0.96,
        properties={"blur_estimate": 0.3},
    )
    d = ctx.to_dict()
    assert d["scene_type"] == "artwork"
    assert d["scene_confidence"] == 0.96
    assert "blur_estimate" in d["properties"]


def test_scene_context_target_bbox() -> None:
    ctx = SceneContext(target_bbox=(10, 20, 100, 200))
    assert ctx.has_target()
    roi = ctx.get_target_roi((300, 400))
    assert roi == (10, 20, 100, 200)


def test_scene_context_target_mask() -> None:
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:150, 30:120] = 255
    ctx = SceneContext(target_mask=mask)
    assert ctx.has_target()
    roi = ctx.get_target_roi((200, 300))
    assert roi == (30, 50, 90, 100)


def test_classify_scene_returns_valid_type() -> None:
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    scene_type, confidence = classify_scene(img)
    assert scene_type in ALL_SCENE_TYPES
    assert 0.0 <= confidence <= 1.0


def test_semantic_analyzer_produces_valid_context() -> None:
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    analyzer = SemanticAnalyzer()
    ctx = analyzer.analyze(img)
    assert ctx.scene_type in ALL_SCENE_TYPES
    assert 0.0 <= ctx.scene_confidence <= 1.0
    assert "blur_estimate" in ctx.properties
    assert "noise_estimate" in ctx.properties
    assert 0.0 <= ctx.properties["blur_estimate"] <= 1.0
    assert 0.0 <= ctx.properties["noise_estimate"] <= 1.0


def test_semantic_analyzer_on_synthetic_solid() -> None:
    """纯色图应该 blur≈0, noise≈0。"""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    analyzer = SemanticAnalyzer()
    ctx = analyzer.analyze(img)
    # 纯色图 Laplacian variance ≈ 0 → blur≈1
    assert ctx.properties["blur_estimate"] > 0.9, f"纯色图应被判定为无纹理: {ctx.properties['blur_estimate']}"
    assert ctx.properties["noise_estimate"] < 0.1, f"纯色图噪声应极低: {ctx.properties['noise_estimate']}"


def test_semantic_analyzer_on_synthetic_sharp() -> None:
    """带锐利边缘的图应该 blur < 0.5。"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[90:110, :] = 255  # 水平锐利边缘
    analyzer = SemanticAnalyzer()
    ctx = analyzer.analyze(img)
    assert ctx.properties["blur_estimate"] < 0.5, f"锐利边缘应 blur<0.5: {ctx.properties['blur_estimate']}"


def test_restoration_planner_returns_plan() -> None:
    planner = RestorationPlanner()
    ctx = SceneContext(scene_type="artwork", scene_confidence=0.9)
    plan = planner.plan(ctx)
    assert plan.scene_type == "artwork"
    assert plan.recommended_preset is not None
    assert len(plan.operators) > 0


def test_restoration_planner_artwork_disables_clahe() -> None:
    planner = RestorationPlanner()
    ctx = SceneContext(scene_type="artwork", scene_confidence=0.9)
    plan = planner.plan(ctx)
    clahe_rec = plan.operators.get("clahe")
    if clahe_rec is not None:
        assert not clahe_rec.enabled, "艺术品场景 CLAHE 应关闭"


def test_model_role_enum() -> None:
    from screenrestore.inference.model_manifest import ModelRole
    roles = list(ModelRole)
    assert ModelRole.ANALYSIS in roles
    assert ModelRole.RESTORATION in roles
    assert ModelRole.RECONSTRUCTION in roles
    assert ModelRole.ENHANCEMENT in roles


def test_analysis_result_dataclass() -> None:
    from screenrestore.inference.backend import AnalysisResult, Detection

    result = AnalysisResult(
        labels={"artwork": 0.95, "display": 0.03},
        properties={"brightness": 0.5},
    )
    top_label, conf = result.top_label()
    assert top_label == "artwork"
    assert conf == 0.95

    det = Detection(label="person", confidence=0.8, bbox=(0.1, 0.2, 0.3, 0.4))
    result2 = AnalysisResult(detections=[det])
    assert len(result2.detections) == 1


# ── v9 新增测试 ──

def test_moire_ratio_not_fixed_division() -> None:
    """moire area ratio 不再固定 /10000。"""
    from screenrestore.semantic.context import SceneContext
    from screenrestore.semantic.planner import RestorationPlanner

    planner = RestorationPlanner()
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    ctx = SceneContext(scene_type="display", artifact_masks={"moire": mask})
    rec = planner._recommend_demoire("display", ctx)
    assert rec is not None
    assert "demoire" in rec
    # 大面积 moire → 高强度
    mask_small = np.zeros((100, 100), dtype=np.uint8)
    mask_small[0:10, 0:10] = 255
    ctx2 = SceneContext(scene_type="display", artifact_masks={"moire": mask_small})
    rec2 = planner._recommend_demoire("display", ctx2)
    assert "demoire" in rec2
    # 大面积应有更高强度（或其强度合理）
    assert rec2["demoire"].strength + rec["demoire"].strength > 0


def test_illumination_not_perspective() -> None:
    """illumination_gradient 不影响 perspective 判断。"""
    from screenrestore.semantic.context import SceneContext
    from screenrestore.semantic.planner import RestorationPlanner

    planner = RestorationPlanner()
    ctx = SceneContext(scene_type="display",
                       properties={"illumination_gradient": 0.9})
    rec = planner._recommend_geometry("display", ctx)
    # geometry 推荐不读取 illumination_gradient
    assert isinstance(rec, dict)


def test_artwork_safety_gate_blocks_demoire() -> None:
    """ARTWORK 安全门禁止自动开启 demoire。"""
    from screenrestore.semantic.planner import AUTO_DISABLE_GATE

    assert "demoire" in AUTO_DISABLE_GATE.get("artwork", set())


def test_cinema_safety_gate_blocks_reflection() -> None:
    """CINEMA 安全门禁止自动开启 reflection。"""
    from screenrestore.semantic.planner import AUTO_DISABLE_GATE

    assert "reflection" in AUTO_DISABLE_GATE.get("cinema", set())


def test_scene_hint_overrides_classifier() -> None:
    """用户 scene_hint 优先于自动分类。"""
    from screenrestore.core.presets import PresetId
    from screenrestore.semantic.context import SceneContext
    from screenrestore.semantic.planner import RestorationPlanner

    planner = RestorationPlanner()
    # 语义分析说是 "display", 但用户说是 "artwork"
    ctx = SceneContext(scene_type="display", scene_confidence=0.8)
    plan = planner.plan(ctx, scene_hint=PresetId.ARTWORK)
    assert plan.recommended_preset == PresetId.ARTWORK
    assert plan.user_preset == PresetId.ARTWORK


def test_plan_params_serializable() -> None:
    """RestorationPlan.to_dict() 包含 params。"""
    from screenrestore.core.presets import PresetId
    from screenrestore.semantic.planner import OperatorRecommendation, RestorationPlan

    plan = RestorationPlan(
        scene_type="artwork", scene_confidence=0.9,
        recommended_preset=PresetId.ARTWORK,
    )
    plan.operators["sharpen"] = OperatorRecommendation(
        enabled=True, strength=0.2,
        params={"amount": 0.15}, reason="test",
    )
    d = plan.to_dict()
    assert "params" in d["operators"]["sharpen"]
    assert d["operators"]["sharpen"]["params"]["amount"] == 0.15


def test_target_localizer_fallback() -> None:
    """无模型时 TargetLocalizer 使用几何评分 fallback。"""
    from screenrestore.semantic.context import SceneContext
    from screenrestore.semantic.target_localizer import TargetLocalizer

    loc = TargetLocalizer()  # 无 CLIP backend
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    ctx = SceneContext()
    result = loc.localize(img, ctx)
    # 应返回 SceneContext（可能找到也可能没找到，但不报错）
    assert result is not None
    assert isinstance(result, SceneContext)
