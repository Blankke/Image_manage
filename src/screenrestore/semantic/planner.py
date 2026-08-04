"""恢复计划器：根据 SceneContext + SceneHint 自动决定算子配置。

用户明确的 scene/preset 是权威提示 (authoritative hint)：
- 自动分类仅作为建议
- scene-aware safety gating 防止语义诊断误判破坏 preset 安全规则

输入 SceneContext + SceneHint → 输出 RestorationPlan。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from screenrestore.core.presets import PresetId

from .context import SceneContext
from .scene_classifier import (
    SCENE_ARTWORK,
    SCENE_CINEMA,
    SCENE_DISPLAY,
    SCENE_DOCUMENT,
    SCENE_GLOSSY_ARTWORK,
    SCENE_OTHER,
)

# ── Safety gate: 哪些场景禁止自动开启哪些算子 ────────────
# 格式: scene_type → set of operator_ids that must NOT be auto-enabled
AUTO_DISABLE_GATE: dict[str, set[str]] = {
    SCENE_ARTWORK: {"demoire", "dehalo", "clahe"},
    SCENE_GLOSSY_ARTWORK: {"demoire", "clahe"},
    SCENE_CINEMA: {"reflection", "clahe"},
    SCENE_DOCUMENT: {"dehalo"},
}

# 哪些场景仅在有极强证据时才允许自动开启
STRICT_GATE: dict[str, dict[str, float]] = {
    # scene_type → {operator_id: minimum_evidence_threshold}
    SCENE_CINEMA: {"demoire": 0.85},  # 仅极强 screen-lattice evidence
    SCENE_ARTWORK: {"reflection": 0.90},  # 仅确实看到反光
}


@dataclass
class OperatorRecommendation:
    """单个算子的推荐配置。"""

    enabled: bool
    strength: float  # 0~1 相对强度
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class RestorationPlan:
    """恢复计划：每个算子的推荐配置 + 推荐 preset。"""

    scene_type: str
    scene_confidence: float
    recommended_preset: PresetId
    user_preset: PresetId | None = None  # 用户明确选择的 preset
    operators: dict[str, OperatorRecommendation] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "scene_confidence": self.scene_confidence,
            "recommended_preset": self.recommended_preset.value,
            "user_preset": self.user_preset.value if self.user_preset else None,
            "operators": {
                op_id: {
                    "enabled": rec.enabled,
                    "strength": rec.strength,
                    "params": rec.params,
                    "reason": rec.reason,
                }
                for op_id, rec in self.operators.items()
            },
            "notes": self.notes,
        }


class RestorationPlanner:
    """根据语义分析结果生成恢复计划。

    这是语义层和现有 ImagePipeline 之间的桥梁：
    SceneContext → 算子推荐 → 用户确认 → Pipeline 配置。
    """

    def plan(
        self,
        ctx: SceneContext,
        *,
        scene_hint: PresetId | None = None,
    ) -> RestorationPlan:
        """生成恢复计划。

        Args:
            ctx: 语义分析结果
            scene_hint: 用户明确选择的 preset (权威提示)
        """
        # 用户 preset 优先于自动分类
        effective_preset = scene_hint if scene_hint is not None else self._map_to_preset(ctx.scene_type)
        effective_scene = self._preset_to_scene(effective_preset)
        confidence = ctx.scene_confidence

        # 用 effective scene 覆盖 ctx.scene_type（用户选择优先）
        plan = RestorationPlan(
            scene_type=effective_scene,
            scene_confidence=confidence,
            recommended_preset=effective_preset,
            user_preset=scene_hint,
        )

        # 根据 effective scene 和退化指标生成算子推荐
        plan.operators.update(self._recommend_geometry(effective_scene, ctx))
        plan.operators.update(self._recommend_demoire(effective_scene, ctx))
        plan.operators.update(self._recommend_reflection(effective_scene, ctx))
        plan.operators.update(self._recommend_dehalo(effective_scene, ctx))
        plan.operators.update(self._recommend_exposure(effective_scene, ctx))
        plan.operators.update(self._recommend_denoise(effective_scene, ctx))
        plan.operators.update(self._recommend_sharpen(effective_scene, ctx))
        plan.operators.update(self._recommend_clahe(effective_scene, ctx))
        plan.operators.update(self._recommend_illumination(effective_scene, ctx))

        # 安全 gating: 删除不应自动开启的算子推荐
        self._apply_safety_gating(effective_scene, plan)

        return plan

    # ── 安全 gating ─────────────────────────────────────

    def _apply_safety_gating(self, scene_type: str, plan: RestorationPlan) -> None:
        """强制关闭对当前场景不安全的自动推荐。"""
        auto_disable = AUTO_DISABLE_GATE.get(scene_type, set())
        strict = STRICT_GATE.get(scene_type, {})

        for op_id in list(plan.operators.keys()):
            if op_id in auto_disable:
                del plan.operators[op_id]
                plan.notes.append(f"安全门: {op_id} 在 {scene_type} 场景禁止自动开启")
            elif op_id in strict:
                rec = plan.operators[op_id]
                threshold = strict[op_id]
                if rec.strength < threshold:
                    del plan.operators[op_id]
                    plan.notes.append(
                        f"严格门: {op_id} 证据 {rec.strength:.2f} < 阈值 {threshold}"
                    )

    # ── preset ↔ scene 映射 ──────────────────────────────

    @staticmethod
    def _map_to_preset(scene_type: str) -> PresetId:
        mapping = {
            SCENE_DISPLAY: PresetId.DISPLAY,
            SCENE_CINEMA: PresetId.CINEMA,
            SCENE_ARTWORK: PresetId.ARTWORK,
            SCENE_GLOSSY_ARTWORK: PresetId.GLOSSY_ARTWORK,
            SCENE_DOCUMENT: PresetId.DOCUMENT,
        }
        return mapping.get(scene_type, PresetId.CUSTOM)

    @staticmethod
    def _preset_to_scene(preset: PresetId) -> str:
        mapping = {
            PresetId.DISPLAY: SCENE_DISPLAY,
            PresetId.CINEMA: SCENE_CINEMA,
            PresetId.ARTWORK: SCENE_ARTWORK,
            PresetId.GLOSSY_ARTWORK: SCENE_GLOSSY_ARTWORK,
            PresetId.DOCUMENT: SCENE_DOCUMENT,
            PresetId.ELECTRONIC_POSTER: SCENE_DISPLAY,
            PresetId.LED: SCENE_DISPLAY,
        }
        return mapping.get(preset, SCENE_OTHER)

    # ── preset 映射 ──────────────────────────────────────

    @staticmethod
    def _map_to_preset(scene_type: str) -> PresetId:
        mapping = {
            SCENE_DISPLAY: PresetId.DISPLAY,
            SCENE_CINEMA: PresetId.CINEMA,
            SCENE_ARTWORK: PresetId.ARTWORK,
            SCENE_GLOSSY_ARTWORK: PresetId.GLOSSY_ARTWORK,
            SCENE_DOCUMENT: PresetId.DOCUMENT,
        }
        return mapping.get(scene_type, PresetId.CUSTOM)

    # ── 各算子推荐逻辑 (scene_type 来自用户选择或分类) ──

    def _recommend_geometry(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        # perspective 证据来自目标 polygon 存在性，而非 illumination
        has_target = ctx.has_target()
        if scene_type in (SCENE_ARTWORK, SCENE_GLOSSY_ARTWORK):
            return {
                "geometry": OperatorRecommendation(
                    enabled=True, strength=1.0,
                    reason="艺术品/画作场景需要透视校正",
                ),
                "lens_distortion": OperatorRecommendation(
                    enabled=True, strength=0.5,
                    reason="画作拍摄常有镜头畸变",
                ),
            }
        if scene_type == SCENE_DISPLAY:
            return {
                "geometry": OperatorRecommendation(
                    enabled=True, strength=1.0 if has_target else 0.5,
                    reason=f"显示器场景透视校正 (目标={'已定位' if has_target else '待定位'})",
                ),
            }
        return {}

    def _recommend_demoire(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        # 计算真实的摩尔纹面积比
        moire_ratio = 0.0
        if "moire" in ctx.artifact_masks:
            mask = ctx.artifact_masks["moire"]
            total_pixels = mask.size
            moire_pixels = float((mask > 127).sum()) if mask.dtype == np.uint8 else float(mask.sum())
            moire_ratio = moire_pixels / max(total_pixels, 1)

        # screen lattice evidence
        screen_conf = ctx.properties.get("screen_lattice_confidence", 0.0)
        moire_conf = max(moire_ratio * 2.5, screen_conf)

        # DETECTOR_INCONSISTENT: 高 moire_ratio 但低 screen_lattice_confidence
        inconsistent = (moire_ratio > 0.3 and screen_conf < 0.2)

        if scene_type == SCENE_DISPLAY:
            if inconsistent:
                return {
                    "demoire": OperatorRecommendation(
                        enabled=True,
                        strength=0.3,  # 降级：不一致时保守
                        reason=(
                            f"DETECTOR_INCONSISTENT: moire_ratio={moire_ratio:.3f} "
                            f"but screen_lattice={screen_conf:.2f} — 降级运行"
                        ),
                        params={"mode": "chroma"},
                    ),
                }
            return {
                "demoire": OperatorRecommendation(
                    enabled=True,
                    strength=min(0.5 + moire_conf, 1.0),
                    reason=f"显示器场景, moire_ratio={moire_ratio:.3f} screen_conf={screen_conf:.2f}",
                    params={"mode": "joint_edge_aware"},
                ),
            }
        if scene_type == SCENE_ARTWORK:
            return {}
        if scene_type == SCENE_CINEMA:
            if screen_conf > 0.7:
                return {
                    "demoire": OperatorRecommendation(
                        enabled=True, strength=min(screen_conf, 0.5),
                        reason=f"电影院检测到极强 screen evidence ({screen_conf:.2f})",
                        params={"mode": "chroma"},
                    ),
                }
            return {}
        if inconsistent:
            return {}  # 不一致时不在非 DISPLAY 场景启用
        return {}

    def _recommend_reflection(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        reflection_ratio = 0.0
        if "reflection" in ctx.artifact_masks:
            mask = ctx.artifact_masks["reflection"]
            total = mask.size
            pixels = float((mask > 127).sum()) if mask.dtype == np.uint8 else float(mask.sum())
            reflection_ratio = pixels / max(total, 1)

        highlight = ctx.properties.get("highlight_clipping", 0)

        if scene_type == SCENE_GLOSSY_ARTWORK:
            saturated = highlight > 0.001
            return {
                "reflection": OperatorRecommendation(
                    enabled=True,
                    strength=min(reflection_ratio * 3 + 0.3, 1.0),
                    reason=(
                        f"覆膜/玻璃场景, 反光面积={reflection_ratio:.2%}"
                        + (", 存在饱和反光 → 建议多帧" if saturated else "")
                    ),
                ),
            }
        if scene_type == SCENE_ARTWORK:
            # 仅当确实有可见反光时
            if reflection_ratio > 0.02:
                return {
                    "reflection": OperatorRecommendation(
                        enabled=True, strength=min(reflection_ratio * 2 + 0.1, 0.4),
                        reason=f"画作检测到轻微反光 ({reflection_ratio:.2%})",
                    ),
                }
            return {}
        # CINEMA / DISPLAY / DOCUMENT: 安全门阻止，此处不推荐
        return {}

    def _recommend_dehalo(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        if scene_type == SCENE_CINEMA:
            return {
                "dehalo": OperatorRecommendation(
                    enabled=True, strength=0.3,
                    reason="电影院场景轻度去光晕 (auto_gate)",
                    params={"auto_gate": True},
                ),
            }
        # ARTWORK: 安全门阻止
        return {}

    def _recommend_exposure(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        black_r = ctx.properties.get("black_level_r", 0)
        if scene_type == SCENE_CINEMA and black_r > 0.02:
            return {
                "exposure": OperatorRecommendation(
                    enabled=True, strength=0.5,
                    reason=f"电影院黑位抬升 R={black_r:.3f}",
                    params={"auto_black_level_strength": min(black_r * 15, 0.8)},
                ),
            }
        return {}

    def _recommend_denoise(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        noise = ctx.properties.get("noise_estimate", 0)
        if noise > 0.3:
            return {
                "denoise": OperatorRecommendation(
                    enabled=True, strength=min(noise * 2, 1.0),
                    reason=f"噪声水平={noise:.2f}",
                ),
            }
        if scene_type == SCENE_CINEMA:
            return {
                "denoise": OperatorRecommendation(
                    enabled=True, strength=0.15,
                    reason="电影院场景轻度去噪",
                    params={"mode": "luma_chroma", "chroma_strength": 3.0},
                ),
            }
        return {}

    def _recommend_sharpen(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        blur = ctx.properties.get("blur_estimate", 0)
        if scene_type == SCENE_ARTWORK:
            return {
                "sharpen": OperatorRecommendation(
                    enabled=True, strength=min(blur * 0.4 + 0.05, 0.15),
                    reason="艺术品轻度锐化, 保护高光/暗部",
                    params={"highlight_protection": 0.6, "shadow_protection": 0.5},
                ),
            }
        if scene_type == SCENE_CINEMA:
            # cinema 不自作主张锐化电影柔焦
            return {}
        if blur > 0.5:
            return {
                "sharpen": OperatorRecommendation(
                    enabled=True, strength=min(blur * 0.6, 0.6),
                    reason=f"模糊度={blur:.2f}",
                ),
            }
        return {}

    def _recommend_clahe(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        # ARTWORK / GLOSSY_ARTWORK / CINEMA: 安全门阻止
        if scene_type == SCENE_DOCUMENT:
            return {
                "clahe": OperatorRecommendation(
                    enabled=True, strength=0.34,
                    reason="文档场景适度增强局部对比度",
                ),
            }
        return {}

    def _recommend_illumination(
        self, scene_type: str, ctx: SceneContext,
    ) -> dict[str, OperatorRecommendation]:
        illum = ctx.properties.get("illumination_gradient", 0)
        if scene_type == SCENE_ARTWORK and illum > 0.1:
            return {
                "illumination": OperatorRecommendation(
                    enabled=True, strength=min(illum * 2, 0.2),
                    reason=f"照明不均={illum:.2f} (轻度校正)",
                ),
            }
        return {}
