"""恢复计划器：根据 SceneContext 自动决定算子配置。

输入 SceneContext → 输出 RestorationPlan (每个算子的推荐启用状态和参数)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    operators: dict[str, OperatorRecommendation] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "scene_confidence": self.scene_confidence,
            "recommended_preset": self.recommended_preset.value,
            "operators": {
                op_id: {
                    "enabled": rec.enabled,
                    "strength": rec.strength,
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

    def plan(self, ctx: SceneContext) -> RestorationPlan:
        """生成恢复计划。"""
        scene_type = ctx.scene_type
        confidence = ctx.scene_confidence

        preset = self._map_to_preset(scene_type)
        plan = RestorationPlan(
            scene_type=scene_type,
            scene_confidence=confidence,
            recommended_preset=preset,
        )

        # 根据场景类型和退化指标生成算子推荐
        plan.operators.update(self._recommend_geometry(ctx))
        plan.operators.update(self._recommend_demoire(ctx))
        plan.operators.update(self._recommend_reflection(ctx))
        plan.operators.update(self._recommend_dehalo(ctx))
        plan.operators.update(self._recommend_exposure(ctx))
        plan.operators.update(self._recommend_denoise(ctx))
        plan.operators.update(self._recommend_sharpen(ctx))
        plan.operators.update(self._recommend_clahe(ctx))
        plan.operators.update(self._recommend_illumination(ctx))

        return plan

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

    # ── 各算子推荐逻辑 ───────────────────────────────────

    def _recommend_geometry(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        perspective = ctx.properties.get("illumination_gradient", 0)
        if ctx.scene_type in (SCENE_ARTWORK, SCENE_GLOSSY_ARTWORK):
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
        if ctx.scene_type == SCENE_DISPLAY:
            return {
                "geometry": OperatorRecommendation(
                    enabled=perspective > 0.3, strength=min(perspective * 2, 1.0),
                    reason=f"透视强度={perspective:.2f}",
                ),
            }
        return {}

    def _recommend_demoire(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        moire_prob = ctx.properties.get("moire_probability", 0)
        if "moire" in ctx.artifact_masks:
            moire_area = float(ctx.artifact_masks["moire"].sum()) / 255.0
            moire_prob = max(moire_prob, min(moire_area / 10000.0, 1.0))

        if ctx.scene_type == SCENE_DISPLAY:
            return {
                "demoire": OperatorRecommendation(
                    enabled=True,
                    strength=min(0.5 + moire_prob, 1.0),
                    reason=f"显示器场景, 摩尔纹概率={moire_prob:.2f}",
                    params={"mode": "joint_edge_aware"},
                ),
            }
        if ctx.scene_type == SCENE_ARTWORK:
            return {
                "demoire": OperatorRecommendation(
                    enabled=False, strength=0.0,
                    reason="画作无须去摩尔纹",
                ),
            }
        if moire_prob > 0.4:
            return {
                "demoire": OperatorRecommendation(
                    enabled=True, strength=moire_prob,
                    reason=f"检测到摩尔纹 (置信度={moire_prob:.2f})",
                ),
            }
        return {}

    def _recommend_reflection(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        reflection_area = 0.0
        if "reflection" in ctx.artifact_masks:
            reflection_area = float(ctx.artifact_masks["reflection"].sum()) / 255.0 / (
                ctx.artifact_masks["reflection"].size
            )

        highlight = ctx.properties.get("highlight_clipping", 0)

        if ctx.scene_type == SCENE_GLOSSY_ARTWORK:
            saturated = highlight > 0.001
            return {
                "reflection": OperatorRecommendation(
                    enabled=True,
                    strength=min(reflection_area * 3 + 0.3, 1.0),
                    reason=(
                        f"覆膜/玻璃场景, 反光面积={reflection_area:.1%}"
                        + (", 存在饱和反光 → 建议多帧" if saturated else "")
                    ),
                ),
            }
        if reflection_area > 0.03:
            return {
                "reflection": OperatorRecommendation(
                    enabled=True, strength=min(reflection_area * 5, 1.0),
                    reason=f"检测到反光区域 ({reflection_area:.1%})",
                ),
            }
        return {}

    def _recommend_dehalo(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        if ctx.scene_type == SCENE_CINEMA:
            return {
                "dehalo": OperatorRecommendation(
                    enabled=True, strength=0.3,
                    reason="电影院场景轻度去光晕 (auto_gate)",
                    params={"auto_gate": True},
                ),
            }
        if ctx.scene_type == SCENE_ARTWORK:
            return {
                "dehalo": OperatorRecommendation(
                    enabled=False, strength=0.0,
                    reason="画作不处理光晕 (保护原作意图)",
                ),
            }
        return {}

    def _recommend_exposure(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        black_r = ctx.properties.get("black_level_r", 0)
        if ctx.scene_type == SCENE_CINEMA and black_r > 0.02:
            return {
                "exposure": OperatorRecommendation(
                    enabled=True, strength=0.5,
                    reason=f"电影院黑位抬升 R={black_r:.3f}",
                    params={"auto_black_level_strength": min(black_r * 20, 1.0)},
                ),
            }
        return {}

    def _recommend_denoise(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        noise = ctx.properties.get("noise_estimate", 0)
        if noise > 0.3:
            return {
                "denoise": OperatorRecommendation(
                    enabled=True, strength=min(noise * 2, 1.0),
                    reason=f"噪声水平={noise:.2f}",
                ),
            }
        if ctx.scene_type == SCENE_CINEMA:
            return {
                "denoise": OperatorRecommendation(
                    enabled=True, strength=0.15,
                    reason="电影院场景轻度去噪",
                    params={"mode": "luma_chroma", "chroma_strength": 3.0},
                ),
            }
        return {}

    def _recommend_sharpen(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        blur = ctx.properties.get("blur_estimate", 0)
        if ctx.scene_type == SCENE_ARTWORK:
            return {
                "sharpen": OperatorRecommendation(
                    enabled=True, strength=min(blur * 0.5 + 0.05, 0.15),
                    reason="艺术品轻度锐化, 保护高光/暗部",
                    params={"highlight_protection": 0.6, "shadow_protection": 0.5},
                ),
            }
        if blur > 0.5:
            return {
                "sharpen": OperatorRecommendation(
                    enabled=True, strength=min(blur * 0.8, 0.8),
                    reason=f"模糊度={blur:.2f}",
                ),
            }
        return {}

    def _recommend_clahe(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        if ctx.scene_type in (SCENE_ARTWORK, SCENE_GLOSSY_ARTWORK):
            return {
                "clahe": OperatorRecommendation(
                    enabled=False, strength=0.0,
                    reason="艺术品场景关闭 CLAHE (保护原作对比度/防放大反光)",
                ),
            }
        return {}

    def _recommend_illumination(self, ctx: SceneContext) -> dict[str, OperatorRecommendation]:
        illum = ctx.properties.get("illumination_gradient", 0)
        if ctx.scene_type == SCENE_ARTWORK and illum > 0.1:
            return {
                "illumination": OperatorRecommendation(
                    enabled=True, strength=min(illum * 2, 0.2),
                    reason=f"照明不均={illum:.2f} (轻度校正)",
                ),
            }
        return {}
