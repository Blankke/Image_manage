"""语义上下文数据结构。

SceneContext 是 SemanticAnalyzer 的输出，供 RestorationPlanner
和下游算子消费。所有掩码使用二值 uint8 (0/255) 或 float32 [0,1]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LocalizationCandidate:
    """单个定位候选四边形。

    Attributes:
        polygon: 四边形顶点 N×2，像素坐标 (float32)
        source: 候选来源标识 ("contour", "line", "profile", ...)
        runtime_score: 运行时评分 [0,1]（不含 GT 信息）
        geometry_score: 几何评分分量
        semantic_score: 语义评分分量（可选）
    """

    polygon: np.ndarray
    source: str = "contour"
    runtime_score: float = 0.0
    geometry_score: float = 0.0
    semantic_score: float = 0.0
    layer: str = "content"

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon": self.polygon.astype(float).tolist(),
            "source": self.source,
            "runtime_score": round(self.runtime_score, 6),
            "geometry_score": round(self.geometry_score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "layer": self.layer,
        }


@dataclass
class SceneContext:
    """场景语义分析结果。

    Attributes:
        scene_type: 场景分类标签 (display/cinema/artwork/glossy_artwork/document/other)
        scene_confidence: 场景分类置信度 [0,1]
        target_mask: 目标内容区域二值掩码 (H×W, uint8 0/255)，与输入图像同尺寸
        target_bbox: 目标边界框 (x, y, w, h)，像素坐标
        target_polygon: 目标多边形顶点 N×2，像素坐标
        semantic_masks: 语义区域掩码字典 {label: mask}
            - "person": 人物区域
            - "face": 人脸区域
            - "hair": 头发区域
            - "text": 文字区域
            - "fine_texture": 需要保护的细纹理区域
            - "flat_region": 平坦区域
        artifact_masks: 退化区域掩码字典 {label: mask}
            - "reflection": 反光区域
            - "moire": 摩尔纹区域
            - "halo": 光晕区域
            - "banding": 条带区域
            - "screen_region": 屏幕像素结构区域
        properties: 场景属性字典
            - "screen_frequency": 检测到的屏幕采样频率
            - "screen_orientation": 屏幕像素方向 (度)
            - "black_level_offset": 估计的黑位偏移 (R,G,B)
            - "illumination_gradient": 照明梯度强度
            - "blur_estimate": 模糊程度估计
            - "noise_estimate": 噪声水平估计
            - "highlight_clipping": 高光裁剪比例
            - "shadow_clipping": 暗部裁剪比例
    """

    scene_type: str = "other"
    scene_confidence: float = 0.0

    target_mask: np.ndarray | None = None
    target_bbox: tuple[int, int, int, int] | None = None
    target_polygon: np.ndarray | None = None
    outer_polygon: np.ndarray | None = None

    localization_status: str = "not_run"
    localization_confidence: float = 0.0
    localization_backend: str = ""
    localization_rejection_reasons: tuple[str, ...] = ()
    aspect_ratio: float | None = None
    aspect_confidence: float = 0.0

    # v11: 所有定位候选（含未选中者），用于诊断 generation vs ranking 失败
    localization_candidates: list[LocalizationCandidate] = field(default_factory=list)

    semantic_masks: dict[str, np.ndarray] = field(default_factory=dict)
    artifact_masks: dict[str, np.ndarray] = field(default_factory=dict)

    properties: dict[str, float] = field(default_factory=dict)

    def has_target(self) -> bool:
        """是否已检测到目标内容区域。"""
        return (
            self.target_mask is not None
            or self.target_polygon is not None
            or self.target_bbox is not None
        )

    def get_target_roi(self, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        """返回目标区域 (x,y,w,h)，若未检测到则返回全图。"""
        if self.target_bbox is not None:
            return self.target_bbox
        if self.target_mask is not None:
            rows = np.any(self.target_mask > 127, axis=1)
            cols = np.any(self.target_mask > 127, axis=0)
            if not rows.any() or not cols.any():
                return None
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
        h, w = image_shape[:2]
        return (0, 0, w, h)

    def to_dict(self) -> dict:
        """序列化为可 JSON 的诊断报告（不含大体积掩码）。"""
        return {
            "scene_type": self.scene_type,
            "scene_confidence": round(self.scene_confidence, 4),
            "has_target": self.has_target(),
            "target_bbox": list(self.target_bbox) if self.target_bbox else None,
            "localization": {
                "status": self.localization_status,
                "confidence": round(self.localization_confidence, 4),
                "backend": self.localization_backend,
                "rejection_reasons": list(self.localization_rejection_reasons),
                "candidate_count": len(self.localization_candidates),
                "has_outer_polygon": self.outer_polygon is not None,
                "aspect_ratio": (
                    round(self.aspect_ratio, 6) if self.aspect_ratio is not None else None
                ),
                "aspect_confidence": round(self.aspect_confidence, 4),
            },
            "semantic_labels": list(self.semantic_masks.keys()),
            "artifact_labels": list(self.artifact_masks.keys()),
            "properties": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in self.properties.items()},
        }
