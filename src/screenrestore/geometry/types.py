"""自动几何定位使用的稳定数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import cv2
import numpy as np


class TargetClass(StrEnum):
    """第一阶段明确支持的平面目标类别。"""

    ARTWORK = "artwork"
    POSTCARD = "postcard"
    SCREEN = "screen"
    NONE = "none"


def target_class_for_scene(scene: str | None) -> TargetClass | None:
    """把产品预设/场景标签归一化为 QuadLocator 类别提示。"""

    if scene is None:
        return None
    return {
        "artwork": TargetClass.ARTWORK,
        "glossy_artwork": TargetClass.ARTWORK,
        "postcard": TargetClass.POSTCARD,
        "document": TargetClass.POSTCARD,
        "display": TargetClass.SCREEN,
        "electronic_poster": TargetClass.SCREEN,
        "cinema": TargetClass.SCREEN,
        "led": TargetClass.SCREEN,
    }.get(scene)


class TargetLayer(StrEnum):
    """四边形对应的物理层级。"""

    CONTENT = "content"
    OUTER = "outer"
    UNKNOWN = "unknown"


class LocalizationStatus(StrEnum):
    """自动定位是否可以进入无人值守恢复链。"""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RejectionReason(StrEnum):
    """可用于产品统计和数据回流的拒绝原因。"""

    NO_CANDIDATE = "no_candidate"
    INVALID_QUAD = "invalid_quad"
    TARGET_ABSENT = "target_absent"
    TARGET_CLASS_UNCERTAIN = "target_class_uncertain"
    CORNER_UNCERTAIN = "corner_uncertain"
    BOUNDARY_UNCERTAIN = "boundary_uncertain"
    LAYER_AMBIGUOUS = "layer_ambiguous"
    SCORE_AMBIGUOUS = "score_ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class QuadrilateralCandidate:
    """传统或学习型检测器产生的候选四边形。"""

    corners: np.ndarray
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"
    layer: TargetLayer = TargetLayer.UNKNOWN

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float32)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError("候选四边形必须是有限的 4×2 数组")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("候选置信度必须位于 [0, 1]")
        object.__setattr__(self, "corners", corners.copy())


@dataclass(frozen=True, slots=True)
class QuadPrediction:
    """QuadLocator 一次前向推理的结构化输出。

    坐标均位于原始输入图像像素空间。掩码与 boundary map 可以保持模型输出尺寸，
    高分辨率精修器会按需缩放；运行时不会把这些数组写回或原地修改。
    """

    content_quad: np.ndarray | None
    outer_quad: np.ndarray | None = None
    corner_confidences: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    presence_confidence: float = 0.0
    outer_presence_confidence: float = 0.0
    target_class: TargetClass = TargetClass.NONE
    class_confidence: float = 0.0
    layer_confidence: float = 1.0
    content_mask: np.ndarray | None = None
    boundary_map: np.ndarray | None = None
    candidates: tuple[QuadrilateralCandidate, ...] = ()
    backend: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("content_quad", "outer_quad"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (4, 2) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} 必须是有限的 4×2 数组")
            object.__setattr__(self, name, array.copy())
        if len(self.corner_confidences) != 4:
            raise ValueError("必须为四个角分别提供置信度")
        scalar_confidences = (
            *self.corner_confidences,
            self.presence_confidence,
            self.outer_presence_confidence,
            self.class_confidence,
            self.layer_confidence,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in scalar_confidences):
            raise ValueError("所有模型置信度必须位于 [0, 1]")


@dataclass(frozen=True, slots=True)
class AspectEstimate:
    """画幅比例估计及其证据强度。"""

    ratio: float
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError("画幅比例必须是有限正数")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("画幅置信度必须位于 [0, 1]")


@dataclass(frozen=True, slots=True)
class EdgeRefinement:
    """原分辨率四边拟合结果。"""

    corners: np.ndarray
    accepted: bool
    edge_support: tuple[float, float, float, float]
    corner_shifts: tuple[float, float, float, float]
    reason: str = ""

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float32)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError("精修四边形必须是有限的 4×2 数组")
        object.__setattr__(self, "corners", corners.copy())

    @property
    def mean_support(self) -> float:
        return float(np.mean(self.edge_support))


@dataclass(frozen=True, slots=True)
class LocalizationDecision:
    """三个客户端共同消费的自动定位最终决策。"""

    status: LocalizationStatus
    proposed_corners: np.ndarray | None
    coarse_corners: np.ndarray | None
    outer_corners: np.ndarray | None
    target_class: TargetClass
    layer: TargetLayer
    confidence: float
    aspect: AspectEstimate | None
    backend: str
    rejection_reasons: tuple[RejectionReason, ...] = ()
    candidates: tuple[QuadrilateralCandidate, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("proposed_corners", "coarse_corners", "outer_corners"):
            value = getattr(self, name)
            if value is None:
                continue
            corners = np.asarray(value, dtype=np.float32)
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                raise ValueError(f"{name} 必须是有限的 4×2 数组")
            object.__setattr__(self, name, corners.copy())
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("最终置信度必须位于 [0, 1]")
        if self.status == LocalizationStatus.ACCEPTED:
            if self.proposed_corners is None or self.rejection_reasons:
                raise ValueError("接受结果必须包含四角且不能包含拒绝原因")
        elif not self.rejection_reasons:
            raise ValueError("拒绝结果必须提供至少一个原因")

    @property
    def accepted(self) -> bool:
        return self.status == LocalizationStatus.ACCEPTED

    def normalized_corners(self, image_shape: tuple[int, ...]) -> np.ndarray | None:
        """返回 `[0,1]` 四角副本，拒绝结果仍可用于诊断预览。"""

        if self.proposed_corners is None:
            return None
        height, width = image_shape[:2]
        scale = np.array([max(1, width - 1), max(1, height - 1)], np.float32)
        return np.clip(self.proposed_corners / scale, 0.0, 1.0)

    def to_dict(self, image_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
        """序列化有限诊断，不包含图像或大体积掩码。"""

        corners = self.proposed_corners
        if image_shape is not None:
            corners = self.normalized_corners(image_shape)
        return {
            "status": self.status.value,
            "accepted": self.accepted,
            "corners": corners.astype(float).tolist() if corners is not None else None,
            "outer_corners": (
                self.outer_corners.astype(float).tolist()
                if self.outer_corners is not None and image_shape is None
                else (
                    np.clip(
                        self.outer_corners
                        / np.array(
                            [max(1, image_shape[1] - 1), max(1, image_shape[0] - 1)],
                            np.float32,
                        ),
                        0.0,
                        1.0,
                    ).astype(float).tolist()
                    if self.outer_corners is not None and image_shape is not None
                    else None
                )
            ),
            "target_class": self.target_class.value,
            "target_layer": self.layer.value,
            "confidence": round(float(self.confidence), 6),
            "aspect": (
                {
                    "ratio": round(self.aspect.ratio, 6),
                    "confidence": round(self.aspect.confidence, 6),
                    "source": self.aspect.source,
                }
                if self.aspect is not None
                else None
            ),
            "backend": self.backend,
            "rejection_reasons": [reason.value for reason in self.rejection_reasons],
            "candidate_count": len(self.candidates),
            "diagnostics": self.diagnostics,
        }


def quadrilateral_is_valid(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """检查四边形是否有限、凸、非自交且处于合理图像范围。"""

    try:
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(points)):
        return False
    height, width = image_shape[:2]
    if width < 2 or height < 2:
        return False
    margin = 0.03 * max(width, height)
    if (
        np.any(points[:, 0] < -margin)
        or np.any(points[:, 0] > width - 1 + margin)
        or np.any(points[:, 1] < -margin)
        or np.any(points[:, 1] > height - 1 + margin)
    ):
        return False
    contour = points.reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(contour) and abs(float(cv2.contourArea(contour))) >= 4.0)
