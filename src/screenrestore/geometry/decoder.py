"""QuadLocator 角点热图的唯一解码规范。

训练验证、ONNX 运行时、overlay 和 benchmark 都应调用本模块，避免同一热图在不同
入口得到不同四角。热图输入是 logits；坐标输出位于热图像素空间。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from typing import Any

import cv2
import numpy as np

DECODER_VERSION = "quad-peak-local-softargmax-v2"
COHERENT_DECODER_VERSION = "quad-coherent-evidence-v1"
COHERENT_REPAIR_DECODER_VERSION = "quad-coherent-repair-v1"
COHERENT_GUARDED_DECODER_VERSION = "quad-coherent-evidence-guarded-v1"


@dataclass(frozen=True, slots=True)
class CornerDecoderSpec:
    """可写入 checkpoint 与报告的稳定解码契约。"""

    version: str = DECODER_VERSION
    local_window: int = 5
    nms_radius: int = 3
    minimum_peak: float = 0.05

    def __post_init__(self) -> None:
        if self.local_window not in (5, 7) or self.local_window % 2 == 0:
            raise ValueError("局部 soft-argmax 窗口必须为 5 或 7")
        if self.nms_radius < 1:
            raise ValueError("NMS 半径必须至少为 1")
        if not 0.0 <= self.minimum_peak < 1.0:
            raise ValueError("minimum_peak 必须位于 [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CornerPeakDiagnostics:
    """单角原始多峰证据，数值均可直接写入 JSON。"""

    peak1: float
    peak2: float
    peak_difference: float
    peak_ratio: float
    peak_distance: float
    normalized_entropy: float
    local_sharpness: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class DecodedCorners:
    """四通道角点热图的结构化解码结果。"""

    coordinates: np.ndarray | None
    confidences: tuple[float, float, float, float]
    diagnostics: tuple[
        CornerPeakDiagnostics,
        CornerPeakDiagnostics,
        CornerPeakDiagnostics,
        CornerPeakDiagnostics,
    ]
    spec: CornerDecoderSpec
    coherence: dict[str, Any] | None = None


def decode_corner_logits(
    logits: np.ndarray,
    spec: CornerDecoderSpec | None = None,
) -> DecodedCorners:
    """按全局峰、NMS 第二峰和局部 sigmoid soft-argmax 解码四角。

    接受 ``4×H×W`` 或 ``1×4×H×W`` logits。第二峰只用于歧义诊断，不会把坐标
    拉向远处候选；这正是相对旧版全图质心/softmax 的关键区别。
    """

    active_spec = spec or CornerDecoderSpec()
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("角点 logits 必须为 4×H×W 或 1×4×H×W")
    if not np.all(np.isfinite(values)):
        raise ValueError("角点 logits 包含非有限值")
    probabilities = _sigmoid(values)
    coordinates: list[list[float]] = []
    confidences: list[float] = []
    diagnostics: list[CornerPeakDiagnostics] = []
    for heatmap in probabilities:
        decoded = _decode_one(heatmap, active_spec)
        diagnostics.append(decoded)
        confidences.append(decoded.peak1)
        if decoded.peak1 >= active_spec.minimum_peak:
            coordinates.append([decoded.x, decoded.y])
    coordinate_array = (
        np.asarray(coordinates, dtype=np.float32) if len(coordinates) == 4 else None
    )
    return DecodedCorners(
        coordinate_array,
        tuple(confidences),  # type: ignore[arg-type]
        tuple(diagnostics),  # type: ignore[arg-type]
        active_spec,
    )


def decode_coherent_corner_logits(
    logits: np.ndarray,
    content_mask_logits: np.ndarray,
    boundary_logits: np.ndarray,
    spec: CornerDecoderSpec | None = None,
    *,
    top_k: int = 3,
    edge_samples: int = 20,
    repair_only: bool = False,
    minimum_evidence_log_gain: float = 0.0,
) -> DecodedCorners:
    """联合热图、内容 mask 与 boundary，从同一实例选择四个角。

    四个角通道分别生成局部极大值候选，再枚举至多 ``top_k ** 4`` 个组合。候选
    分数是角点置信度、soft mask IoU 与边界支持的等权对数和，可解释为三个独立证据
    的乘积；全局最高峰组合始终在候选集中。这个过程只改变离散实例选择，亚像素坐标
    仍使用与训练一致的局部 sigmoid soft-argmax。设置正的
    ``minimum_evidence_log_gain`` 时，合法独立峰只有在替代候选达到指定对数证据增益后
    才会被改写。
    """

    active_spec = spec or CornerDecoderSpec()
    if top_k < 1 or top_k > 6:
        raise ValueError("top_k 必须位于 1..6")
    if edge_samples < 4:
        raise ValueError("edge_samples 必须至少为 4")
    if not isfinite(minimum_evidence_log_gain) or minimum_evidence_log_gain < 0.0:
        raise ValueError("minimum_evidence_log_gain 必须为有限非负数")
    decoder_version = (
        COHERENT_REPAIR_DECODER_VERSION
        if repair_only
        else (
            COHERENT_GUARDED_DECODER_VERSION
            if minimum_evidence_log_gain > 0.0
            else COHERENT_DECODER_VERSION
        )
    )
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("角点 logits 必须为 4×H×W 或 1×4×H×W")
    if not np.all(np.isfinite(values)):
        raise ValueError("角点 logits 包含非有限值")
    height, width = values.shape[1:]
    mask_values = _single_map(content_mask_logits, (height, width), "content mask")
    boundary_values = _single_map(boundary_logits, (height, width), "boundary")
    probabilities = _sigmoid(values)
    mask_probability = _sigmoid(mask_values)
    boundary_probability = _sigmoid(boundary_values)
    diagnostics = tuple(_decode_one(item, active_spec) for item in probabilities)
    candidates = [
        _local_peak_candidates(item, active_spec, top_k=top_k) for item in probabilities
    ]
    if any(not items for items in candidates):
        return DecodedCorners(
            None,
            tuple(float(item.peak1) for item in diagnostics),  # type: ignore[arg-type]
            diagnostics,  # type: ignore[arg-type]
            active_spec,
            {
                "version": decoder_version,
                "top_k": top_k,
                "evaluated_combinations": 0,
                "valid_combinations": 0,
                "reason": "corner_below_minimum_peak",
            },
        )

    independent = tuple(items[0] for items in candidates)
    independent_points = np.asarray(
        [[item[0], item[1]] for item in independent], dtype=np.float32
    )
    independent_valid = _semantic_quad_is_valid(
        independent_points,
        width=width,
        height=height,
    )
    if repair_only and independent_valid:
        return DecodedCorners(
            independent_points,
            tuple(float(item[2]) for item in independent),  # type: ignore[arg-type]
            diagnostics,  # type: ignore[arg-type]
            active_spec,
            {
                "version": COHERENT_REPAIR_DECODER_VERSION,
                "top_k": top_k,
                "edge_samples": edge_samples,
                "evaluated_combinations": 1,
                "valid_combinations": 1,
                "selected_ranks": [0, 0, 0, 0],
                "changed_from_independent_peaks": False,
                "reason": "independent_peaks_semantically_valid",
            },
        )

    best: dict[str, Any] | None = None
    evaluated = 0
    valid = 0
    for combination in product(*candidates):
        evaluated += 1
        points = np.asarray([[item[0], item[1]] for item in combination], np.float32)
        if not _semantic_quad_is_valid(points, width=width, height=height):
            continue
        valid += 1
        peaks = np.asarray([item[2] for item in combination], np.float64)
        mask_iou = _soft_mask_iou(points, mask_probability)
        boundary_support = _edge_support(points, boundary_probability, edge_samples)
        # 三类证据以相同权重进入；对数域避免连乘下溢，也不引入从私人集拟合的权重。
        score = float(
            np.mean(np.log(np.clip(peaks, 1e-8, 1.0)))
            + np.log(max(mask_iou, 1e-8))
            + np.log(max(boundary_support, 1e-8))
        )
        ranks = [int(item[3]) for item in combination]
        # 完全同分时优先较低 rank，使结果在平台间稳定并保留原最高峰。
        tie_break = -sum(ranks)
        if best is None or (score, tie_break) > (best["score"], best["tie_break"]):
            best = {
                "coordinates": points,
                "confidences": tuple(float(value) for value in peaks),
                "score": score,
                "tie_break": tie_break,
                "selected_ranks": ranks,
                "mask_iou": mask_iou,
                "boundary_support": boundary_support,
            }

    if best is None:
        return DecodedCorners(
            None,
            tuple(float(item.peak1) for item in diagnostics),  # type: ignore[arg-type]
            diagnostics,  # type: ignore[arg-type]
            active_spec,
            {
                "version": decoder_version,
                "top_k": top_k,
                "evaluated_combinations": evaluated,
                "valid_combinations": 0,
                "reason": "no_semantically_valid_combination",
            },
        )
    independent_score: float | None = None
    evidence_gain: float | None = None
    if independent_valid:
        independent_peaks = np.asarray([item[2] for item in independent], np.float64)
        independent_mask_iou = _soft_mask_iou(independent_points, mask_probability)
        independent_boundary_support = _edge_support(
            independent_points,
            boundary_probability,
            edge_samples,
        )
        independent_score = float(
            np.mean(np.log(np.clip(independent_peaks, 1e-8, 1.0)))
            + np.log(max(independent_mask_iou, 1e-8))
            + np.log(max(independent_boundary_support, 1e-8))
        )
        evidence_gain = float(best["score"] - independent_score)
        if evidence_gain < minimum_evidence_log_gain:
            best = {
                "coordinates": independent_points,
                "confidences": tuple(float(value) for value in independent_peaks),
                "score": independent_score,
                "tie_break": 0,
                "selected_ranks": [0, 0, 0, 0],
                "mask_iou": independent_mask_iou,
                "boundary_support": independent_boundary_support,
            }
    return DecodedCorners(
        best["coordinates"],
        best["confidences"],
        diagnostics,  # type: ignore[arg-type]
        active_spec,
        {
            "version": decoder_version,
            "top_k": top_k,
            "edge_samples": edge_samples,
            "evaluated_combinations": evaluated,
            "valid_combinations": valid,
            "selected_ranks": best["selected_ranks"],
            "changed_from_independent_peaks": any(rank > 0 for rank in best["selected_ranks"]),
            "evidence_log_score": best["score"],
            "independent_evidence_log_score": independent_score,
            "evidence_log_gain": evidence_gain,
            "minimum_evidence_log_gain": minimum_evidence_log_gain,
            "mask_iou": best["mask_iou"],
            "boundary_support": best["boundary_support"],
        },
    )


def _single_map(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).squeeze()
    if result.ndim != 2 or result.shape != shape:
        raise ValueError(f"{name} logits 必须与角点热图空间尺寸一致")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} logits 包含非有限值")
    return result


def _local_peak_candidates(
    heatmap: np.ndarray,
    spec: CornerDecoderSpec,
    *,
    top_k: int,
) -> list[tuple[float, float, float, int]]:
    """返回 ``(x, y, peak, rank)``；每个峰使用统一局部窗口求亚像素坐标。"""

    suppressed = heatmap.copy()
    height, width = heatmap.shape
    radius = spec.local_window // 2
    output: list[tuple[float, float, float, int]] = []
    for rank in range(top_k):
        flat_index = int(np.argmax(suppressed))
        peak_y, peak_x = divmod(flat_index, width)
        peak = float(suppressed[peak_y, peak_x])
        if peak < spec.minimum_peak:
            break
        y0, y1 = max(0, peak_y - radius), min(height, peak_y + radius + 1)
        x0, x1 = max(0, peak_x - radius), min(width, peak_x + radius + 1)
        local = heatmap[y0:y1, x0:x1]
        total = max(float(local.sum()), 1e-8)
        local_y, local_x = np.indices(local.shape, dtype=np.float32)
        x = float(((local_x + x0) * local).sum() / total)
        y = float(((local_y + y0) * local).sum() / total)
        output.append((x, y, peak, rank))
        sy0 = max(0, peak_y - spec.nms_radius)
        sy1 = min(height, peak_y + spec.nms_radius + 1)
        sx0 = max(0, peak_x - spec.nms_radius)
        sx1 = min(width, peak_x + spec.nms_radius + 1)
        suppressed[sy0:sy1, sx0:sx1] = -1.0
    return output


def _semantic_quad_is_valid(points: np.ndarray, *, width: int, height: int) -> bool:
    """验证通道顺序 TL/TR/BR/BL，拒绝交叉、退化和越过图像的大量 padding 峰。"""

    if not np.all(np.isfinite(points)) or not cv2.isContourConvex(points):
        return False
    area = abs(float(cv2.contourArea(points)))
    if area < max(4.0, 0.001 * width * height):
        return False
    try:
        from .rectify import order_corners

        ordered = order_corners(points)
    except ValueError:
        return False
    diagonal = max(1.0, float(np.hypot(width, height)))
    return bool(np.max(np.linalg.norm(ordered - points, axis=1)) <= 1e-5 * diagonal)


def _soft_mask_iou(points: np.ndarray, mask_probability: np.ndarray) -> float:
    polygon = np.zeros(mask_probability.shape, np.uint8)
    cv2.fillConvexPoly(polygon, np.rint(points).astype(np.int32), 1)
    polygon_float = polygon.astype(np.float32)
    intersection = float(np.sum(mask_probability * polygon_float))
    union = float(mask_probability.sum() + polygon_float.sum() - intersection)
    return intersection / max(union, 1e-8)


def _edge_support(points: np.ndarray, boundary: np.ndarray, samples: int) -> float:
    positions = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    edge_points = []
    for index in range(4):
        start = points[index]
        end = points[(index + 1) % 4]
        edge_points.append(start[None] + (end - start)[None] * positions[:, None])
    coordinates = np.concatenate(edge_points, axis=0)
    map_x = coordinates[:, 0].reshape(1, -1).astype(np.float32)
    map_y = coordinates[:, 1].reshape(1, -1).astype(np.float32)
    sampled = cv2.remap(
        boundary,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return float(np.mean(sampled))


def _decode_one(heatmap: np.ndarray, spec: CornerDecoderSpec) -> CornerPeakDiagnostics:
    height, width = heatmap.shape
    flat_index = int(np.argmax(heatmap))
    peak_y, peak_x = divmod(flat_index, width)
    peak1 = float(heatmap[peak_y, peak_x])

    suppressed = heatmap.copy()
    # 使用方形邻域覆盖局部 decoder 的完整窗口。旧版圆形邻域会漏掉
    # (dx=3, dy=1) 等主峰肩部，把同一宽峰误报为第二候选。
    y0_suppressed = max(0, peak_y - spec.nms_radius)
    y1_suppressed = min(height, peak_y + spec.nms_radius + 1)
    x0_suppressed = max(0, peak_x - spec.nms_radius)
    x1_suppressed = min(width, peak_x + spec.nms_radius + 1)
    suppressed[y0_suppressed:y1_suppressed, x0_suppressed:x1_suppressed] = -1.0
    second_index = int(np.argmax(suppressed))
    second_y, second_x = divmod(second_index, width)
    peak2 = max(0.0, float(suppressed[second_y, second_x]))

    radius = spec.local_window // 2
    y0, y1 = max(0, peak_y - radius), min(height, peak_y + radius + 1)
    x0, x1 = max(0, peak_x - radius), min(width, peak_x + radius + 1)
    local = heatmap[y0:y1, x0:x1]
    local_total = max(float(local.sum()), 1e-8)
    local_y, local_x = np.indices(local.shape, dtype=np.float32)
    decoded_x = float(((local_x + x0) * local).sum() / local_total)
    decoded_y = float(((local_y + y0) * local).sum() / local_total)

    global_weights = heatmap / max(float(heatmap.sum()), 1e-8)
    entropy = -float(np.sum(global_weights * np.log(np.clip(global_weights, 1e-12, 1.0))))
    normalized_entropy = entropy / max(float(np.log(max(2, heatmap.size))), 1e-8)
    local_mean_without_peak = (local_total - peak1) / max(1, local.size - 1)
    local_sharpness = float(np.clip(peak1 - local_mean_without_peak, 0.0, 1.0))
    peak_distance = float(np.hypot(second_x - peak_x, second_y - peak_y))
    return CornerPeakDiagnostics(
        peak1=peak1,
        peak2=peak2,
        peak_difference=max(0.0, peak1 - peak2),
        peak_ratio=peak1 / max(peak2, 1e-8),
        peak_distance=peak_distance,
        normalized_entropy=float(np.clip(normalized_entropy, 0.0, 1.0)),
        local_sharpness=local_sharpness,
        x=decoded_x,
        y=decoded_y,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


__all__ = [
    "COHERENT_DECODER_VERSION",
    "COHERENT_GUARDED_DECODER_VERSION",
    "COHERENT_REPAIR_DECODER_VERSION",
    "DECODER_VERSION",
    "CornerDecoderSpec",
    "CornerPeakDiagnostics",
    "DecodedCorners",
    "decode_coherent_corner_logits",
    "decode_corner_logits",
]
