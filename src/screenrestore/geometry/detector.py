"""QuadLocator 检测协议、经典 fallback 与 ONNX 运行时。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .classic import detect_classic_candidates
from .rectify import order_corners
from .types import QuadPrediction, QuadrilateralCandidate, TargetClass, TargetLayer


@runtime_checkable
class QuadDetector(Protocol):
    """输入原图并输出语义四边形的最小协议。"""

    def predict(
        self,
        image_rgb: np.ndarray,
        target_hint: TargetClass | None = None,
    ) -> QuadPrediction:
        """运行检测；实现不得原地修改输入图像。"""


class ClassicQuadDetector:
    """传统候选生成 fallback；没有 content/outer 证据，不能被正式自动路径接受。"""

    def __init__(self, *, max_candidates: int = 8, detection_max_edge: int = 1200) -> None:
        self.max_candidates = max_candidates
        self.detection_max_edge = detection_max_edge

    def predict(
        self,
        image_rgb: np.ndarray,
        target_hint: TargetClass | None = None,
    ) -> QuadPrediction:
        candidates = detect_classic_candidates(
            image_rgb,
            max_candidates=self.max_candidates,
            detection_max_edge=self.detection_max_edge,
        )
        if not candidates:
            return QuadPrediction(
                content_quad=None,
                target_class=TargetClass.NONE,
                backend="classic",
            )
        best = candidates[0]
        target_class = target_hint if target_hint not in (None, TargetClass.NONE) else TargetClass.NONE
        second_score = candidates[1].confidence if len(candidates) > 1 else 0.0
        margin = float(np.clip(best.confidence - second_score, 0.0, 1.0))
        scores = dict(best.scores)
        scores["candidate_margin"] = margin
        enriched_best = QuadrilateralCandidate(
            best.corners,
            best.confidence,
            scores,
            best.source,
            best.layer,
        )
        return QuadPrediction(
            content_quad=best.corners,
            corner_confidences=(best.confidence,) * 4,
            presence_confidence=best.confidence,
            target_class=target_class,
            class_confidence=1.0 if target_class != TargetClass.NONE else 0.0,
            # 传统轮廓没有 content/outer 语义，正式自动路径必须因此拒绝。
            layer_confidence=0.0,
            candidates=(enriched_best, *candidates[1:]),
            backend="classic",
        )


class OnnxQuadDetector:
    """运行 QuadLocator-S ONNX 模型并解码热图、mask 与分类头。

    权威模型契约：输入 ``image`` 为 ``B×3×H×W`` RGB float32 [0,1]；输出依次为
    ``content_corner_heatmaps``、``outer_corner_heatmaps``、``content_mask_logits``、
    ``boundary_logits``、``presence_logits``、``class_logits``。输出名可通过构造参数覆盖。
    """

    DEFAULT_OUTPUTS = (
        "content_corner_heatmaps",
        "outer_corner_heatmaps",
        "content_mask_logits",
        "boundary_logits",
        "presence_logits",
        "class_logits",
    )
    CLASS_ORDER = (
        TargetClass.ARTWORK,
        TargetClass.POSTCARD,
        TargetClass.SCREEN,
        TargetClass.NONE,
    )

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int | None = None,
        providers: list[str] | None = None,
        output_names: tuple[str, str, str, str, str, str] = DEFAULT_OUTPUTS,
    ) -> None:
        if input_size is not None and (input_size < 128 or input_size % 32):
            raise ValueError("QuadLocator 输入尺寸必须不小于 128 且为 32 的倍数")
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"QuadLocator 模型不存在：{path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("运行 ONNX QuadLocator 需要安装 screenrestore[inference-onnx]") from exc
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        self.model_path = path
        self.output_names = output_names
        self._session = ort.InferenceSession(
            str(path),
            providers=providers or ["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError("QuadLocator ONNX 必须恰好有一个图像输入")
        self._input_name = inputs[0].name
        fixed_size = _fixed_square_input_size(inputs[0].shape)
        if fixed_size is not None and input_size is not None and fixed_size != input_size:
            raise ValueError(
                f"QuadLocator 模型固定输入为 {fixed_size}，与请求的 {input_size} 不一致"
            )
        self.input_size = fixed_size or input_size or 640
        if self.input_size < 128 or self.input_size % 32:
            raise RuntimeError("QuadLocator ONNX 输入边长必须不小于 128 且为 32 的倍数")
        available_outputs = {output.name for output in self._session.get_outputs()}
        missing = set(output_names) - available_outputs
        if missing:
            raise RuntimeError(f"QuadLocator ONNX 缺少输出：{sorted(missing)}")

    def predict(
        self,
        image_rgb: np.ndarray,
        target_hint: TargetClass | None = None,
    ) -> QuadPrediction:
        tensor, transform = _letterbox_tensor(image_rgb, self.input_size)
        raw = self._session.run(list(self.output_names), {self._input_name: tensor})
        content_heatmaps, outer_heatmaps, mask_logits, boundary_logits, presence, classes = raw
        content_quad, corner_confidences = _decode_corner_heatmaps(
            content_heatmaps,
            transform,
            image_rgb.shape,
        )
        outer_quad, _outer_confidences = _decode_corner_heatmaps(
            outer_heatmaps,
            transform,
            image_rgb.shape,
        )
        class_probabilities = _softmax(np.asarray(classes, dtype=np.float32).reshape(-1))
        class_index = int(np.argmax(class_probabilities))
        target_class = self.CLASS_ORDER[class_index]
        class_confidence = float(class_probabilities[class_index])
        # hint 仅用于诊断下游分布，不覆盖模型类别结论。
        _ = target_hint
        content_mask = _sigmoid(np.asarray(mask_logits, np.float32).squeeze())
        boundary_map = _sigmoid(np.asarray(boundary_logits, np.float32).squeeze())
        candidates: list[QuadrilateralCandidate] = []
        if content_quad is not None:
            candidates.append(
                QuadrilateralCandidate(
                    content_quad,
                    float(np.mean(corner_confidences)),
                    {"heatmap_mean": float(np.mean(corner_confidences))},
                    "quadlocator_onnx",
                    TargetLayer.CONTENT,
                )
            )
        if outer_quad is not None:
            candidates.append(
                QuadrilateralCandidate(
                    outer_quad,
                    float(np.mean(_outer_confidences)),
                    {"heatmap_mean": float(np.mean(_outer_confidences))},
                    "quadlocator_onnx",
                    TargetLayer.OUTER,
                )
            )
        return QuadPrediction(
            content_quad=content_quad,
            outer_quad=outer_quad,
            corner_confidences=corner_confidences,
            presence_confidence=float(_sigmoid(np.asarray(presence, np.float32)).reshape(-1)[0]),
            target_class=target_class,
            class_confidence=class_confidence,
            layer_confidence=_layer_confidence(content_quad, outer_quad, content_mask),
            content_mask=content_mask,
            boundary_map=boundary_map,
            candidates=tuple(candidates),
            backend="quadlocator_onnx",
        )


def _fixed_square_input_size(shape: list[object]) -> int | None:
    """读取 ONNX 的固定方形输入；动态空间维度返回 ``None``。"""

    if len(shape) != 4:
        raise RuntimeError("QuadLocator ONNX 输入必须是 B×3×H×W")
    channels, height, width = shape[1:]
    if isinstance(channels, int) and channels != 3:
        raise RuntimeError("QuadLocator ONNX 输入通道数必须为 3")
    if isinstance(height, int) and isinstance(width, int):
        if height != width:
            raise RuntimeError("QuadLocator ONNX 固定空间输入必须为方形")
        return height
    return None


def _letterbox_tensor(
    image_rgb: np.ndarray,
    size: int,
) -> tuple[np.ndarray, tuple[float, float, int, int, int]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype not in (
        np.uint8,
        np.float32,
    ):
        raise ValueError("QuadLocator 需要 H×W×3 RGB uint8/float32 图像")
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=interpolation)
    if resized.dtype == np.uint8:
        resized = resized.astype(np.float32) / 255.0
    else:
        resized = np.clip(resized.astype(np.float32), 0.0, 1.0)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    scale_x = (resized_width - 1) / max(1, width - 1)
    scale_y = (resized_height - 1) / max(1, height - 1)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return np.transpose(canvas, (2, 0, 1))[None], (
        scale_x,
        scale_y,
        offset_x,
        offset_y,
        size,
    )


def _decode_corner_heatmaps(
    heatmaps: np.ndarray,
    transform: tuple[float, float, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, tuple[float, float, float, float]]:
    values = _sigmoid(np.asarray(heatmaps, dtype=np.float32))
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] != 4:
        raise RuntimeError("角点热图输出必须为 1×4×H×W")
    scale_x, scale_y, offset_x, offset_y, input_size = transform
    output_height, output_width = values.shape[2:]
    points: list[list[float]] = []
    confidences: list[float] = []
    for heatmap in values[0]:
        confidence = float(np.max(heatmap))
        confidences.append(confidence)
        if confidence < 0.05:
            continue
        threshold = max(0.05, confidence * 0.55)
        weights = np.where(heatmap >= threshold, heatmap, 0.0)
        total = float(weights.sum())
        if total <= 1e-8:
            continue
        yy, xx = np.indices(heatmap.shape, dtype=np.float32)
        model_x = float((xx * weights).sum() / total) * (input_size - 1) / max(
            1, output_width - 1
        )
        model_y = float((yy * weights).sum() / total) * (input_size - 1) / max(
            1, output_height - 1
        )
        x = (model_x - offset_x) / max(scale_x, 1e-8)
        y = (model_y - offset_y) / max(scale_y, 1e-8)
        points.append(
            [
                float(np.clip(x, 0, image_shape[1] - 1)),
                float(np.clip(y, 0, image_shape[0] - 1)),
            ]
        )
    confidence_tuple = tuple(confidences)  # type: ignore[assignment]
    if len(points) != 4:
        return None, confidence_tuple
    try:
        return order_corners(np.asarray(points, np.float32)), confidence_tuple
    except ValueError:
        return None, confidence_tuple


def _layer_confidence(
    content_quad: np.ndarray | None,
    outer_quad: np.ndarray | None,
    content_mask: np.ndarray,
) -> float:
    """组合内容分割与显式 content/outer 预测的一致性证据。

    面积大小不能用来推断画芯/卡纸/画框语义。这里只使用模型内容 mask，以及模型已
    明确输出两个层级后应满足的几何包含关系；outer 与 content 矛盾时必须降低置信度。
    """

    if content_quad is None:
        return 0.0
    mask_evidence = float(np.clip(np.mean(content_mask > 0.5) * 4.0, 0.0, 1.0))
    if outer_quad is None:
        return float(0.65 + 0.25 * mask_evidence)
    containment = _content_containment_score(content_quad, outer_quad)
    return float(np.clip((0.50 + 0.30 * mask_evidence) * containment, 0.0, 1.0))


def _content_containment_score(content_quad: np.ndarray, outer_quad: np.ndarray) -> float:
    """衡量 content 四角是否落在显式 outer 内，容许热图量化造成的微小越界。"""

    outer = order_corners(outer_quad).reshape(-1, 1, 2)
    content = order_corners(content_quad)
    diagonal = max(1.0, float(np.linalg.norm(outer_quad.max(axis=0) - outer_quad.min(axis=0))))
    tolerance = diagonal * 0.012
    distances = np.asarray(
        [cv2.pointPolygonTest(outer, tuple(float(value) for value in point), True) for point in content],
        dtype=np.float32,
    )
    # 四角全部在边界内为 1；在一条热图像素带外时线性衰减，超过容差立即归零。
    return float(np.mean(np.clip((distances + tolerance) / tolerance, 0.0, 1.0)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponent = np.exp(np.clip(shifted, -30.0, 30.0))
    return exponent / max(float(exponent.sum()), 1e-8)
