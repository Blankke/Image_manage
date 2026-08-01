"""支持 overlap、padding 和加权融合的通用分块推理。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from screenrestore.core.operator import ProcessingContext

from .backend import InferenceError

TileFunction = Callable[[np.ndarray], np.ndarray]


def tiled_inference(
    image_rgb: np.ndarray,
    infer_tile: TileFunction,
    context: ProcessingContext,
    tile_size: int = 512,
    overlap: int = 32,
    padding: int = 16,
) -> np.ndarray:
    """分割 RGB 图、推理边缘 tile，并用线性权重消除接缝。"""

    if image_rgb.dtype != np.float32 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise InferenceError("分块推理需要 H×W×3 RGB float32 图像")
    if tile_size < 16 or not 0 <= overlap < tile_size or not 0 <= padding < tile_size // 2:
        raise InferenceError("tile_size/overlap/padding 参数无效")
    height, width = image_rgb.shape[:2]
    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    total = len(y_starts) * len(x_starts)
    output_sum: np.ndarray | None = None
    weight_sum: np.ndarray | None = None
    output_scale: int | None = None
    completed = 0

    for y in y_starts:
        for x in x_starts:
            context.cancellation.check()
            tile_height = min(tile_size, height - y)
            tile_width = min(tile_size, width - x)
            extended, crop = _extract_with_padding(
                image_rgb, x, y, tile_width, tile_height, padding
            )
            inferred = infer_tile(extended)
            if inferred.dtype != np.float32 or inferred.ndim != 3 or inferred.shape[2] != 3:
                raise InferenceError("tile 回调必须返回 RGB float32 图像")
            scale_y = inferred.shape[0] / extended.shape[0]
            scale_x = inferred.shape[1] / extended.shape[1]
            scale = round(scale_x)
            if (
                scale < 1
                or abs(scale_x - scale) > 1e-5
                or abs(scale_y - scale) > 1e-5
                or (output_scale is not None and output_scale != scale)
            ):
                raise InferenceError("所有 tile 必须使用一致的整数放大倍率")
            output_scale = scale
            top, left, bottom, right = (value * scale for value in crop)
            core = inferred[top : inferred.shape[0] - bottom, left : inferred.shape[1] - right]
            expected_shape = (tile_height * scale, tile_width * scale)
            core = core[: expected_shape[0], : expected_shape[1]]
            if core.shape[:2] != expected_shape:
                raise InferenceError("tile 输出尺寸与输入倍率不一致")
            if output_sum is None:
                output_sum = np.zeros((height * scale, width * scale, 3), np.float64)
                weight_sum = np.zeros((height * scale, width * scale, 1), np.float64)
            weight = _blend_weight(
                core.shape[0],
                core.shape[1],
                overlap * scale,
                y > 0,
                y + tile_height < height,
                x > 0,
                x + tile_width < width,
            )[..., None]
            output_y = y * scale
            output_x = x * scale
            output_sum[
                output_y : output_y + core.shape[0], output_x : output_x + core.shape[1]
            ] += core.astype(np.float64) * weight
            assert weight_sum is not None
            weight_sum[
                output_y : output_y + core.shape[0], output_x : output_x + core.shape[1]
            ] += weight
            completed += 1
            context.report(completed / total, f"分块推理 {completed}/{total}")
    assert output_sum is not None and weight_sum is not None
    output = output_sum / np.maximum(weight_sum, 1e-8)
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0).astype(np.float32))


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _extract_with_padding(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    source_height, source_width = image.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(source_width, x + width + padding), min(source_height, y + height + padding)
    extracted = image[y0:y1, x0:x1]
    missing_left = max(0, padding - x)
    missing_top = max(0, padding - y)
    missing_right = max(0, x + width + padding - source_width)
    missing_bottom = max(0, y + height + padding - source_height)
    mode = "reflect" if min(extracted.shape[:2]) > 1 else "edge"
    extended = np.pad(
        extracted,
        ((missing_top, missing_bottom), (missing_left, missing_right), (0, 0)),
        mode=mode,
    )
    return extended, (padding, padding, padding, padding)


def _blend_weight(
    height: int,
    width: int,
    overlap: int,
    fade_top: bool,
    fade_bottom: bool,
    fade_left: bool,
    fade_right: bool,
) -> np.ndarray:
    y_weight = np.ones(height, np.float64)
    x_weight = np.ones(width, np.float64)
    y_overlap = min(overlap, height // 2)
    x_overlap = min(overlap, width // 2)
    if y_overlap:
        ramp = np.linspace(1e-3, 1.0, y_overlap, dtype=np.float64)
        if fade_top:
            y_weight[:y_overlap] = ramp
        if fade_bottom:
            y_weight[-y_overlap:] = ramp[::-1]
    if x_overlap:
        ramp = np.linspace(1e-3, 1.0, x_overlap, dtype=np.float64)
        if fade_left:
            x_weight[:x_overlap] = ramp
        if fade_right:
            x_weight[-x_overlap:] = ramp[::-1]
    return np.outer(y_weight, x_weight)
