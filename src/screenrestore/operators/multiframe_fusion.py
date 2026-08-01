"""多帧对齐与基于观测置信度的像素融合。

该模块不生成不存在的纹理。它只在已对齐的实拍帧之间选择或融合可靠观测，因此可
利用不同帧补回瞬态反光、遮挡、过曝和局部模糊；所有帧都丢失的区域会被明确标为
``unresolved``，不会伪装成真实恢复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.parameters import ParameterModel


class AlignmentModel(StrEnum):
    """帧间相机运动模型。"""

    AUTO = "auto"
    TRANSLATION = "translation"
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    NONE = "none"


@dataclass
class MultiFrameFusionParameters(ParameterModel):
    """多帧融合参数。"""

    alignment: AlignmentModel = AlignmentModel.AUTO
    reference_index: int = -1
    max_frames: int = 8
    max_alignment_dimension: int = 1400
    minimum_overlap: float = 0.55
    minimum_alignment_score: float = 0.12
    outlier_threshold: float = 0.1
    exposure_compensation: bool = True

    def validate(self) -> None:
        if not -1 <= self.reference_index < self.max_frames:
            raise ValueError("参考帧索引超出可用范围")
        if not 2 <= self.max_frames <= 20:
            raise ValueError("多帧融合数量必须位于 2..20")
        if not 320 <= self.max_alignment_dimension <= 4096:
            raise ValueError("对齐代理最长边必须位于 320..4096")
        if not 0.1 <= self.minimum_overlap <= 1.0:
            raise ValueError("最小重叠率必须位于 0.1..1")
        if not 0.0 <= self.minimum_alignment_score <= 1.0:
            raise ValueError("最小对齐分数必须位于 [0, 1]")
        if not 0.02 <= self.outlier_threshold <= 0.5:
            raise ValueError("时域离群阈值必须位于 0.02..0.5")


@dataclass(slots=True)
class MultiFrameFusionResult:
    """融合图、质量图和观测来源诊断。"""

    image_rgb: np.ndarray
    confidence_map: np.ndarray
    recovered_observation_mask: np.ndarray
    unresolved_mask: np.ndarray
    reference_index: int
    aligned_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    transforms: tuple[list[list[float]], ...]
    diagnostics: dict[str, object] = field(default_factory=dict)


def align_and_fuse(
    images_rgb: list[np.ndarray],
    params: MultiFrameFusionParameters | None = None,
    context: ProcessingContext | None = None,
) -> MultiFrameFusionResult:
    """对齐 2～20 张同场景图，并只融合可靠的真实观测。"""

    settings = params or MultiFrameFusionParameters()
    settings.validate()
    if len(images_rgb) < 2:
        raise ValueError("多帧融合至少需要两张图片")
    selected = images_rgb[: settings.max_frames]
    for image in selected:
        _validate_rgb(image)
    active_context = context or ProcessingContext(preview=False)
    active_context.report(0.01, "选择多帧参考图")

    reference_index = _select_reference(selected, settings.reference_index)
    reference = np.ascontiguousarray(selected[reference_index])
    reference_height, reference_width = reference.shape[:2]
    aligned_images: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    aligned_indices: list[int] = []
    rejected_indices: list[int] = []
    transforms: list[list[list[float]]] = []
    alignment_scores: list[float] = []

    for sequence_index, image in enumerate(selected):
        active_context.cancellation.check()
        if image.shape[:2] != reference.shape[:2]:
            aspect_error = abs(
                image.shape[1] / image.shape[0] - reference_width / reference_height
            )
            if aspect_error > 0.02:
                rejected_indices.append(sequence_index)
                continue
            current = cv2.resize(
                image,
                (reference_width, reference_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            current = image

        if sequence_index == reference_index:
            output_to_input = np.eye(3, dtype=np.float64)
            score = 1.0
            aligned = reference.copy()
            valid = np.ones(reference.shape[:2], dtype=bool)
        else:
            try:
                output_to_input, score = _estimate_output_to_input_map(
                    reference,
                    current,
                    settings,
                )
                aligned, valid = _warp_to_reference(
                    current,
                    output_to_input,
                    (reference_width, reference_height),
                )
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                rejected_indices.append(sequence_index)
                continue
            overlap = float(valid.mean())
            content_score = _content_consistency(reference, aligned, valid)
            score *= 0.55 + 0.45 * content_score
            if (
                overlap < settings.minimum_overlap
                or score < settings.minimum_alignment_score
                or content_score < 0.28
            ):
                rejected_indices.append(sequence_index)
                continue

        aligned_images.append(aligned)
        valid_masks.append(valid)
        aligned_indices.append(sequence_index)
        transforms.append(output_to_input.tolist())
        alignment_scores.append(score)
        active_context.report(
            0.08 + 0.42 * (sequence_index + 1) / len(selected),
            f"已对齐 {len(aligned_images)} 张图片",
        )

    if len(aligned_images) < 2:
        raise ValueError("有效对齐帧不足两张；请改用更相近的连拍照片或手动关闭对齐")
    # 保证融合数组的第一项是参考帧，便于诊断“从其他帧真实补回”的区域。
    reference_position = aligned_indices.index(reference_index)
    if reference_position != 0:
        for collection in (aligned_images, valid_masks, aligned_indices, transforms, alignment_scores):
            collection.insert(0, collection.pop(reference_position))

    gains = _exposure_gains(aligned_images, valid_masks) if settings.exposure_compensation else [1.0] * len(aligned_images)
    active_context.report(0.52, "融合可靠观测")
    fused, confidence, recovered, unresolved = _fuse_in_strips(
        aligned_images,
        valid_masks,
        alignment_scores,
        gains,
        settings.outlier_threshold,
        active_context,
    )
    active_context.report(1.0, "多帧观测融合完成")
    diagnostics: dict[str, object] = {
        "input_frames": len(images_rgb),
        "used_frames": len(aligned_images),
        "rejected_frames": len(rejected_indices) + max(0, len(images_rgb) - len(selected)),
        "alignment_scores": [round(float(value), 5) for value in alignment_scores],
        "exposure_gains": [round(float(value), 5) for value in gains],
        "mean_confidence": float(confidence.mean()),
        "recovered_from_other_observation_fraction": float(recovered.mean()),
        "unresolved_fraction": float(unresolved.mean()),
        "claim": "observed-multiframe-fusion",
    }
    return MultiFrameFusionResult(
        image_rgb=fused,
        confidence_map=confidence,
        recovered_observation_mask=recovered,
        unresolved_mask=unresolved,
        reference_index=reference_index,
        aligned_indices=tuple(aligned_indices),
        rejected_indices=tuple(rejected_indices),
        transforms=tuple(transforms),
        diagnostics=diagnostics,
    )


def _select_reference(images: list[np.ndarray], requested: int) -> int:
    if requested >= 0:
        if requested >= len(images):
            raise ValueError("参考帧索引超过实际图片数量")
        return requested
    scores: list[float] = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, 640.0 / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        usable = float(np.mean((gray > 10) & (gray < 248)))
        scores.append(np.log1p(sharpness) * (0.4 + 0.6 * usable))
    return int(np.argmax(scores))


def _estimate_output_to_input_map(
    reference: np.ndarray,
    current: np.ndarray,
    params: MultiFrameFusionParameters,
) -> tuple[np.ndarray, float]:
    """估计“参考输出坐标→当前输入坐标”的逆采样矩阵。"""

    height, width = reference.shape[:2]
    scale = min(1.0, params.max_alignment_dimension / max(height, width))
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    if scale < 1.0:
        reference_gray = cv2.resize(reference_gray, size, interpolation=cv2.INTER_AREA)
        current_gray = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
    reference_float = reference_gray.astype(np.float32) / 255.0
    current_float = current_gray.astype(np.float32) / 255.0
    shift, phase_score = cv2.phaseCorrelate(reference_float, current_float)
    phase_map = np.array(
        [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if params.alignment == AlignmentModel.NONE:
        return np.eye(3, dtype=np.float64), 1.0
    if params.alignment == AlignmentModel.TRANSLATION:
        return _proxy_map_to_full(phase_map, scale), float(np.clip(phase_score, 0.0, 1.0))

    requested_motion = (
        cv2.MOTION_HOMOGRAPHY
        if params.alignment == AlignmentModel.HOMOGRAPHY
        else cv2.MOTION_AFFINE
    )
    initial = phase_map if requested_motion == cv2.MOTION_HOMOGRAPHY else phase_map[:2].copy()
    try:
        score, estimated = cv2.findTransformECC(
            reference_float,
            current_float,
            initial.astype(np.float32),
            requested_motion,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6),
            inputMask=None,
            gaussFiltSize=5,
        )
        matrix = (
            np.asarray(estimated, dtype=np.float64)
            if estimated.shape == (3, 3)
            else np.vstack((estimated, [0.0, 0.0, 1.0])).astype(np.float64)
        )
        if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix[:2, :2])) < 0.08:
            raise ValueError("帧间变换退化")
        return _proxy_map_to_full(matrix, scale), float(np.clip(score, 0.0, 1.0))
    except cv2.error:
        if params.alignment == AlignmentModel.HOMOGRAPHY:
            feature_map, feature_score = _feature_homography(reference_gray, current_gray)
            if feature_map is not None:
                return _proxy_map_to_full(feature_map, scale), feature_score
        return _proxy_map_to_full(phase_map, scale), float(np.clip(phase_score, 0.0, 1.0))


def _feature_homography(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """在 ECC 失败时用 ORB/RANSAC 估计参考→当前单应矩阵。"""

    detector = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
    reference_points, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    current_points, current_descriptors = detector.detectAndCompute(current_gray, None)
    if reference_descriptors is None or current_descriptors is None:
        return None, 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        reference_descriptors,
        current_descriptors,
        k=2,
    )
    good = [first for first, second in matches if first.distance < second.distance * 0.72]
    if len(good) < 12:
        return None, 0.0
    reference_locations = np.float32([reference_points[item.queryIdx].pt for item in good])
    current_locations = np.float32([current_points[item.trainIdx].pt for item in good])
    matrix, inliers = cv2.findHomography(
        reference_locations,
        current_locations,
        cv2.RANSAC,
        3.0,
    )
    if matrix is None or inliers is None:
        return None, 0.0
    inlier_fraction = float(inliers.mean())
    support = min(1.0, len(good) / 80.0)
    return matrix.astype(np.float64), inlier_fraction * (0.5 + 0.5 * support)


def _proxy_map_to_full(matrix: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return matrix.astype(np.float64)
    down = np.diag([scale, scale, 1.0])
    up = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    full = up @ matrix @ down
    return full / full[2, 2]


def _warp_to_reference(
    image: np.ndarray,
    output_to_input: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    mask_flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    if np.allclose(output_to_input[2], [0.0, 0.0, 1.0], atol=1e-10):
        transform = output_to_input[:2].astype(np.float32)
        aligned = cv2.warpAffine(
            image,
            transform,
            size,
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpAffine(
            np.ones(image.shape[:2], np.uint8),
            transform,
            size,
            flags=mask_flags,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        aligned = cv2.warpPerspective(
            image,
            output_to_input,
            size,
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpPerspective(
            np.ones(image.shape[:2], np.uint8),
            output_to_input,
            size,
            flags=mask_flags,
            borderMode=cv2.BORDER_CONSTANT,
        )
    return aligned, valid > 0


def _exposure_gains(images: list[np.ndarray], masks: list[np.ndarray]) -> list[float]:
    """在重叠中间调上估计逐像素对数增益，避免内容分布或模糊改变直方图中位数。"""

    reference = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    reference_gradient = np.maximum(
        np.abs(cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)),
        np.abs(cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)),
    )
    gains = [1.0]
    for image, mask in zip(images[1:], masks[1:], strict=True):
        current = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        current_gradient = np.maximum(
            np.abs(cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)),
            np.abs(cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)),
        )
        usable = (
            masks[0]
            & mask
            & (reference > 0.08)
            & (reference < 0.92)
            & (current > 0.08)
            & (current < 0.92)
            & (reference_gradient < 0.08)
            & (current_gradient < 0.08)
        )
        if np.count_nonzero(usable) < 256:
            gains.append(1.0)
            continue
        log_gain = np.log(reference[usable] + 1e-4) - np.log(current[usable] + 1e-4)
        estimated = float(np.clip(np.exp(np.median(log_gain)), 0.75, 1.35))
        gains.append(1.0 if abs(np.log(estimated)) < 0.03 else estimated)
    return gains


def _content_consistency(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid: np.ndarray,
) -> float:
    """以低分辨率亮度相关性拒绝已经切换内容的屏幕帧，防止时域重影。"""

    height, width = reference.shape[:2]
    scale = min(1.0, 320.0 / max(height, width))
    size = (max(16, round(width * scale)), max(16, round(height * scale)))
    reference_gray = cv2.resize(
        cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY),
        size,
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    aligned_gray = cv2.resize(
        cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY),
        size,
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    valid_small = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    if np.count_nonzero(valid_small) < 256:
        return 0.0
    reference_values = reference_gray[valid_small]
    aligned_values = aligned_gray[valid_small]
    reference_values -= float(reference_values.mean())
    aligned_values -= float(aligned_values.mean())
    denominator = float(np.linalg.norm(reference_values) * np.linalg.norm(aligned_values))
    if denominator < 1e-6:
        mean_difference = abs(float(reference_gray[valid_small].mean() - aligned_gray[valid_small].mean()))
        return 1.0 if mean_difference < 8.0 else 0.0
    correlation = float(np.dot(reference_values, aligned_values) / denominator)
    return float(np.clip(correlation, 0.0, 1.0))


def _fuse_in_strips(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    alignment_scores: list[float],
    gains: list[float],
    outlier_threshold: float,
    context: ProcessingContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分条融合，避免全分辨率多帧 float32 堆栈造成峰值内存爆炸。"""

    height, width = images[0].shape[:2]
    output = np.empty_like(images[0])
    confidence_map = np.empty((height, width), np.float32)
    recovered_mask = np.zeros((height, width), dtype=bool)
    unresolved_mask = np.zeros((height, width), dtype=bool)
    strip_height = max(32, min(192, round(24_000_000 / max(1, width * len(images) * 16))))

    for start in range(0, height, strip_height):
        context.cancellation.check()
        end = min(height, start + strip_height)
        values = np.stack(
            [
                np.clip(image[start:end].astype(np.float32) / 255.0 * gain, 0.0, 1.0)
                for image, gain in zip(images, gains, strict=True)
            ],
            axis=0,
        )
        valid = np.stack([mask[start:end] for mask in masks], axis=0)
        luma = values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722
        masked_luma = np.where(valid, luma, np.nan)
        with np.errstate(invalid="ignore"):
            temporal_median = np.nanmedian(masked_luma, axis=0)
        temporal_median = np.where(np.isfinite(temporal_median), temporal_median, luma[0])
        residual = np.abs(luma - temporal_median[None, ...])
        consensus = np.exp(-np.square(residual / outlier_threshold))

        shadow_quality = np.clip(luma / 0.075, 0.0, 1.0)
        highlight_quality = np.clip((1.0 - luma) / 0.055, 0.0, 1.0)
        chroma = values.max(axis=3) - values.min(axis=3)
        glare = np.clip((luma - 0.72) / 0.25, 0.0, 1.0) * np.clip(
            (0.16 - chroma) / 0.16,
            0.0,
            1.0,
        )
        observation_quality = shadow_quality * highlight_quality * (1.0 - 0.72 * glare)
        # 局部高频较完整的帧获得温和加权，用连拍中的清晰观测替换局部运动模糊；
        # 权重下限保留 0.65，避免把平坦区域误判为“没有信息”。
        detail = np.stack(
            [
                cv2.GaussianBlur(
                    np.abs(cv2.Laplacian(frame_luma, cv2.CV_32F, ksize=3)),
                    (0, 0),
                    1.2,
                )
                for frame_luma in luma
            ],
            axis=0,
        )
        relative_detail = detail / np.maximum(detail.max(axis=0, keepdims=True), 0.015)
        observation_quality *= 0.65 + 0.35 * np.clip(relative_detail, 0.0, 1.0)
        observation_quality *= np.asarray(alignment_scores, np.float32)[:, None, None]
        weights = valid * observation_quality * consensus
        weight_sum = weights.sum(axis=0)
        fused = (values * weights[..., None]).sum(axis=0) / np.maximum(
            weight_sum[..., None],
            1e-6,
        )

        with np.errstate(invalid="ignore"):
            median_rgb = np.nanmedian(np.where(valid[..., None], values, np.nan), axis=0)
        no_weight = weight_sum < 1e-4
        median_rgb = np.nan_to_num(median_rgb, nan=0.0)
        fused[no_weight] = median_rgb[no_weight]
        no_observation = valid.sum(axis=0) == 0
        fused[no_observation] = values[0][no_observation]
        output[start:end] = np.clip(np.rint(fused * 255.0), 0, 255).astype(np.uint8)

        available = np.maximum(valid.sum(axis=0), 1)
        confidence = np.clip(weight_sum / available, 0.0, 1.0)
        confidence_map[start:end] = confidence
        reference_bad = (~valid[0]) | (observation_quality[0] < 0.2) | (consensus[0] < 0.2)
        other_good = np.any(weights[1:] > 0.2, axis=0)
        recovered_mask[start:end] = reference_bad & other_good
        unresolved_mask[start:end] = no_observation | (weight_sum < 0.08)
        context.report(
            0.52 + 0.46 * end / height,
            f"融合可靠观测 {end}/{height} 行",
        )
    return output, confidence_map, recovered_mask, unresolved_mask


def _validate_rgb(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("多帧模块需要 H×W×3 RGB uint8 图像")
