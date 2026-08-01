"""Web、CLI 和测试可复用的无状态恢复服务层。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.presets import PresetId, apply_preset, build_default_pipeline
from screenrestore.io.image_exporter import ExportFormat, encode_image_bytes
from screenrestore.io.image_loader import decode_image_bytes
from screenrestore.operators.geometry import detect_quadrilaterals
from screenrestore.operators.lens_distortion import (
    LensCalibrationParameters,
    LensDistortionParameters,
    calibrate_lens,
    undistort_lens,
)
from screenrestore.operators.mesh_warp import MeshWarpParameters
from screenrestore.operators.multiframe_fusion import (
    MultiFrameFusionParameters,
    align_and_fuse,
)


@dataclass(frozen=True, slots=True)
class UploadedImage:
    """已验证但尚未解码的上传图片。"""

    filename: str
    data: bytes


@dataclass(slots=True)
class RestoreResult:
    """Web 恢复结果与有限 JSON 诊断。"""

    image_rgb: np.ndarray
    diagnostics: dict[str, object]


class WebRestoreService:
    """组合多帧预处理与共享 ``ImagePipeline``，不持有用户图片。"""

    def __init__(self, max_pixels_per_image: int = 80_000_000) -> None:
        if max_pixels_per_image <= 0:
            raise ValueError("Web 单图像素上限必须大于 0")
        self.max_pixels_per_image = max_pixels_per_image

    def decode_uploads(self, uploads: list[UploadedImage]) -> list[np.ndarray]:
        """解码 1～20 张上传图，响应结束后即可释放原始字节。"""

        if not uploads:
            raise ValueError("至少需要上传一张图片")
        if len(uploads) > 20:
            raise ValueError("一次最多处理 20 张图片")
        return [
            decode_image_bytes(
                upload.data,
                upload.filename,
                max_pixels=self.max_pixels_per_image,
            )[0]
            for upload in uploads
        ]

    def detect(
        self,
        image_rgb: np.ndarray,
        lens_settings: dict[str, Any] | None = None,
        *,
        include_preview: bool = False,
    ) -> dict[str, object]:
        """在可选镜头校正后的代理图上返回归一化四边形候选。"""

        working = image_rgb
        lens_metadata: dict[str, object] | None = None
        if lens_settings and bool(lens_settings.get("enabled", False)):
            lens_params = _lens_parameters(lens_settings)
            working, lens_metadata = undistort_lens(image_rgb, lens_params)
        proxy, scale = _proxy(working, 1400)
        candidates = detect_quadrilaterals(proxy)
        height, width = working.shape[:2]
        response_candidates = []
        for candidate in candidates:
            normalized = candidate.corners / scale / np.array(
                [max(1, width - 1), max(1, height - 1)],
                np.float32,
            )
            response_candidates.append(
                {
                    "corners": np.clip(normalized, 0.0, 1.0).tolist(),
                    "confidence": candidate.confidence,
                    "scores": candidate.scores,
                }
            )
        response: dict[str, object] = {
            "image_size": [width, height],
            "candidates": response_candidates,
            "lens": lens_metadata,
        }
        if include_preview and lens_metadata is not None:
            preview, _ = _proxy(working, 1400)
            encoded = encode_image_bytes(preview, ExportFormat.JPEG, quality=88)
            response["corrected_preview"] = {
                "data_url": "data:image/jpeg;base64,"
                + base64.b64encode(encoded).decode("ascii"),
                "size": [preview.shape[1], preview.shape[0]],
            }
        return response

    def restore(
        self,
        images_rgb: list[np.ndarray],
        settings: dict[str, Any] | None = None,
    ) -> RestoreResult:
        """运行多帧融合、镜头、透视、网格和经典恢复流水线。"""

        if not images_rgb:
            raise ValueError("至少需要一张图片")
        values = settings or {}
        _reject_unknown(
            values,
            {"preset", "corners", "ratio_mode", "custom_ratio", "lens", "mesh", "fusion"},
            "恢复设置",
        )
        preset = PresetId(str(values.get("preset", PresetId.DISPLAY.value)))
        if preset == PresetId.CUSTOM:
            raise ValueError("Web 恢复不能直接选择 custom 预设")

        fusion_diagnostics: dict[str, object]
        if len(images_rgb) > 1:
            raw_fusion = values.get("fusion", {})
            if not isinstance(raw_fusion, dict):
                raise ValueError("fusion 必须是 JSON 对象")
            fusion_params = MultiFrameFusionParameters.from_dict(raw_fusion)
            fusion_result = align_and_fuse(images_rgb, fusion_params, ProcessingContext(preview=False))
            source = fusion_result.image_rgb
            fusion_diagnostics = fusion_result.diagnostics
        else:
            source = images_rgb[0]
            gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            potentially_missing = (gray <= 2) | (gray >= 253)
            fusion_diagnostics = {
                "input_frames": 1,
                "used_frames": 1,
                "claim": "single-observation",
                "potentially_missing_fraction": float(potentially_missing.mean()),
                "unresolved_fraction": float(potentially_missing.mean()),
            }

        pipeline = build_default_pipeline()
        apply_preset(pipeline, preset)
        lens_settings = values.get("lens", {})
        if not isinstance(lens_settings, dict):
            raise ValueError("lens 必须是 JSON 对象")
        lens_enabled = bool(lens_settings.get("enabled", False))
        if lens_enabled:
            lens_params = _lens_parameters(lens_settings)
            pipeline.state("lens_distortion").enabled = True
            pipeline.state("lens_distortion").params = lens_params
        else:
            pipeline.state("lens_distortion").enabled = False

        geometry_values = pipeline.state("geometry").params.to_dict()
        ratio_mode = str(values.get("ratio_mode", geometry_values["ratio_mode"]))
        geometry_values["ratio_mode"] = ratio_mode
        if "custom_ratio" in values:
            geometry_values["custom_ratio"] = float(values["custom_ratio"])
        corners = values.get("corners", geometry_values["corners"])
        detection_diagnostics: dict[str, object] | None = None
        if corners == "auto":
            detected = self.detect(
                source,
                lens_settings if lens_enabled else None,
            )
            candidates = detected["candidates"]
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("自动四角检测失败，请在画布上手动调整四角")
            best = candidates[0]
            if not isinstance(best, dict):
                raise ValueError("自动四角检测返回无效候选")
            geometry_values["corners"] = best["corners"]
            detection_diagnostics = {
                "confidence": best["confidence"],
                "scores": best["scores"],
                "corrected_input_size": detected["image_size"],
            }
        else:
            geometry_values["corners"] = corners
        pipeline.update_parameters("geometry", geometry_values)

        mesh_settings = values.get("mesh", {})
        if not isinstance(mesh_settings, dict):
            raise ValueError("mesh 必须是 JSON 对象")
        mesh_enabled = bool(mesh_settings.get("enabled", False))
        if mesh_enabled:
            pipeline.state("mesh_warp").enabled = True
            pipeline.state("mesh_warp").params = _mesh_parameters(mesh_settings)
        else:
            pipeline.state("mesh_warp").enabled = False

        context = ProcessingContext(preview=False)
        output = pipeline.process(source, context)
        diagnostics: dict[str, object] = {
            "status": "ok",
            "preset": preset.value,
            "input_frames": len(images_rgb),
            "input_size": [source.shape[1], source.shape[0]],
            "output_size": [output.shape[1], output.shape[0]],
            "fusion": fusion_diagnostics,
            "geometry_detection": detection_diagnostics,
            "operator_timings": {
                key: round(value, 6) for key, value in pipeline.last_timings.items()
            },
            "lens": _json_metadata(context.metadata.get("lens_distortion")),
            "mesh": _json_metadata(context.metadata.get("mesh_warp")),
            "physical_limits": (
                "多帧结果仅融合实际观测；所有输入中均被遮挡、饱和或模糊的信息仍标记为未解决。"
                if len(images_rgb) > 1
                else "单张图已彻底丢失的信息无法物理还原；经典修复只能抑制伪影或推断邻域。"
            ),
        }
        return RestoreResult(np.ascontiguousarray(output), diagnostics)

    def calibrate(
        self,
        images_rgb: list[np.ndarray],
        settings: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        """运行棋盘格镜头标定并返回可直接填入恢复 API 的 lens 对象。"""

        raw = settings or {}
        calibration_params = LensCalibrationParameters.from_dict(raw)
        result = calibrate_lens(images_rgb, calibration_params)
        response = result.to_dict()
        lens = dict(result.parameters.to_dict())
        lens["enabled"] = True
        response["lens"] = lens
        return response


def _lens_parameters(raw: dict[str, Any]) -> LensDistortionParameters:
    values = {key: value for key, value in raw.items() if key != "enabled"}
    params = LensDistortionParameters.from_dict(values)
    params.validate()
    return params


def _mesh_parameters(raw: dict[str, Any]) -> MeshWarpParameters:
    values = {key: value for key, value in raw.items() if key != "enabled"}
    params = MeshWarpParameters.from_dict(values)
    params.validate()
    return params


def _proxy(image: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale >= 1.0:
        return image, 1.0
    return (
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def _reject_unknown(data: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{label}包含未知字段：{', '.join(sorted(unknown))}")


def _json_metadata(value: object) -> object:
    """仅把小型标量/列表元数据带到 Web 响应，绝不返回像素数组。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items() if not isinstance(item, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    return str(value)[:200]
