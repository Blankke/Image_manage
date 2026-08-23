"""与 GUI 共用核心流水线的 ScreenRestore CLI。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.presets import (
    PresetId,
    apply_preset,
    build_default_pipeline,
    build_registry,
)
from screenrestore.diagnostics.logging_config import configure_logging
from screenrestore.geometry import (
    AutomaticGeometryService,
    OnnxQuadDetector,
    target_class_for_scene,
)
from screenrestore.io.image_exporter import ExportOptions, export_image, infer_export_format
from screenrestore.io.image_loader import ImageLoadError, load_image
from screenrestore.io.project_file import ProjectFileError, load_project, verify_project_source
from screenrestore.operators.lens_distortion import undistort_lens

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建公开 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="screenrestore",
        description="离线电子化恢复斜拍画作、明信片、显示器、投影和 LED 屏照片",
    )
    parser.add_argument("input", nargs="?", help="输入 PNG/JPEG/WebP/BMP/TIFF 图像")
    parser.add_argument("--output", "-o", required=True, help="输出 PNG/JPEG/WebP/TIFF 路径")
    parser.add_argument("--project", help="加载 .screenrestore.json 流水线配置")
    parser.add_argument(
        "--preset",
        choices=[item.value for item in PresetId if item != PresetId.CUSTOM],
        help="覆盖项目或默认场景预设",
    )
    parser.add_argument(
        "--corners",
        nargs="+",
        metavar="CORNER",
        help="auto，或四个 x,y 坐标（支持归一化坐标）",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖输出文件")
    parser.add_argument("--quality", type=int, default=92, help="JPEG/WebP 质量，1..100")
    parser.add_argument("--remove-gps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-exif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-diagnostics", action="store_true", help="向 stdout 输出 JSON 诊断")
    parser.add_argument("--quad-model", type=Path, help="可选 QuadLocator-S ONNX 模型路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 CLI，成功返回 0，用户输入或处理错误返回 2。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    registry = build_registry()
    warnings: list[str] = []
    try:
        if args.project:
            loaded = load_project(args.project, registry)
            input_path = Path(args.input).expanduser() if args.input else loaded.source_path
            pipeline = loaded.pipeline
            warnings.extend(loaded.warnings)
        else:
            if not args.input:
                parser.error("未使用 --project 时必须提供 input")
            input_path = Path(args.input).expanduser()
            pipeline = build_default_pipeline(registry)
            loaded = None
        document = load_image(input_path)
        if loaded is not None:
            warnings = verify_project_source(loaded, document)
        if args.preset:
            apply_preset(pipeline, PresetId(args.preset))
        effective_preset = (
            PresetId(args.preset)
            if args.preset
            else loaded.preset
            if loaded is not None
            else PresetId.DISPLAY
        )
        geometry_service = AutomaticGeometryService(
            OnnxQuadDetector(args.quad_model) if args.quad_model is not None else None
        )
        geometry_localization = _configure_corners(
            pipeline,
            document,
            args.corners,
            geometry_service,
            effective_preset.value,
        )
        output_path = Path(args.output).expanduser()
        export_format = infer_export_format(output_path)
        options = ExportOptions(
            format=export_format,
            quality=args.quality,
            keep_exif=args.keep_exif,
            remove_gps=args.remove_gps,
            overwrite=args.overwrite,
        )
        started = time.perf_counter()
        context = ProcessingContext(
            preview=False,
            progress=(
                lambda fraction, message: print(
                    f"{fraction * 100:6.1f}% {message}", file=sys.stderr
                )
                if args.verbose
                else None
            ),
        )
        result = pipeline.process(document.original_rgb, context, document.content_hash)
        destination = export_image(result, output_path, options, document.path)
        elapsed = time.perf_counter() - started
        diagnostics = {
            "status": "ok",
            "input": str(document.path),
            "output": str(destination),
            "input_size": [document.width, document.height],
            "output_size": [int(result.shape[1]), int(result.shape[0])],
            "elapsed_seconds": round(elapsed, 4),
            "backend": "CPU/OpenCV",
            "preset": effective_preset.value,
            "geometry_localization": geometry_localization,
            "operator_timings": {key: round(value, 6) for key, value in pipeline.last_timings.items()},
            "warnings": warnings,
        }
        LOGGER.info("CLI 处理完成：%s", diagnostics)
        if args.json_diagnostics:
            print(json.dumps(diagnostics, ensure_ascii=False))
        elif not args.verbose:
            print(destination)
        return 0
    except (ImageLoadError, ProjectFileError, ValueError, OSError, RuntimeError) as exc:
        LOGGER.exception("CLI 处理失败")
        if args.json_diagnostics:
            print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"错误：{exc}", file=sys.stderr)
        return 2


def _configure_corners(  # type: ignore[no-untyped-def]
    pipeline,
    document,
    raw_corners,
    geometry_service: AutomaticGeometryService,
    target_scene: str,
) -> dict[str, object] | None:
    """解析自动或显式四角并更新共用几何算子。"""

    if not raw_corners:
        return None
    params = pipeline.state("geometry").params.to_dict()
    if len(raw_corners) == 1 and raw_corners[0].lower() == "auto":
        working = document.original_rgb
        lens_state = pipeline.state("lens_distortion")
        if lens_state.enabled:
            working, _metadata = undistort_lens(working, lens_state.params)
        decision = geometry_service.localize(
            working,
            target_class_for_scene(target_scene),
        )
        if not decision.accepted or decision.proposed_corners is None:
            reasons = ", ".join(reason.value for reason in decision.rejection_reasons)
            raise ValueError(f"自动定位拒绝继续：{reasons}")
        height, width = working.shape[:2]
        normalized = decision.proposed_corners / np.array(
            [width - 1, height - 1], np.float32
        )
        params["corners"] = np.clip(normalized, 0.0, 1.0).tolist()
        pipeline.update_parameters("geometry", params)
        return decision.to_dict(working.shape)
    values: list[float] = []
    for item in raw_corners:
        values.extend(float(value) for value in item.replace(";", ",").split(",") if value)
    if len(values) != 8:
        raise ValueError("--corners 需要 auto 或恰好四个 x,y 坐标")
    points = np.asarray(values, np.float32).reshape(4, 2)
    if float(points.max()) > 1.0:
        points /= np.array([document.width - 1, document.height - 1], np.float32)
    params["corners"] = np.clip(points, 0.0, 1.0).tolist()
    pipeline.update_parameters("geometry", params)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
