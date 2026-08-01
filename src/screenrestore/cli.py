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
from screenrestore.io.image_exporter import ExportOptions, export_image, infer_export_format
from screenrestore.io.image_loader import ImageLoadError, load_image
from screenrestore.io.project_file import ProjectFileError, load_project, verify_project_source
from screenrestore.operators.geometry import detect_quadrilaterals

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建公开 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="screenrestore",
        description="离线恢复斜拍显示器、投影、LED 屏和电子海报照片",
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
        geometry_confidence = _configure_corners(pipeline, document, args.corners)
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
            "geometry_confidence": geometry_confidence,
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


def _configure_corners(pipeline, document, raw_corners) -> float | None:  # type: ignore[no-untyped-def]
    """解析自动或显式四角并更新共用几何算子。"""

    if not raw_corners:
        return None
    params = pipeline.state("geometry").params.to_dict()
    if len(raw_corners) == 1 and raw_corners[0].lower() == "auto":
        proxy = document.proxy()
        candidates = detect_quadrilaterals(proxy)
        if not candidates:
            raise ValueError("自动四角检测失败；请显式指定四角或使用完整图像边界")
        height, width = proxy.shape[:2]
        normalized = candidates[0].corners / np.array([width - 1, height - 1], np.float32)
        params["corners"] = np.clip(normalized, 0.0, 1.0).tolist()
        pipeline.update_parameters("geometry", params)
        return candidates[0].confidence
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

