"""用根目录两个真实样例验证 ScreenRestore。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/validate_samples.py --quad-model models/weights/quadlocator-s.onnx

说明：脚本读取根目录的“电影屏幕测试.jpg”和“纸质海报测试.jpg”，自动检测四角，
分别应用 cinema/document 预设，在 validation_outputs/ 保存几何基线、完整恢复图和
有限长度的 JSON 诊断。不会覆盖原图，也不会上传任何内容。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.pipeline import ImagePipeline
from screenrestore.core.presets import (
    PresetId,
    apply_preset,
    build_default_pipeline,
    build_registry,
)
from screenrestore.geometry import (
    AutomaticGeometryService,
    OnnxQuadDetector,
    target_class_for_scene,
)
from screenrestore.io.image_exporter import ExportFormat, ExportOptions, export_image
from screenrestore.io.image_loader import load_image


@dataclass(frozen=True, slots=True)
class SampleCase:
    """一个真实样例及其预设。"""

    name: str
    filename: str
    preset: PresetId


CASES = (
    SampleCase("电影屏幕", "电影屏幕测试.jpg", PresetId.CINEMA),
    SampleCase("纸质海报", "纸质海报测试.jpg", PresetId.DOCUMENT),
)


def main(argv: list[str] | None = None) -> int:
    """处理两个样例并输出诊断 JSON。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quad-model", type=Path)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output_directory = (args.output_dir or root / "validation_outputs").expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    geometry_service = AutomaticGeometryService(
        OnnxQuadDetector(args.quad_model) if args.quad_model is not None else None
    )
    diagnostics: list[dict[str, object]] = []
    total_steps = len(CASES) * 4
    completed = 0

    for case in CASES:
        source_path = root / case.filename
        _progress(completed, total_steps, f"加载 {case.name}")
        document = load_image(source_path)
        completed += 1

        decision = geometry_service.localize(
            document.original_rgb,
            target_class_for_scene(case.preset.value),
        )
        if not decision.accepted or decision.proposed_corners is None:
            reasons = ", ".join(reason.value for reason in decision.rejection_reasons)
            raise RuntimeError(f"{case.name} 自动定位被拒绝：{reasons}")
        normalized = decision.proposed_corners / np.array(
            [document.width - 1, document.height - 1], np.float32
        )
        pipeline = build_default_pipeline(registry)
        apply_preset(pipeline, case.preset)
        geometry_params = pipeline.state("geometry").params.to_dict()
        geometry_params["corners"] = np.clip(normalized, 0, 1).tolist()
        pipeline.update_parameters("geometry", geometry_params)
        completed += 1
        _progress(completed, total_steps, f"几何基线 {case.name}")

        geometry_pipeline = ImagePipeline.from_dict(pipeline.to_dict(), registry)
        for state in geometry_pipeline.states:
            if state.operator.id not in {"orientation", "geometry"}:
                state.enabled = False
        geometry_result = geometry_pipeline.process(
            document.original_rgb,
            ProcessingContext(preview=False),
            document.content_hash + ":geometry",
        )
        export_image(
            geometry_result,
            output_directory / f"{case.name}_仅几何.png",
            ExportOptions(ExportFormat.PNG, overwrite=True, keep_exif=False),
        )
        completed += 1
        _progress(completed, total_steps, f"完整恢复 {case.name}")

        restored = pipeline.process(
            document.original_rgb,
            ProcessingContext(preview=False),
            document.content_hash + ":full",
        )
        restored_path = export_image(
            restored,
            output_directory / f"{case.name}_自动恢复.png",
            ExportOptions(ExportFormat.PNG, overwrite=True, keep_exif=False),
        )
        completed += 1
        _progress(completed, total_steps, f"诊断 {case.name}")
        diagnostics.append(
            {
                "case": case.name,
                "preset": case.preset.value,
                "input_size": [document.width, document.height],
                "output_size": [int(restored.shape[1]), int(restored.shape[0])],
                "localization_confidence": round(decision.confidence, 6),
                "localization_backend": decision.backend,
                "corners": np.rint(decision.proposed_corners).astype(int).tolist(),
                "geometry_metrics": _quality_metrics(geometry_result),
                "restored_metrics": _quality_metrics(restored),
                "operator_timings": {
                    key: round(value, 6) for key, value in pipeline.last_timings.items()
                },
                "output": str(restored_path),
            }
        )

    report_path = output_directory / "diagnostics.json"
    report_path.write_text(
        json.dumps({"status": "ok", "cases": diagnostics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _progress(total_steps, total_steps, f"完成：{report_path}")
    print(json.dumps({"status": "ok", "report": str(report_path)}, ensure_ascii=False))
    return 0


def _quality_metrics(image_rgb: np.ndarray) -> dict[str, float]:
    """计算不依赖参考真值的有限诊断；指标不等价于主观质量。"""

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return {
        "black_clipping_ratio": round(float(np.mean(gray <= 2 / 255)), 6),
        "white_clipping_ratio": round(float(np.mean(gray >= 253 / 255)), 6),
        "mean_luminance": round(float(gray.mean()), 6),
        "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 8),
        "row_periodic_energy": round(_profile_periodic_energy(gray.mean(axis=1)), 8),
        "column_periodic_energy": round(_profile_periodic_energy(gray.mean(axis=0)), 8),
        "row_black_level_nonuniformity": round(
            _black_level_nonuniformity(gray),
            8,
        ),
    }


def _profile_periodic_energy(profile: np.ndarray) -> float:
    """报告去除慢趋势后的一维均方能量，用于观察条带而非充当真值评分。"""

    sigma = max(8.0, len(profile) / 18.0)
    trend = gaussian_filter1d(profile, sigma, mode="reflect")
    periodic = profile - trend
    return float(np.mean(np.square(periodic)))


def _black_level_nonuniformity(gray: np.ndarray) -> float:
    """衡量逐行低分位黑位相对慢趋势的起伏，辅助观察宽光幕。"""

    profile = np.quantile(gray, 0.03, axis=1).astype(np.float32)
    trend = gaussian_filter1d(profile, max(12.0, len(profile) / 6.0), mode="reflect")
    residual = gaussian_filter1d(profile - trend, 6.0, mode="reflect")
    return float(np.sqrt(np.mean(np.square(residual))))


def _progress(current: int, total: int, message: str) -> None:
    """向 stderr 输出固定宽度进度条。"""

    fraction = min(1.0, current / max(1, total))
    filled = round(fraction * 24)
    bar = "#" * filled + "-" * (24 - filled)
    print(f"\r[{bar}] {fraction * 100:6.1f}% {message}", end="", file=sys.stderr)
    if current >= total:
        print(file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
