"""批量验证“屏摄测试图—对应原图”并输出可视化差异。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/validate_paired_samples.py
    python scripts/validate_paired_samples.py --only 电影测试二

说明：原图只用于定位测试照片里的内容边界和计算客观指标，不会作为恢复输入，也
不会把原图像素混入输出。默认读取“测试数据”，把 PNG 和 JSON 写入
“validation_outputs/paired_reference”；原始测试数据保持只读且不会上传网络。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from screenrestore.core.operator import ProcessingContext
from screenrestore.core.pipeline import ImagePipeline, OperatorRegistry
from screenrestore.core.presets import (
    PresetId,
    ProcessingMode,
    apply_preset,
    apply_processing_mode,
    build_default_pipeline,
    build_registry,
)
from screenrestore.inference.backend import InferenceError
from screenrestore.inference.factory import create_inference_backend
from screenrestore.inference.model_manifest import ModelRole, load_manifest
from screenrestore.io.image_exporter import ExportFormat, ExportOptions, export_image
from screenrestore.io.image_loader import load_image
from screenrestore.operators.geometry import AspectRatioMode
from screenrestore.validation import (
    align_for_comparison,
    compare_images,
    difference_heatmap,
    extract_reference_region,
    register_reference,
)


@dataclass(frozen=True, slots=True)
class PairedCase:
    """一组实拍图、原图和适用的经典预设。"""

    name: str
    photo: Path
    reference: Path
    preset: PresetId


def main(argv: list[str] | None = None) -> int:
    """执行配对定位、恢复、二次对齐和指标统计。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "测试数据",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "validation_outputs"
        / "paired_reference",
    )
    parser.add_argument("--only", help="只运行名称包含该文本的样本")
    parser.add_argument(
        "--ai-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "examples"
        / "realesrgan-general-x4v3-torch.json",
        help="用于 AI Enhanced 对照的 enhancement 模型清单",
    )
    parser.add_argument("--skip-ai", action="store_true", help="跳过 AI Enhanced 对照")
    args = parser.parse_args(argv)
    data_directory = args.data_directory.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    cases = _discover_cases(data_directory)
    if args.only:
        cases = [case for case in cases if args.only in case.name]
    if not cases:
        raise SystemExit("没有找到符合条件的配对样本")
    output_directory.mkdir(parents=True, exist_ok=True)
    ai_manifest, ai_status = _available_ai_manifest(
        None if args.skip_ai else args.ai_manifest.expanduser().resolve()
    )

    registry = build_registry()
    reports: list[dict[str, object]] = []
    steps_per_case = 7 if ai_manifest is not None else 6
    total_steps = len(cases) * steps_per_case
    completed = 0
    for case in cases:
        _progress(completed, total_steps, f"加载 {case.name}")
        photo_document = load_image(case.photo)
        reference_document = load_image(case.reference)
        photo = photo_document.original_rgb
        reference = reference_document.original_rgb
        completed += 1

        _progress(completed, total_steps, f"参考定位 {case.name}")
        registration = register_reference(photo, reference)
        geometry_result = extract_reference_region(photo, registration, reference.shape)
        completed += 1

        _progress(completed, total_steps, f"忠实恢复 {case.name}")
        pipeline = _paired_pipeline(
            registry,
            case.preset,
            registration.corners_photo,
            photo.shape,
            reference.shape,
        )
        context = ProcessingContext(preview=False)
        restored = pipeline.process(
            photo,
            context,
            source_id=f"paired:{photo_document.content_hash}",
        )
        movie_ablation: dict[str, np.ndarray] = {}
        if case.name == "电影测试二":
            movie_ablation = {
                "01_geometry": geometry_result,
                "02_banding": _process_until(
                    registry,
                    case,
                    registration.corners_photo,
                    photo,
                    reference.shape,
                    "banding",
                ),
                "03_classic_without_tone": _process_until(
                    registry,
                    case,
                    registration.corners_photo,
                    photo,
                    reference.shape,
                    "denoise",
                ),
                "04_fidelity": restored,
            }
        completed += 1

        ai_restored: np.ndarray | None = None
        ai_context: ProcessingContext | None = None
        ai_error: str | None = None
        if ai_manifest is not None:
            _progress(completed, total_steps, f"AI Enhanced {case.name}")
            ai_pipeline = _paired_pipeline(
                registry,
                case.preset,
                registration.corners_photo,
                photo.shape,
                reference.shape,
            )
            _enable_ai_enhancement(ai_pipeline, ai_manifest)
            ai_context = ProcessingContext(preview=False)
            try:
                ai_restored = ai_pipeline.process(
                    photo,
                    ai_context,
                    source_id=f"paired-ai:{photo_document.content_hash}",
                )
            except InferenceError as exc:
                # 单个模型失败不抹掉已完成的 Fidelity 基准；错误进入结构化报告。
                ai_error = str(exc)
            if ai_restored is not None and case.name == "电影测试二":
                movie_ablation["05_ai_enhanced"] = ai_restored
            completed += 1

        _progress(completed, total_steps, f"对齐评分 {case.name}")
        aligned_geometry, geometry_alignment = align_for_comparison(
            geometry_result,
            reference,
        )
        aligned_restored, restored_alignment = align_for_comparison(restored, reference)
        geometry_direct_metrics = compare_images(geometry_result, reference)
        restored_direct_metrics = compare_images(restored, reference)
        geometry_aligned_metrics = compare_images(aligned_geometry, reference)
        restored_aligned_metrics = compare_images(aligned_restored, reference)
        aligned_ai: np.ndarray | None = None
        ai_alignment: dict[str, object] | None = None
        ai_aligned_metrics: dict[str, float] | None = None
        ai_direct_metrics: dict[str, float] | None = None
        if ai_restored is not None:
            aligned_ai, ai_alignment = align_for_comparison(ai_restored, reference)
            ai_direct_metrics = compare_images(ai_restored, reference)
            ai_aligned_metrics = compare_images(aligned_ai, reference)
        completed += 1

        _progress(completed, total_steps, f"输出图像 {case.name}")
        prefix = output_directory / case.name
        _save(geometry_result, prefix.with_name(prefix.name + "_参考定位几何.png"))
        _save(restored, prefix.with_name(prefix.name + "_忠实恢复.png"))
        _save(aligned_restored, prefix.with_name(prefix.name + "_对齐原图.png"))
        _save(
            difference_heatmap(aligned_restored, reference),
            prefix.with_name(prefix.name + "_差异热图.png"),
        )
        _save(
            _draw_reference_corners(photo, registration.corners_photo),
            prefix.with_name(prefix.name + "_内容定位.png"),
        )
        if ai_restored is not None and aligned_ai is not None:
            _save(ai_restored, prefix.with_name(prefix.name + "_AI增强.png"))
            _save(aligned_ai, prefix.with_name(prefix.name + "_AI对齐原图.png"))
            _save(
                difference_heatmap(aligned_ai, reference),
                prefix.with_name(prefix.name + "_AI差异热图.png"),
            )
        movie_ablation_metrics: dict[str, dict[str, float]] | None = None
        if movie_ablation:
            movie_ablation_metrics = {}
            for stage, image in movie_ablation.items():
                _save(image, output_directory / f"电影测试二_{stage}.png")
                aligned_stage, _alignment = align_for_comparison(image, reference)
                movie_ablation_metrics[stage] = compare_images(aligned_stage, reference)
        completed += 1

        _progress(completed, total_steps, f"汇总 {case.name}")
        reports.append(
            {
                "case": case.name,
                "preset": case.preset.value,
                "photo": str(case.photo),
                "reference": str(case.reference),
                "photo_size": [photo.shape[1], photo.shape[0]],
                "reference_size": [reference.shape[1], reference.shape[0]],
                "geometry_output_size": [geometry_result.shape[1], geometry_result.shape[0]],
                "restored_output_size": [restored.shape[1], restored.shape[0]],
                "reference_localization": registration.to_dict(),
                "geometry_direct_metrics": geometry_direct_metrics,
                "restored_direct_metrics": restored_direct_metrics,
                "geometry_aligned_metrics": geometry_aligned_metrics,
                "restored_aligned_metrics": restored_aligned_metrics,
                "geometry_comparison_alignment": geometry_alignment,
                "restored_comparison_alignment": restored_alignment,
                "ai_direct_metrics": ai_direct_metrics,
                "ai_aligned_metrics": ai_aligned_metrics,
                "ai_comparison_alignment": ai_alignment,
                "ai_error": ai_error,
                "movie_ablation_metrics": movie_ablation_metrics,
                "metric_change_aligned": _metric_change(
                    geometry_aligned_metrics,
                    restored_aligned_metrics,
                ),
                "operator_timings": {
                    key: round(value, 6) for key, value in pipeline.last_timings.items()
                },
                "pipeline_metadata": _json_metadata(context.metadata),
                "ai_pipeline_metadata": _json_metadata(
                    ai_context.metadata if ai_context is not None else None
                ),
                "reference_usage": (
                    "原图仅用于定位与评分；恢复流水线的唯一图像输入是实拍图。"
                ),
            }
        )
        completed += 1

    report = {
        "status": "ok",
        "method": "reference-localized-observation-only-restoration",
        "case_count": len(reports),
        "ai": ai_status,
        "summary": _summarize_reports(reports),
        "cases": reports,
    }
    report_path = output_directory / "diagnostics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path = output_directory / "index.html"
    html_path.write_text(
        _html_report(report, output_directory, data_directory),
        encoding="utf-8",
    )
    _progress(total_steps, total_steps, f"完成：{html_path}")
    print(
        json.dumps(
            {"status": "ok", "report": str(report_path), "html": str(html_path)},
            ensure_ascii=False,
        )
    )
    return 0


def _discover_cases(data_directory: Path) -> list[PairedCase]:
    """按约定文件名发现 1..N 与电影测试二配对。"""

    if not data_directory.is_dir():
        raise ValueError(f"测试数据目录不存在：{data_directory}")
    reference_directory = data_directory / "原图"
    cases: list[PairedCase] = []
    for photo in sorted(data_directory.glob("*测试.jpg")):
        number = photo.name.removesuffix("测试.jpg")
        if not number.isdigit():
            continue
        reference = reference_directory / f"{number}.jpg"
        if not reference.is_file():
            raise ValueError(f"{photo.name} 缺少对应原图：{reference.name}")
        cases.append(PairedCase(f"测试{number}", photo, reference, PresetId.DISPLAY))
    movie_photo = data_directory / "电影测试二.jpg"
    movie_reference = reference_directory / "电影测试二原图.jpg"
    if movie_photo.is_file() and movie_reference.is_file():
        cases.append(
            PairedCase(
                "电影测试二",
                movie_photo,
                movie_reference,
                PresetId.CINEMA,
            )
        )
    return cases


def _paired_pipeline(
    registry: OperatorRegistry,
    preset: PresetId,
    corners_photo: np.ndarray,
    photo_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
) -> ImagePipeline:
    """创建与普通产品路径一致、但四角由测试参考定位的流水线。"""

    pipeline = build_default_pipeline(registry)
    apply_preset(pipeline, preset)
    photo_height, photo_width = photo_shape[:2]
    reference_height, reference_width = reference_shape[:2]
    normalized = corners_photo / np.array(
        [max(1, photo_width - 1), max(1, photo_height - 1)],
        dtype=np.float32,
    )
    geometry = pipeline.state("geometry").params.to_dict()
    geometry.update(
        {
            "corners": np.clip(normalized, 0.0, 1.0).tolist(),
            "ratio_mode": AspectRatioMode.CUSTOM.value,
            "custom_ratio": reference_width / reference_height,
            "auto_crop": False,
        }
    )
    pipeline.update_parameters("geometry", geometry)
    return pipeline


def _process_until(
    registry: OperatorRegistry,
    case: PairedCase,
    corners_photo: np.ndarray,
    photo_rgb: np.ndarray,
    reference_shape: tuple[int, ...],
    stop_operator_id: str,
) -> np.ndarray:
    """运行电影消融到指定节点，后续算子全部关闭。"""

    pipeline = _paired_pipeline(
        registry,
        case.preset,
        corners_photo,
        photo_rgb.shape,
        reference_shape,
    )
    reached_stop = False
    for state in pipeline.states:
        if reached_stop:
            state.enabled = False
        elif state.operator.id == stop_operator_id:
            reached_stop = True
    if not reached_stop:
        raise ValueError(f"消融终点不存在：{stop_operator_id}")
    pipeline.cache.clear()
    return pipeline.process(
        photo_rgb,
        ProcessingContext(preview=False),
        source_id=f"paired-ablation:{case.name}:{stop_operator_id}",
    )


def _available_ai_manifest(path: Path | None) -> tuple[Path | None, dict[str, object]]:
    """确认对照模型角色和本机依赖；不联网下载任何内容。"""

    if path is None:
        return None, {"enabled": False, "status": "skipped"}
    try:
        manifest = load_manifest(path)
    except InferenceError as exc:
        return None, {"enabled": False, "status": "invalid", "reason": str(exc)[:256]}
    if manifest.role != ModelRole.ENHANCEMENT:
        return None, {
            "enabled": False,
            "status": "wrong-role",
            "manifest_id": manifest.id,
        }
    available, _reason = create_inference_backend(manifest).is_available()
    if not available:
        return None, {
            "enabled": False,
            "status": "local-runtime-or-files-missing",
            "manifest_id": manifest.id,
        }
    return path, {
        "enabled": True,
        "status": "available",
        "manifest_id": manifest.id,
        "name": manifest.name,
        "role": manifest.role.value,
        "task": manifest.task,
        "generated_detail_warning": (
            "AI Enhanced 使用模型先验生成统计纹理，不等同于原始数字帧。"
        ),
    }


def _enable_ai_enhancement(pipeline: ImagePipeline, manifest_path: Path) -> None:
    """在 Fidelity 基础上只启用一个低强度感知增强节点。"""

    state = pipeline.state("enhancement_model")
    values = state.params.to_dict()
    values.update(
        {
            "manifest_path": str(manifest_path),
            "model_strength": 0.2,
            "denoise_strength": 0.25,
            "output_scale": 1.0,
            "blend_strength": 1.0,
        }
    )
    state.params = state.operator.parameter_type.from_dict(values)
    state.operator.validate(state.params)
    state.enabled = True
    apply_processing_mode(pipeline, ProcessingMode.AI_ENHANCED)


def _summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    """汇总三路指标均值和相对 Geometry 胜出样本数。"""

    keys = (
        "psnr_db",
        "luminance_ssim",
        "gradient_correlation",
        "delta_e_mean",
        "spectral_peak_excess_db",
    )
    variants = {
        "geometry": "geometry_aligned_metrics",
        "fidelity": "restored_aligned_metrics",
        "ai_enhanced": "ai_aligned_metrics",
    }
    means: dict[str, object] = {}
    for name, field in variants.items():
        rows = [item[field] for item in reports if isinstance(item.get(field), dict)]
        means[name] = {
            "case_count": len(rows),
            **{
                key: round(
                    float(np.mean([row[key] for row in rows if key in row])),  # type: ignore[index]
                    6,
                )
                for key in keys
                if any(key in row for row in rows)  # type: ignore[operator]
            },
        }
    wins: dict[str, dict[str, int]] = {}
    for name, field in variants.items():
        if name == "geometry":
            continue
        win_counts = {key: 0 for key in keys}
        for item in reports:
            baseline = item.get("geometry_aligned_metrics")
            candidate = item.get(field)
            if not isinstance(baseline, dict) or not isinstance(candidate, dict):
                continue
            for key in keys:
                if key not in baseline or key not in candidate:
                    continue
                if key in {"delta_e_mean", "spectral_peak_excess_db"}:
                    improved = abs(float(candidate[key])) < abs(float(baseline[key]))
                else:
                    improved = float(candidate[key]) > float(baseline[key])
                win_counts[key] += int(improved)
        wins[name] = win_counts
    return {"aligned_metric_means": means, "wins_over_geometry": wins}


def _html_report(
    report: dict[str, object],
    output_directory: Path,
    data_directory: Path,
) -> str:
    """生成无需服务端即可打开的 8 组可视化比较报告。"""

    cards: list[str] = []
    for item in report["cases"]:  # type: ignore[index]
        case = str(item["case"])
        photo_path = Path(str(item["photo"]))
        reference_path = Path(str(item["reference"]))
        photo_link = os.path.relpath(photo_path, output_directory).replace(os.sep, "/")
        reference_link = os.path.relpath(reference_path, output_directory).replace(os.sep, "/")
        images = [
            ("实拍输入", photo_link),
            ("对应原图（仅评分）", reference_link),
            ("Geometry", f"{case}_参考定位几何.png"),
            ("Fidelity", f"{case}_忠实恢复.png"),
        ]
        if item.get("ai_aligned_metrics") is not None:
            images.append(("AI Enhanced", f"{case}_AI增强.png"))
        figures = "".join(
            (
                "<figure><img loading='lazy' src='"
                + html.escape(path, quote=True)
                + "' alt='"
                + html.escape(f"{case} {label}", quote=True)
                + "'><figcaption>"
                + html.escape(label)
                + "</figcaption></figure>"
            )
            for label, path in images
        )
        geometry = item["geometry_aligned_metrics"]
        fidelity = item["restored_aligned_metrics"]
        ai_metrics = item.get("ai_aligned_metrics")
        rows = [
            _metric_row("Geometry", geometry),
            _metric_row("Fidelity", fidelity),
        ]
        if isinstance(ai_metrics, dict):
            rows.append(_metric_row("AI Enhanced", ai_metrics))
        movie_ablation = _movie_ablation_html(item)
        cards.append(
            "<article><h2>"
            + html.escape(case)
            + "</h2><div class='images'>"
            + figures
            + "</div><table><thead><tr><th>阶段</th><th>PSNR ↑</th>"
            + "<th>SSIM ↑</th><th>梯度相关 ↑</th><th>ΔE ↓</th><th>频谱峰偏差</th>"
            + "<th>亮度偏差</th><th>黑裁切</th><th>白裁切</th>"
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + movie_ablation
            + "</article>"
        )
    summary = html.escape(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    data_label = html.escape(str(data_directory))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ScreenRestore 8 组配对基准</title><style>
body{{margin:0;background:#0b0e11;color:#edf0f2;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:32px}}
h1{{font-size:clamp(28px,5vw,62px);margin:0}}p{{color:#aab2b8}}article{{margin:28px 0;padding:22px;background:#151a1f;border:1px solid #2a333a;border-radius:14px}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}figure{{margin:0;background:#080a0c}}img{{width:100%;height:260px;object-fit:contain;display:block}}figcaption{{padding:8px;color:#dff24c}}
table{{width:100%;border-collapse:collapse;margin-top:16px;overflow:auto}}th,td{{padding:8px;border-bottom:1px solid #30383e;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#080a0c;padding:16px;border-radius:10px}}
</style></head><body><main><h1>8 组实拍 / 原图基准</h1><p>数据目录：{data_label}。原图只用于定位和评分，未进入恢复像素路径。</p>
<pre>{summary}</pre>{''.join(cards)}</main></body></html>"""


def _metric_row(label: str, metrics: object) -> str:
    """渲染一行有限指标，避免把完整 JSON 塞入 HTML。"""

    if not isinstance(metrics, dict):
        return ""
    values = (
        metrics["psnr_db"],
        metrics["luminance_ssim"],
        metrics["gradient_correlation"],
        metrics["delta_e_mean"],
        metrics["spectral_peak_excess_db"],
        metrics["luminance_bias"],
        metrics["black_clipping_ratio"],
        metrics["white_clipping_ratio"],
    )
    return "<tr><td>" + html.escape(label) + "</td>" + "".join(
        f"<td>{float(value):.4f}</td>" for value in values
    ) + "</tr>"


def _movie_ablation_html(item: dict[str, object]) -> str:
    """把电影样本的逐节点输出和分项指标嵌入同一份离线报告。"""

    metrics = item.get("movie_ablation_metrics")
    if not isinstance(metrics, dict):
        return ""
    labels = (
        ("01_geometry", "① 仅参考定位几何"),
        ("02_banding", "② 条带 / 宽光幕校正"),
        ("03_classic_without_tone", "③ 降噪后、色调前"),
        ("04_fidelity", "④ Fidelity"),
        ("05_ai_enhanced", "⑤ AI Enhanced"),
    )
    figures = "".join(
        "<figure><img loading='lazy' src='电影测试二_"
        + html.escape(stage, quote=True)
        + ".png' alt='"
        + html.escape(label, quote=True)
        + "'><figcaption>"
        + html.escape(label)
        + "</figcaption></figure>"
        for stage, label in labels
        if stage in metrics
    )
    rows = "".join(
        _metric_row(label, metrics.get(stage))
        for stage, label in labels
        if stage in metrics
    )
    return (
        "<section><h3>电影测试二逐阶段消融</h3>"
        "<p>同一定位与输出尺寸下逐节点比较，用于区分几何、经典恢复和 AI 的贡献。</p>"
        "<div class='images'>"
        + figures
        + "</div><table><thead><tr><th>阶段</th><th>PSNR ↑</th><th>SSIM ↑</th>"
        "<th>梯度相关 ↑</th><th>ΔE ↓</th><th>频谱峰偏差</th><th>亮度偏差</th>"
        "<th>黑裁切</th><th>白裁切</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></section>"
    )


def _save(image_rgb: np.ndarray, path: Path) -> None:
    """以可覆盖 PNG 输出验证产物。"""

    export_image(
        image_rgb,
        path,
        ExportOptions(
            ExportFormat.PNG,
            overwrite=True,
            keep_exif=False,
        ),
    )


def _draw_reference_corners(photo_rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """绘制参考定位轮廓，便于人工确认没有选到外层显示器。"""

    output = photo_rgb.copy()
    thickness = max(2, round(max(photo_rgb.shape[:2]) / 450))
    cv2.polylines(
        output,
        [np.rint(corners).astype(np.int32)],
        True,
        (32, 230, 70),
        thickness,
        cv2.LINE_AA,
    )
    for index, point in enumerate(corners):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(output, center, thickness * 3, (255, 80, 30), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(index + 1),
            (center[0] + thickness * 3, center[1] - thickness * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, thickness / 3),
            (255, 80, 30),
            thickness,
            cv2.LINE_AA,
        )
    return output


def _metric_change(
    baseline: dict[str, float],
    restored: dict[str, float],
) -> dict[str, float]:
    """报告恢复相对几何基线的指标差，正负方向由指标本身解释。"""

    return {
        key: round(restored[key] - baseline[key], 6)
        for key in baseline.keys() & restored.keys()
    }


def _json_metadata(value: object) -> object:
    """把流水线元数据中的 ndarray 转成尺寸描述，避免输出像素内容。"""

    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:256]


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
