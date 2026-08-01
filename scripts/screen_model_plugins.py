"""用合成屏幕退化筛选本地模型插件，不读取或上传用户照片。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/screen_model_plugins.py \
      --manifest models/manifests/nafnet-gopro-width32-onnx.json \
      --output validation_outputs/model_screening.json \
      --task all

说明：脚本分别评估去摩尔纹、运动模糊、失焦和“干净输入不应被改写”，输出 PSNR
改善、梯度误差和色彩误差。它只用于筛选候选权重，不代表真实拍摄域最终质量。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from screenrestore.core.operator import ProcessingContext
from screenrestore.inference.factory import create_inference_backend
from screenrestore.inference.model_manifest import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("all", "demoire", "deblur"), default="all")
    parser.add_argument("--preview-directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """依次运行候选清单，并原子写入排序报告。"""

    args = build_parser().parse_args(argv)
    manifests = [load_manifest(path) for path in args.manifest]
    clean = _screen_chart()
    cases = _benchmark_cases(clean, args.task)
    total_steps = len(manifests) * (len(cases) + 1)
    completed = 0
    reports = []
    if args.preview_directory:
        args.preview_directory.mkdir(parents=True, exist_ok=True)
        _save_rgb(args.preview_directory / "00_clean.png", clean)
        for name, degraded in cases.items():
            _save_rgb(args.preview_directory / f"01_{name}_input.png", degraded)

    for manifest in manifests:
        backend = create_inference_backend(manifest)
        available, reason = backend.is_available()
        if not available:
            reports.append({"id": manifest.id, "status": "unavailable", "reason": reason})
            completed += len(cases) + 1
            _progress(completed / total_steps, f"跳过 {manifest.id}")
            continue
        started = time.perf_counter()
        clean_output = backend.run(clean, ProcessingContext(preview=False))
        clean_output = cv2.resize(clean_output, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
        clean_change = _mae(clean_output, clean)
        if args.preview_directory:
            _save_rgb(args.preview_directory / f"02_{manifest.id}_clean.png", clean_output)
        completed += 1
        _progress(completed / total_steps, f"{manifest.id} 干净输入保护")
        case_reports = []
        for name, degraded in cases.items():
            restored = backend.run(degraded, ProcessingContext(preview=False))
            restored = cv2.resize(restored, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
            metrics = _metrics(clean, degraded, restored)
            case_reports.append({"case": name, **metrics})
            if args.preview_directory:
                _save_rgb(args.preview_directory / f"02_{manifest.id}_{name}.png", restored)
            completed += 1
            _progress(completed / total_steps, f"{manifest.id} / {name}")
        mean_gain = float(np.mean([item["psnr_gain_db"] for item in case_reports]))
        mean_gradient_gain = float(
            np.mean([item["gradient_error_reduction"] for item in case_reports])
        )
        # 干净输入改写超过 3/255 后快速扣分，避免高 PSNR 但生成纹理的候选胜出。
        score = mean_gain + mean_gradient_gain * 3.0 - max(0.0, clean_change - 3.0) * 0.35
        if clean_change > 8.0:
            decision = "reject_clean_content_drift"
        elif mean_gain <= 0.15:
            decision = "reject_no_measurable_gain"
        else:
            decision = "candidate_for_real_photo_review"
        reports.append(
            {
                "id": manifest.id,
                "name": manifest.name,
                "status": "ok",
                "score": score,
                "decision": decision,
                "clean_input_mae": clean_change,
                "elapsed_seconds": time.perf_counter() - started,
                "cases": case_reports,
            }
        )
    reports.sort(key=lambda item: float(item.get("score", -1e9)), reverse=True)
    payload = {
        "schema_version": 1,
        "method": "synthetic-screen-restoration-screening",
        "warning": "合成排名只用于淘汰明显不合适权重，最终选择仍需独立实拍验证。",
        "task": args.task,
        "candidates": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    _progress(1.0, f"完成：{args.output}")
    return 0


def _screen_chart(width: int = 320, height: int = 192) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    image = np.stack(
        (
            0.12 + 0.72 * xx / max(1, width - 1),
            0.1 + 0.7 * yy / max(1, height - 1),
            0.18 + 0.5 * (xx + yy) / max(1, width + height - 2),
        ),
        axis=2,
    )
    chart = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    for x in range(12, width, 17):
        cv2.line(chart, (x, 0), (x, height - 1), (210, 210, 210), 1)
    cv2.putText(chart, "SCREEN 0123", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.rectangle(chart, (18, 105), (140, 167), (28, 34, 42), -1)
    cv2.circle(chart, (244, 132), 35, (230, 70, 45), 3)
    return chart


def _benchmark_cases(clean: np.ndarray, task: str) -> dict[str, np.ndarray]:
    cases: dict[str, np.ndarray] = {}
    if task in {"all", "demoire"}:
        height, width = clean.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        phases = (0.0, 2.1, 4.2)
        wave = np.stack(
            [np.sin(2 * np.pi * (xx * 0.165 + yy * 0.117) + phase) for phase in phases],
            axis=2,
        )
        cases["color_moire"] = np.clip(clean.astype(np.float32) + wave * 34.0, 0, 255).astype(np.uint8)
    if task in {"all", "deblur"}:
        kernel = np.zeros((13, 13), np.float32)
        cv2.line(kernel, (1, 8), (11, 4), 1.0, 1)
        kernel /= max(float(kernel.sum()), 1e-6)
        cases["motion_blur"] = cv2.filter2D(clean, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        cases["defocus"] = cv2.GaussianBlur(clean, (0, 0), 2.2)
    return cases


def _metrics(clean: np.ndarray, degraded: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    before_mse = float(np.mean(np.square(degraded.astype(np.float32) - clean.astype(np.float32))))
    after_mse = float(np.mean(np.square(restored.astype(np.float32) - clean.astype(np.float32))))
    before_psnr = 10.0 * np.log10(255.0**2 / max(before_mse, 1e-8))
    after_psnr = 10.0 * np.log10(255.0**2 / max(after_mse, 1e-8))
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    degraded_gray = cv2.cvtColor(degraded, cv2.COLOR_RGB2GRAY).astype(np.float32)
    restored_gray = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY).astype(np.float32)
    clean_gradient = cv2.Laplacian(clean_gray, cv2.CV_32F)
    before_gradient_error = float(np.mean(np.abs(cv2.Laplacian(degraded_gray, cv2.CV_32F) - clean_gradient)))
    after_gradient_error = float(np.mean(np.abs(cv2.Laplacian(restored_gray, cv2.CV_32F) - clean_gradient)))
    return {
        "psnr_before_db": float(before_psnr),
        "psnr_after_db": float(after_psnr),
        "psnr_gain_db": float(after_psnr - before_psnr),
        "mae_before": _mae(degraded, clean),
        "mae_after": _mae(restored, clean),
        "gradient_error_reduction": float(
            1.0 - after_gradient_error / max(before_gradient_error, 1e-6)
        ),
    }


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, "RGB").save(path)


def _progress(fraction: float, message: str) -> None:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    filled = round(fraction * 28)
    print(
        f"\r[{'#' * filled}{'-' * (28 - filled)}] {fraction * 100:5.1f}% {message}",
        end="\n" if fraction >= 1.0 else "",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
