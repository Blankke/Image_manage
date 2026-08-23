"""运行严格 photo-only 的自动几何端到端基准。

使用范例：
    source .venv/bin/activate
    which python
    python -m benchmarks.geometry_e2e.run --data-directory 测试数据 --smoke
    python -m benchmarks.geometry_e2e.run --quad-model models/weights/quadlocator-s.onnx

推理阶段只打开实拍图。``oracle_corners`` 在定位决策返回后才用于打分；clean reference
路径从未传入本模块。脚本始终显示逐样本进度条，并将 JSON 写入显式输出路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from screenrestore.geometry import (
    AutomaticGeometryService,
    LocalizationDecision,
    OnnxQuadDetector,
    TargetClass,
)
from screenrestore.io.image_loader import load_image
from screenrestore.validation import (
    GeometryGate,
    GeometryGroundTruth,
    aggregate_geometry_results,
    evaluate_geometry_decision,
)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    photo: str
    target_class: TargetClass


CASES = (
    Case("后台芭蕾", "后台芭蕾.jpg", TargetClass.ARTWORK),
    Case("复古街头", "复古街头.jpg", TargetClass.ARTWORK),
    Case("电脑屏幕", "电脑屏幕.jpg", TargetClass.SCREEN),
    Case("红发女子", "红发女子.jpg", TargetClass.SCREEN),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-directory", type=Path, default=root / "测试数据")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=root / "benchmarks" / "ground_truth" / "targets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "evaluation" / "geometry_e2e.json",
    )
    parser.add_argument("--quad-model", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="允许少量样本运行，但报告会明确标记 smoke，不能作为发布证明",
    )
    args = parser.parse_args(argv)
    data_directory = args.data_directory.expanduser().resolve()
    detector = OnnxQuadDetector(args.quad_model) if args.quad_model is not None else None
    service = AutomaticGeometryService(detector)
    predictions: list[tuple[Case, tuple[int, ...], LocalizationDecision]] = []
    available = [case for case in CASES if (data_directory / case.photo).is_file()]
    if not available:
        raise ValueError("没有发现可运行的 photo-only 几何样本")
    for index, case in enumerate(available):
        _progress(index, len(available), f"定位 {case.name}")
        # 唯一图像读取点：这里只读取实拍 photo，不构造 reference 路径。
        photo = load_image(data_directory / case.photo).original_rgb
        decision = service.localize(photo, case.target_class)
        predictions.append((case, photo.shape, decision))

    # 所有预测冻结后才加载人工标签，代码结构上隔离 oracle 对推理的影响。
    gt_values = json.loads(args.ground_truth.expanduser().resolve().read_text(encoding="utf-8"))
    case_reports: list[dict[str, object]] = []
    for case, photo_shape, raw_decision in predictions:
        if case.name not in gt_values:
            raise ValueError(f"ground truth 缺少场景：{case.name}")
        decision = raw_decision
        truth = GeometryGroundTruth(
            np.asarray(gt_values[case.name]["oracle_corners"], dtype=np.float32),
            case.target_class,
        )
        case_reports.append(
            {
                "case": case.name,
                "photo": case.photo,
                "decision": decision.to_dict(photo_shape),
                "metrics": evaluate_geometry_decision(decision, truth),
            }
        )
    _progress(len(available), len(available), "汇总 gate")
    gate = GeometryGate(minimum_samples=1 if args.smoke else 100)
    summary = aggregate_geometry_results(
        [report["metrics"] for report in case_reports],  # type: ignore[list-item]
        gate,
    )
    report = {
        "protocol": "e2e_auto",
        "protocol_version": 1,
        "run_kind": "smoke" if args.smoke else "release_gate",
        "inference_inputs": ["photo_rgb"],
        "forbidden_inference_inputs": ["clean_reference", "oracle_corners"],
        "oracle_loaded_after_all_predictions": True,
        "backend": "quadlocator_onnx" if args.quad_model is not None else "classic_fallback",
        "summary": summary,
        "cases": case_reports,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if summary["status"] == "PASS" else 1


def _progress(done: int, total: int, message: str) -> None:
    width = 24
    fraction = min(1.0, done / max(1, total))
    filled = round(width * fraction)
    ending = "\n" if done >= total else "\r"
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {message}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
