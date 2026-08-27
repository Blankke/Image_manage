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
from typing import Any

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
        "--manifest",
        type=Path,
        help=(
            "标准 geometry JSONL；启用后先对 data-directory 内照片完成预测，再读取该清单打分"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="--manifest 的 image 相对根目录；SmartDoc 等外部数据清单必须显式指定",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="--manifest 模式下参与评分的数据 split，默认 test",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="允许少量样本运行，但报告会明确标记 smoke，不能作为发布证明",
    )
    args = parser.parse_args(argv)
    data_directory = args.data_directory.expanduser().resolve()
    detector = OnnxQuadDetector(args.quad_model) if args.quad_model is not None else None
    service = AutomaticGeometryService(detector)
    if args.manifest is None:
        case_reports, group_ids = _run_legacy_cases(service, data_directory, args.ground_truth)
    else:
        if args.dataset_root is None:
            raise ValueError("--manifest 模式必须提供 --dataset-root")
        case_reports, group_ids = _run_manifest_cases(
            service,
            data_directory,
            args.manifest,
            args.dataset_root,
            args.split,
        )
    _progress(len(case_reports), len(case_reports), "汇总 gate")
    gate = GeometryGate(minimum_samples=1 if args.smoke else 100)
    summary = aggregate_geometry_results(
        [report["metrics"] for report in case_reports],  # type: ignore[list-item]
        gate,
        group_ids,
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


def _run_legacy_cases(
    service: AutomaticGeometryService,
    data_directory: Path,
    ground_truth_path: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    """保留四张回归烟测，同时保证人工角点在预测完成后才读取。"""

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
    gt_values = json.loads(ground_truth_path.expanduser().resolve().read_text(encoding="utf-8"))
    case_reports: list[dict[str, object]] = []
    for case, photo_shape, decision in predictions:
        if case.name not in gt_values:
            raise ValueError(f"ground truth 缺少场景：{case.name}")
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
    return case_reports, [case.name for case, _shape, _decision in predictions]


def _run_manifest_cases(
    service: AutomaticGeometryService,
    data_directory: Path,
    manifest_path: Path,
    dataset_root: Path,
    split: str,
) -> tuple[list[dict[str, object]], list[str]]:
    """以通用 geometry 清单评测真实数据，且预测阶段不读取任何人工标签。

    ``data_directory`` 只用于扫描用户已明确指定的数据集照片。清单在所有照片预测完成后
    才读取，从而避免 ``content_quad``、类别或可见性标注进入自动定位路径。
    """

    photos = _find_photos(data_directory)
    if not photos:
        raise ValueError(f"数据目录中没有可评测图片：{data_directory}")
    predictions: dict[Path, tuple[tuple[int, ...], LocalizationDecision]] = {}
    for index, photo_path in enumerate(photos):
        _progress(index, len(photos), f"定位 {photo_path.name}")
        photo = load_image(photo_path).original_rgb
        # 不传 target hint，模型必须仅凭照片自行判断类别与内容层。
        predictions[photo_path.resolve()] = (photo.shape, service.localize(photo))

    # 预测冻结后才读取标准清单中的四角、类别与 split 标注。
    records = _read_manifest(manifest_path)
    case_reports: list[dict[str, object]] = []
    group_ids: list[str] = []
    for record in records:
        if record.get("split") != split:
            continue
        image_path = _resolve_manifest_image(dataset_root, str(record["image"]))
        predicted = predictions.get(image_path)
        if predicted is None:
            continue
        photo_shape, decision = predicted
        truth = _truth_from_manifest_record(record, photo_shape)
        case_reports.append(
            {
                "case": str(record.get("id", image_path.name)),
                "photo": str(record["image"]),
                "decision": decision.to_dict(photo_shape),
                "metrics": evaluate_geometry_decision(decision, truth),
            }
        )
        group_ids.append(str(record["group_id"]))
    if not case_reports:
        raise ValueError("没有找到同时位于 data-directory 与指定 manifest split 的图片")
    return case_reports, group_ids


def _resolve_manifest_image(dataset_root: Path, image: str) -> Path:
    """解析数据根相对 image，并拒绝清单越出显式数据集范围。"""

    relative = Path(image)
    if relative.is_absolute():
        raise ValueError("manifest image 必须是 dataset-root 相对路径")
    root = dataset_root.expanduser().resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("manifest image 不能越出 dataset-root")
    return resolved


def _find_photos(data_directory: Path) -> list[Path]:
    """只枚举显式数据目录中的常见图片格式，避免隐式扫描 private 数据。"""

    if not data_directory.is_dir():
        raise ValueError(f"数据目录不存在：{data_directory}")
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return sorted(
        (path for path in data_directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.as_posix(),
    )


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """读取已冻结预测的标准 geometry JSONL，并验证评测所需的最小字段。"""

    path = manifest_path.expanduser().resolve()
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest 第 {line_number} 行不是合法 JSON") from exc
            required = {"image", "split", "present", "target_class", "group_id"}
            missing = required - record.keys() if isinstance(record, dict) else required
            if not isinstance(record, dict) or missing:
                raise ValueError(f"manifest 第 {line_number} 行缺少字段：{sorted(missing)}")
            records.append(record)
    return records


def _truth_from_manifest_record(
    record: dict[str, Any],
    image_shape: tuple[int, ...],
) -> GeometryGroundTruth:
    """将 `[0,1]` 标注转换为像素真值；负样本只检查自动拒绝。"""

    present = bool(record["present"])
    try:
        target_class = TargetClass(str(record["target_class"]))
    except ValueError as exc:
        raise ValueError(f"未知 target_class：{record['target_class']!r}") from exc
    if not present:
        if target_class != TargetClass.NONE:
            raise ValueError("present=false 时 target_class 必须为 none")
        return GeometryGroundTruth(None, TargetClass.NONE)
    if target_class == TargetClass.NONE:
        raise ValueError("present=true 时 target_class 不能为 none")
    normalized = np.asarray(record.get("content_quad"), dtype=np.float32)
    if normalized.shape != (4, 2) or np.any(~np.isfinite(normalized)) or np.any((normalized < 0) | (normalized > 1)):
        raise ValueError("content_quad 必须是 [0,1] 范围内的 4×2 数组")
    scale = np.array([max(1, image_shape[1] - 1), max(1, image_shape[0] - 1)], np.float32)
    in_scope = bool(record.get("in_scope", record.get("visible", True)))
    return GeometryGroundTruth(normalized * scale, target_class, in_scope=in_scope)


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
