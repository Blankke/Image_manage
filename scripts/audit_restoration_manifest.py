"""审计专项恢复训练清单、配对文件和 group 泄漏。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/audit_restoration_manifest.py \
        --data-root "$SCREENRESTORE_DATA_ROOT" \
        --manifest "$SCREENRESTORE_DATA_ROOT/manifests/sidd.denoise.jsonl"

说明：仅接受相对 data-root 的图像路径，检查 JSONL 契约、图像可解码性、配对尺寸、
任务分布与 group/capture session 跨 split 泄漏。默认拒绝 private 路径；若操作者明确
授权本地受控采集集，可额外传入 --allow-private。报告不输出图像内容、文件名或像素。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

_TASKS = {
    "fidelity",
    "photometric",
    "reflection_single",
    "reflection_multiframe",
    "demoire",
    "dewarp",
    "super_resolution",
    "router",
}
_SPLITS = {"train", "validation", "test"}
_PATH_FIELDS = (
    "input_image",
    "target_image",
    "reflection_mask",
    "unresolved_mask",
    "validity_mask",
)
_REQUIRED = {
    "sample_id",
    "task",
    "split",
    "subject_id",
    "group_id",
    "capture_session",
    "reference_type",
    "alignment",
    "artifact_labels",
    "artifact_severity",
    "degradation_trace",
    "input_image",
    "target_image",
    "device",
    "source",
    "license",
    "license_restriction",
}
_OPTIONAL = {
    "observed_frames",
    "target_metadata",
    "degradation_parameters",
    "artifact_masks",
    "moire_metadata",
    "photometric_parameters",
    "lens_parameters",
    "dense_backward_grid",
    "camera_metadata",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="可选 JSON 审计报告路径")
    parser.add_argument("--allow-private", action="store_true", help="显式允许清单引用 private 目录")
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="仅检查清单结构和 split，不读取图像；正式训练前不应使用。",
    )
    args = parser.parse_args(argv)
    report = audit_manifest(
        args.data_root,
        args.manifest,
        allow_private=args.allow_private,
        check_images=not args.skip_image_check,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def audit_manifest(
    data_root: Path,
    manifest: Path,
    *,
    allow_private: bool = False,
    check_images: bool = True,
) -> dict[str, object]:
    """审计 JSONL；遇到无效监督或泄漏立即失败，避免污染后续训练结果。"""

    root = data_root.expanduser().resolve()
    manifest_path = manifest.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("data-root 不存在或不是目录")
    if not manifest_path.is_file():
        raise ValueError("manifest 不存在或不是文件")
    tasks: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    dimensions: Counter[str] = Counter()
    record_count = 0
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        record = _record(raw_line, line_number)
        _validate_record(record, line_number)
        task = str(record["task"])
        split = str(record["split"])
        tasks[task] += 1
        splits[split] += 1
        group_splits[str(record["group_id"])].add(split)
        session_splits[str(record["capture_session"])].add(split)
        subject_splits[str(record["subject_id"])].add(split)
        if check_images:
            input_shape = _image_shape(root, record["input_image"], line_number, allow_private)
            target_shape = _image_shape(root, record["target_image"], line_number, allow_private)
            _validate_pair_shape(task, input_shape, target_shape, line_number)
            dimensions[f"{input_shape[1]}x{input_shape[0]}"] += 1
            for field in ("reflection_mask", "validity_mask"):
                if record.get(field) is not None:
                    _mask_shape = _image_shape(root, record[field], line_number, allow_private)
                    if _mask_shape != input_shape:
                        raise ValueError(f"第 {line_number} 行 {field} 与 input_image 尺寸不一致")
            for path in record.get("observed_frames", []):
                frame_shape = _image_shape(root, path, line_number, allow_private)
                if frame_shape != input_shape:
                    raise ValueError(f"第 {line_number} 行 observed_frames 与 input_image 尺寸不一致")
        record_count += 1
    if record_count == 0:
        raise ValueError("manifest 没有有效记录")
    leaked_groups = sum(len(values) > 1 for values in group_splits.values())
    leaked_sessions = sum(len(values) > 1 for values in session_splits.values())
    leaked_subjects = sum(len(values) > 1 for values in subject_splits.values())
    if leaked_groups or leaked_sessions or leaked_subjects:
        raise ValueError(
            "检测到数据泄漏："
            f"subject_id={leaked_subjects}，group_id={leaked_groups}，"
            f"capture_session={leaked_sessions} 跨 split"
        )
    return {
        "format_version": 2,
        "kind": "restoration_manifest_audit",
        "status": "ok",
        "records": record_count,
        "task_counts": dict(sorted(tasks.items())),
        "split_counts": dict(sorted(splits.items())),
        "group_count": len(group_splits),
        "subject_count": len(subject_splits),
        "capture_session_count": len(session_splits),
        "image_checked": check_images,
        "input_size_counts": dict(sorted(dimensions.items())) if check_images else {},
        "private_access_explicit": allow_private,
    }


def _record(raw_line: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"第 {line_number} 行不是合法 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"第 {line_number} 行必须是对象")
    return value


def _validate_record(record: dict[str, Any], line_number: int) -> None:
    missing = sorted(_REQUIRED - record.keys())
    unknown = sorted(set(record) - (_REQUIRED | set(_PATH_FIELDS) | _OPTIONAL))
    if missing or unknown:
        raise ValueError(f"第 {line_number} 行契约不匹配：缺少={missing}，未知字段={unknown}")
    if record["task"] not in _TASKS or record["split"] not in _SPLITS:
        raise ValueError(f"第 {line_number} 行 task 或 split 不合法")
    scalar_fields = _REQUIRED - {
        "task",
        "split",
        "alignment",
        "artifact_labels",
        "artifact_severity",
        "degradation_trace",
    }
    for field in scalar_fields:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"第 {line_number} 行 {field} 必须是非空字符串")
    if not isinstance(record["alignment"], dict) or not {
        "method",
        "coordinate_space",
    }.issubset(record["alignment"]):
        raise ValueError(f"第 {line_number} 行 alignment 缺少 method/coordinate_space")
    if not isinstance(record["artifact_labels"], list):
        raise ValueError(f"第 {line_number} 行 artifact_labels 必须是列表")
    if not isinstance(record["artifact_severity"], dict):
        raise ValueError(f"第 {line_number} 行 artifact_severity 必须是对象")
    trace = record["degradation_trace"]
    trace_required = {"version", "seed", "target_stage", "identity", "artifacts", "steps"}
    if not isinstance(trace, dict) or not trace_required.issubset(trace):
        raise ValueError(f"第 {line_number} 行 degradation_trace 契约不完整")
    if not isinstance(trace["steps"], list):
        raise ValueError(f"第 {line_number} 行 degradation_trace.steps 必须是列表")
    if record["task"] == "reflection_multiframe":
        frames = record.get("observed_frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"第 {line_number} 行 reflection_multiframe 必须提供 observed_frames")
    if "observed_frames" in record and (
        not isinstance(record["observed_frames"], list)
        or any(not isinstance(value, str) or not value for value in record["observed_frames"])
    ):
        raise ValueError(f"第 {line_number} 行 observed_frames 必须是非空路径字符串列表")
    for field in _PATH_FIELDS:
        if field in record and record[field] is not None and not isinstance(record[field], str):
            raise ValueError(f"第 {line_number} 行 {field} 必须是字符串或 null")


def _image_shape(root: Path, relative: Any, line_number: int, allow_private: bool) -> tuple[int, int]:
    if not isinstance(relative, str):
        raise ValueError(f"第 {line_number} 行图像路径必须是字符串")
    path = _resolve_relative(root, relative, line_number, allow_private)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise ValueError(f"第 {line_number} 行图像无法解码")
    return int(image.shape[0]), int(image.shape[1])


def _resolve_relative(root: Path, value: str, line_number: int, allow_private: bool) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"第 {line_number} 行图像路径必须位于 data-root 内")
    if "private" in relative.parts and not allow_private:
        raise ValueError(f"第 {line_number} 行引用 private；需要显式传入 --allow-private")
    candidate = (root / relative).resolve()
    if not candidate.is_file() or root not in candidate.parents:
        raise ValueError(f"第 {line_number} 行图像不存在或越出 data-root")
    return candidate


def _validate_pair_shape(
    task: str,
    input_shape: tuple[int, int],
    target_shape: tuple[int, int],
    line_number: int,
) -> None:
    if task == "super_resolution":
        if target_shape[0] < input_shape[0] or target_shape[1] < input_shape[1]:
            raise ValueError(f"第 {line_number} 行 super_resolution target 尺寸必须不小于 input")
    elif input_shape != target_shape:
        raise ValueError(f"第 {line_number} 行同尺寸恢复任务的 input/target 尺寸不一致")


if __name__ == "__main__":
    raise SystemExit(main())
