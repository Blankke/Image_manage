#!/usr/bin/env python3
"""ScreenRestore 私人开发验证集的准备、冻结预测、标注、评分与恢复入口。

使用范例（所有 Python 命令均在项目虚拟环境中执行）：
    source .venv/bin/activate
    which python
    python scripts/private_development_validation.py prepare \
      --data-root /Users/caozichen/screenrestore-data \
      --image-directory /Users/caozichen/screenrestore-data/private-set-extract/private_set \
      --output-directory output/private-validation/b0/prepare \
      --annotations /Users/caozichen/screenrestore-data/private-validation/annotations.jsonl
    python scripts/private_development_validation.py freeze \
      --data-root /Users/caozichen/screenrestore-data \
      --index output/private-validation/b0/prepare/dataset-index.json \
      --quad-model /path/to/quadlocator-s.onnx \
      --output-directory output/private-validation/b0/frozen-predictions
    python scripts/private_development_validation.py label \
      --data-root /Users/caozichen/screenrestore-data \
      --index output/private-validation/b0/prepare/dataset-index.json \
      --annotations /Users/caozichen/screenrestore-data/private-validation/annotations.jsonl
    python scripts/private_development_validation.py score \
      --index output/private-validation/b0/prepare/dataset-index.json \
      --predictions output/private-validation/b0/frozen-predictions/predictions.json \
      --annotations /Users/caozichen/screenrestore-data/private-validation/annotations.jsonl \
      --output-directory output/private-validation/b0/score
    python scripts/private_development_validation.py review \
      --data-root /Users/caozichen/screenrestore-data \
      --index output/private-validation/b0/prepare/dataset-index.json \
      --predictions output/private-validation/b0/frozen-predictions/predictions.json \
      --annotations /Users/caozichen/screenrestore-data/private-validation/annotations.jsonl \
      --output-directory output/private-validation/b0/annotation-review
    python scripts/private_development_validation.py restore \
      --data-root /Users/caozichen/screenrestore-data \
      --index output/private-validation/b0/prepare/dataset-index.json \
      --predictions output/private-validation/b0/frozen-predictions/predictions.json \
      --annotations /Users/caozichen/screenrestore-data/private-validation/annotations.jsonl \
      --output-directory output/private-validation/b0/restored

约束：私人图片和标注只允许位于显式 data-root 内；评估产物只允许写入仓库已忽略的
``output/``。freeze 在读取任何人工四角前完成，score/restore 会核对数据索引和图片摘要，
防止把另一版图片、预测或标注混在同一份结论中。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenrestore.core.operator import ProcessingContext  # noqa: E402
from screenrestore.core.presets import (  # noqa: E402
    PresetId,
    ProcessingMode,
    apply_preset,
    apply_processing_mode,
    build_default_pipeline,
)
from screenrestore.geometry import (  # noqa: E402
    AspectRatioMode,
    AutomaticGeometryService,
    ConfidencePolicy,
    CorrectnessCalibrator,
    ModelAgreementQuadDetector,
    OnnxQuadDetector,
    TargetClass,
    order_corners,
)
from screenrestore.io.image_loader import load_image  # noqa: E402
from screenrestore.provenance import (  # noqa: E402
    ArchiveVariant,
    ProvenanceMap,
    ProvenanceReport,
)
from screenrestore.validation.geometry_benchmark import (  # noqa: E402
    GeometryGate,
    aggregate_geometry_results,
    corner_metrics,
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DOMAIN_BY_DIRECTORY = {
    "magazine": "artwork",
    "postcards": "postcard",
    "posters": "poster",
    "screen": "screen",
}
TARGET_CLASS_BY_DOMAIN = {
    "artwork": "artwork",
    "postcard": "postcard",
    "poster": "artwork",
    "screen": "screen",
}
PRESET_BY_DOMAIN = {
    "artwork": PresetId.ARTWORK,
    "postcard": PresetId.ARTWORK,
    "poster": PresetId.ARTWORK,
    "screen": PresetId.DISPLAY,
}
CAPTURE_VIEW_BY_NUMBER = {
    1: "frontal",
    2: "left_oblique",
    3: "right_oblique",
    4: "forward_oblique",
    5: "backward_oblique",
    6: "far",
}
GLARE_LEVELS = ("none", "light", "medium", "strong")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_data_path(path: Path, data_root: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    root = data_root.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"私人数据路径必须位于 data-root 内：{resolved}")
    if must_exist and not resolved.exists():
        raise ValueError(f"路径不存在：{resolved}")
    return resolved


def _ensure_output_directory(path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    resolved = path.expanduser().resolve()
    allowed = repo / "output"
    if not resolved.is_relative_to(allowed):
        raise ValueError("验证产物必须写入仓库 output/ 目录")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def discover_frames(image_directory: Path, data_root: Path) -> list[dict[str, Any]]:
    """按 ``domain/object/1..6`` 建立对象级 group 与逐帧记录。"""

    root = _ensure_data_path(image_directory, data_root, must_exist=True)
    records: list[dict[str, Any]] = []
    object_directories = sorted(
        directory
        for directory in root.rglob("*")
        if directory.is_dir()
        and any(path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir())
    )
    for object_index, directory in enumerate(object_directories, start=1):
        relative_object = directory.relative_to(root)
        if len(relative_object.parts) != 2 or relative_object.parts[0] not in DOMAIN_BY_DIRECTORY:
            raise ValueError(f"对象目录必须符合 domain/object：{relative_object.as_posix()}")
        images = sorted(
            (path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: int(path.stem) if path.stem.isdigit() else 10_000,
        )
        numbers = [int(path.stem) for path in images if path.stem.isdigit()]
        if len(images) != 6 or numbers != list(CAPTURE_VIEW_BY_NUMBER):
            raise ValueError(f"每个对象必须且只能包含 1..6 六张图：{relative_object.as_posix()}")
        domain = DOMAIN_BY_DIRECTORY[relative_object.parts[0]]
        stable = relative_object.as_posix().lower()
        group_id = f"private-dev:{_token(stable)}"
        source_id = f"private-source:{_token('source:' + stable)}"
        for path in images:
            number = int(path.stem)
            document = load_image(path)
            relative_image = path.resolve().relative_to(data_root.resolve()).as_posix()
            records.append(
                {
                    "image_id": f"{group_id}:view-{number}",
                    "image": relative_image,
                    "image_sha256": _sha256(path),
                    "width": document.width,
                    "height": document.height,
                    "domain": domain,
                    "target_class": TARGET_CLASS_BY_DOMAIN[domain],
                    "group_id": group_id,
                    "subject_id": f"private-subject:{_token('subject:' + stable)}",
                    "digital_source_id": source_id,
                    "capture_session": f"private-session:{_token('session:' + stable)}",
                    "capture_number": number,
                    "capture_view": CAPTURE_VIEW_BY_NUMBER[number],
                    "object_order": object_index,
                }
            )
        print(
            f"[准备 {object_index}/{len(object_directories)}] {relative_object.as_posix()}",
            file=sys.stderr,
        )
    if not records:
        raise ValueError("未发现私人验证图片")
    return records


def _dataset_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    core = {
        "kind": "screenrestore_private_development_index",
        "format_version": 1,
        "split": "validation",
        "sample_count": len(records),
        "independent_group_count": len({row["group_id"] for row in records}),
        "domain_groups": {
            domain: len({row["group_id"] for row in records if row["domain"] == domain})
            for domain in TARGET_CLASS_BY_DOMAIN
        },
        "frames": records,
    }
    return {**core, "dataset_sha256": _json_sha256(core)}


def _draft_annotation(row: dict[str, Any]) -> dict[str, Any]:
    nested = row["domain"] == "screen" or "whiteedge" in row["image"].lower()
    capture_conditions = [
        "frontal" if row["capture_number"] == 1 else "mild_perspective",
    ]
    if row["capture_number"] in (4, 5, 6):
        capture_conditions = ["moderate_perspective"]
    if nested:
        capture_conditions.append("nested_layer")
    return {
        **row,
        "split": "validation",
        "device": "private-phone-unknown",
        "present": True,
        "in_scope": True,
        "content_quad": None,
        "outer_quad": None,
        "visible": True,
        "occlusion": 0.0,
        "glare_level": "none",
        "scene_type": "poster" if row["domain"] == "poster" else row["domain"],
        "capture_conditions": capture_conditions,
        "source": "private-development-validation",
        "annotation_status": "pending",
    }


def prepare(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    image_directory = _ensure_data_path(args.image_directory, data_root, must_exist=True)
    annotations = _ensure_data_path(args.annotations, data_root, must_exist=False)
    output = _ensure_output_directory(args.output_directory)
    records = discover_frames(image_directory, data_root)
    index = _dataset_index(records)
    _write_json(output / "dataset-index.json", index)
    if annotations.exists():
        existing = {row["image_id"]: row for row in _read_jsonl(annotations)}
        if set(existing) != {row["image_id"] for row in records}:
            raise ValueError("既有标注与当前数据索引不一致；请另存新标注，禁止静默混用")
        for row in records:
            if existing[row["image_id"]].get("image_sha256") != row["image_sha256"]:
                raise ValueError("既有标注绑定的图片摘要已变化")
        annotation_action = "preserved"
    else:
        _write_jsonl_atomic(annotations, (_draft_annotation(row) for row in records))
        annotation_action = "created_pending_draft"
    result = {
        "status": "READY_FOR_FROZEN_INFERENCE",
        "dataset_sha256": index["dataset_sha256"],
        "sample_count": index["sample_count"],
        "independent_group_count": index["independent_group_count"],
        "domain_groups": index["domain_groups"],
        "annotations": str(annotations),
        "annotation_action": annotation_action,
        "release_readiness": "BLOCKED: fewer_than_100_independent_groups",
        "usage": "development_validation_only",
    }
    _write_json(output / "result.json", result)
    print(output / "result.json")
    return 0


def _load_index(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.pop("dataset_sha256", None)
    actual = _json_sha256(value)
    value["dataset_sha256"] = expected
    if value.get("kind") != "screenrestore_private_development_index" or expected != actual:
        raise ValueError("私人验证索引格式或摘要无效")
    if value.get("sample_count") != len(value.get("frames", [])):
        raise ValueError("私人验证索引计数不一致")
    return value


def _verify_frame(path: Path, row: dict[str, Any]) -> None:
    if not path.is_file() or _sha256(path) != row["image_sha256"]:
        raise ValueError(f"图片缺失或摘要变化：{row['image_id']}")


def _draw_prediction(image_rgb: np.ndarray, decision: dict[str, Any], destination: Path) -> None:
    preview = image_rgb.copy()
    height, width = preview.shape[:2]
    corners = decision.get("corners")
    if corners is not None:
        points = np.rint(np.asarray(corners) * [width - 1, height - 1]).astype(np.int32)
        color = (30, 210, 80) if decision["accepted"] else (235, 70, 60)
        cv2.polylines(preview, [points], True, color, max(3, width // 800))
        for index, point in enumerate(points):
            cv2.putText(preview, str(index), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    maximum = 1400
    scale = min(1.0, maximum / max(height, width))
    if scale < 1.0:
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Image.fromarray(preview).save(destination, quality=90)


def freeze(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    index = _load_index(args.index.expanduser().resolve())
    model = args.quad_model.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"QuadLocator ONNX 不存在：{model}")
    output = _ensure_output_directory(args.output_directory)
    overlay_directory = output / "overlays"
    overlay_directory.mkdir()
    calibrator = None
    calibrator_identity = None
    if args.calibrator is not None:
        calibrator_path = args.calibrator.expanduser().resolve()
        if not calibrator_path.is_file():
            raise ValueError(f"correctness calibrator 不存在：{calibrator_path}")
        calibrator = CorrectnessCalibrator.load(calibrator_path)
        calibrator_identity = {
            "path": str(calibrator_path),
            "sha256": _sha256(calibrator_path),
            "threshold": calibrator.threshold,
        }
    policy = ConfidencePolicy(calibrator=calibrator)
    detector = OnnxQuadDetector(model)
    detector_configuration: dict[str, Any] = {"kind": "quadlocator_onnx"}
    if args.classic_agreement_snap:
        detector = ModelAgreementQuadDetector(
            detector,
            minimum_agreement_iou=args.classic_agreement_min_iou,
            maximum_corner_nce=args.classic_agreement_max_corner_nce,
        )
        detector_configuration = {
            "kind": "quadlocator_onnx_with_classic_agreement",
            "minimum_agreement_iou": args.classic_agreement_min_iou,
            "maximum_corner_nce": args.classic_agreement_max_corner_nce,
            "runtime_gate": (
                "model_refine_rejected_and_agreement_refine_accepted_and_"
                "boundary_support_improved_and_mask_consistency_improved"
            ),
            "acceptance_evidence": "model_original_branch",
        }
    service = AutomaticGeometryService(detector, policy=policy)
    predictions = []
    frames = index["frames"]
    for number, row in enumerate(frames, start=1):
        path = _ensure_data_path(data_root / row["image"], data_root, must_exist=True)
        _verify_frame(path, row)
        document = load_image(path)
        hint = TargetClass(row["target_class"])
        decision = service.localize(document.original_rgb, hint).to_dict(
            document.original_rgb.shape
        )
        predictions.append(
            {
                "image_id": row["image_id"],
                "image_sha256": row["image_sha256"],
                "group_id": row["group_id"],
                "domain": row["domain"],
                "capture_number": row["capture_number"],
                "decision": decision,
            }
        )
        _draw_prediction(
            document.original_rgb, decision, overlay_directory / f"sample-{number:03d}.jpg"
        )
        print(f"[冻结预测 {number}/{len(frames)}] {decision['status']}", file=sys.stderr)
    counts = Counter(item["decision"]["status"] for item in predictions)
    rejection_reasons = Counter(
        reason for item in predictions for reason in item["decision"].get("rejection_reasons", [])
    )
    predicted_classes = Counter(item["decision"]["target_class"] for item in predictions)
    confidences = np.asarray([item["decision"]["confidence"] for item in predictions], np.float64)
    payload = {
        "kind": "screenrestore_private_frozen_predictions",
        "format_version": 1,
        "dataset_sha256": index["dataset_sha256"],
        "model": {"path": str(model), "sha256": _sha256(model)},
        "detector_configuration": detector_configuration,
        "correctness_calibrator": calibrator_identity,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "predictions": predictions,
    }
    payload["prediction_sha256"] = _json_sha256(payload)
    _write_json(output / "predictions.json", payload)
    _write_json(
        output / "result.json",
        {
            "status": "FROZEN",
            "dataset_sha256": index["dataset_sha256"],
            "prediction_sha256": payload["prediction_sha256"],
            "model_sha256": payload["model"]["sha256"],
            "detector_configuration": detector_configuration,
            "correctness_calibrator": calibrator_identity,
            "sample_count": len(predictions),
            "decision_counts": dict(counts),
            "predicted_class_counts": dict(predicted_classes),
            "rejection_reason_counts": dict(rejection_reasons),
            "confidence": {
                "minimum": round(float(confidences.min()), 6),
                "median": round(float(np.median(confidences)), 6),
                "maximum": round(float(confidences.max()), 6),
            },
            "ground_truth_read": False,
        },
    )
    print(output / "result.json")
    return 0


def _scaled_image(
    image_rgb: np.ndarray, maximum_width: int = 1500, maximum_height: int = 900
) -> tuple[np.ndarray, float]:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, maximum_width / width, maximum_height / height)
    if scale == 1.0:
        return image_rgb.copy(), scale
    return cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _label_one(image_rgb: np.ndarray, record: dict[str, Any], position: int, total: int) -> str:
    window = "ScreenRestore private development labeling"
    base, scale = _scaled_image(image_rgb)
    mode: str | None = None
    pending: list[tuple[int, int]] = []

    def mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        nonlocal mode, pending
        if event != cv2.EVENT_LBUTTONDOWN or mode is None:
            return
        pending.append((x, y))
        if len(pending) != 4:
            return
        full = np.asarray(pending, np.float32) / scale
        ordered = order_corners(full)
        normalized = ordered / [image_rgb.shape[1] - 1, image_rgb.shape[0] - 1]
        record[f"{mode}_quad"] = np.clip(normalized, 0.0, 1.0).astype(float).tolist()
        mode, pending = None, []

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    while True:
        canvas = base.copy()
        for key, color in (("content_quad", (0, 255, 255)), ("outer_quad", (255, 0, 255))):
            if record.get(key) is None:
                continue
            points = np.rint(
                np.asarray(record[key]) * [image_rgb.shape[1] - 1, image_rgb.shape[0] - 1] * scale
            ).astype(np.int32)
            cv2.polylines(canvas, [points], True, color, 3)
            for corner, point in enumerate(points):
                cv2.putText(
                    canvas, str(corner), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
        for point in pending:
            cv2.circle(canvas, point, 6, (255, 255, 0), -1)
        lines = [
            f"{position}/{total} {record['domain']} view={record['capture_number']}",
            f"mode={mode or '-'} glare={record['glare_level']} in_scope={record['in_scope']}",
            "C content | O outer | G glare | U out-of-scope | R approve | S save | Q quit",
        ]
        for line_number, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (18, 32 + line_number * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (30, 220, 30),
                2,
            )
        cv2.imshow(window, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("c"), ord("C")):
            mode, pending = "content", []
        elif key in (ord("o"), ord("O")):
            mode, pending = "outer", []
        elif key in (ord("g"), ord("G")):
            current = GLARE_LEVELS.index(record["glare_level"])
            record["glare_level"] = GLARE_LEVELS[(current + 1) % len(GLARE_LEVELS)]
        elif key in (ord("u"), ord("U")):
            record["in_scope"] = not bool(record["in_scope"])
            record["present"] = bool(record["in_scope"])
            if record["in_scope"]:
                record["target_class"] = TARGET_CLASS_BY_DOMAIN[record["domain"]]
            else:
                record["target_class"] = "none"
                record["content_quad"] = None
                record["outer_quad"] = None
        elif key in (8, 127) and pending:
            pending.pop()
        elif key in (ord("r"), ord("R")):
            if record["in_scope"] and record.get("content_quad") is None:
                continue
            record["annotation_status"] = "approved"
            return "save"
        elif key in (ord("s"), ord("S")):
            record["annotation_status"] = "pending_review"
            return "save"
        elif key in (ord("q"), ord("Q")):
            return "quit"


def label(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    index = _load_index(args.index.expanduser().resolve())
    annotations_path = _ensure_data_path(args.annotations, data_root, must_exist=True)
    rows = _read_jsonl(annotations_path)
    by_id = {row["image_id"]: row for row in rows}
    if set(by_id) != {row["image_id"] for row in index["frames"]}:
        raise ValueError("标注与数据索引不一致")
    for position, frame in enumerate(index["frames"], start=1):
        record = by_id[frame["image_id"]]
        if record.get("annotation_status") == "approved" and not args.relabel_approved:
            print(f"[标注 {position}/{len(rows)}] 已批准，跳过", file=sys.stderr)
            continue
        if record.get("image_sha256") != frame["image_sha256"]:
            raise ValueError("标注绑定的图片摘要已变化")
        path = _ensure_data_path(data_root / frame["image"], data_root, must_exist=True)
        document = load_image(path)
        action = _label_one(document.original_rgb, record, position, len(rows))
        _write_jsonl_atomic(annotations_path, (by_id[item["image_id"]] for item in index["frames"]))
        print(f"[标注 {position}/{len(rows)}] {record['annotation_status']}", file=sys.stderr)
        if action == "quit":
            break
    cv2.destroyAllWindows()
    return 0


def _load_bound_inputs(
    index_path: Path,
    prediction_path: Path,
    annotation_path: Path,
    *,
    require_approved: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    index = _load_index(index_path)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    claimed = predictions.pop("prediction_sha256", None)
    actual = _json_sha256(predictions)
    predictions["prediction_sha256"] = claimed
    if claimed != actual or predictions.get("dataset_sha256") != index["dataset_sha256"]:
        raise ValueError("冻结预测摘要无效或不属于当前数据索引")
    by_prediction = {row["image_id"]: row for row in predictions["predictions"]}
    annotations = _read_jsonl(annotation_path)
    expected_ids = {row["image_id"] for row in index["frames"]}
    if (
        set(by_prediction) != expected_ids
        or {row["image_id"] for row in annotations} != expected_ids
    ):
        raise ValueError("索引、冻结预测与标注的帧集合不一致")
    if require_approved and any(row.get("annotation_status") != "approved" for row in annotations):
        raise ValueError("所有帧必须完成四角标注并在标注器中按 R 批准")
    for row in annotations:
        prediction = by_prediction[row["image_id"]]
        if prediction["image_sha256"] != row["image_sha256"]:
            raise ValueError("预测、标注绑定的图片摘要不一致")
        if (
            row.get("annotation_status") == "approved"
            and row.get("in_scope")
            and row.get("content_quad") is None
        ):
            raise ValueError("in-scope 标注缺少 content_quad")
    return index, by_prediction, annotations


def score_rows(
    predictions: dict[str, dict[str, Any]], annotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for truth in annotations:
        decision = predictions[truth["image_id"]]["decision"]
        accepted = bool(decision["accepted"])
        predicted = decision.get("corners")
        in_scope = bool(truth["in_scope"])
        nce, iou, maximum = 1.0, 0.0, None
        strict = not in_scope and not accepted
        layer_correct = True
        outer_evaluable = False
        candidate_closer_to_outer: bool | None = None
        if predicted is not None and truth.get("content_quad") is not None:
            scale = np.array([truth["width"] - 1, truth["height"] - 1], np.float32)
            predicted_px = np.asarray(predicted, np.float32) * scale
            content_px = np.asarray(truth["content_quad"], np.float32) * scale
            nce, iou, maximum_value = corner_metrics(predicted_px, content_px)
            maximum = round(float(maximum_value), 4)
            class_correct = decision["target_class"] == truth["target_class"]
            strict = bool(in_scope and class_correct and nce <= 0.01 and iou >= 0.93)
            outer = truth.get("outer_quad")
            if outer is not None:
                _outer_nce, outer_iou, _outer_max = corner_metrics(
                    predicted_px, np.asarray(outer, np.float32) * scale
                )
                outer_evaluable = True
                candidate_closer_to_outer = bool(outer_iou > iou and not strict)
                layer_correct = not candidate_closer_to_outer
            elif not strict:
                # 缺少 outer GT 时不能把错层率当作已知为零；对未达到 content strict
                # 几何的候选保持保守失败语义。
                layer_correct = False
        results.append(
            {
                "image_id": truth["image_id"],
                "group_id": truth["group_id"],
                "domain": truth["domain"],
                "capture_number": truth["capture_number"],
                "capture_view": truth["capture_view"],
                "has_candidate": predicted is not None,
                "outer_evaluable": outer_evaluable,
                "candidate_closer_to_outer": candidate_closer_to_outer,
                "accepted": accepted,
                "correct": strict,
                "strict_correct": strict,
                "class_correct": decision["target_class"] == truth["target_class"],
                "layer_correct": layer_correct,
                "in_scope": in_scope,
                "corner_nce": round(float(nce), 8),
                "quad_iou": round(float(iou), 8),
                "max_corner_error_px": maximum,
                "confidence": decision["confidence"],
                "rejection_reasons": decision["rejection_reasons"],
            }
        )
    return results


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_geometry_results(results, GeometryGate(), [row["group_id"] for row in results])


def _candidate_geometry(results: list[dict[str, Any]]) -> dict[str, Any]:
    """独立统计所有候选四角；拒绝决策不应掩盖 raw geometry 的实际质量。"""

    candidates = [row for row in results if row["has_candidate"] and row["in_scope"]]
    outer_evaluable = [row for row in candidates if row["outer_evaluable"]]
    nce = np.asarray([row["corner_nce"] for row in candidates], np.float64)
    iou = np.asarray([row["quad_iou"] for row in candidates], np.float64)
    return {
        "candidate_count": len(candidates),
        "candidate_coverage": round(len(candidates) / max(1, len(results)), 8),
        "corner_nce_median": round(float(np.median(nce)), 8) if nce.size else None,
        "corner_nce_p95": round(float(np.percentile(nce, 95)), 8) if nce.size else None,
        "quad_iou_median": round(float(np.median(iou)), 8) if iou.size else None,
        "quad_iou_p05": round(float(np.percentile(iou, 5)), 8) if iou.size else None,
        "strict_geometry_rate": round(
            sum(row["corner_nce"] <= 0.01 and row["quad_iou"] >= 0.93 for row in candidates)
            / max(1, len(candidates)),
            8,
        ),
        "outer_evaluable_count": len(outer_evaluable),
        "candidate_closer_to_outer_count": sum(
            row["candidate_closer_to_outer"] is True for row in outer_evaluable
        ),
        "candidate_closer_to_outer_rate": round(
            sum(row["candidate_closer_to_outer"] is True for row in outer_evaluable)
            / max(1, len(outer_evaluable)),
            8,
        ),
    }


def score(args: argparse.Namespace) -> int:
    index, predictions, annotations = _load_bound_inputs(
        args.index.expanduser().resolve(),
        args.predictions.expanduser().resolve(),
        args.annotations.expanduser().resolve(),
        require_approved=False,
    )
    output = _ensure_output_directory(args.output_directory)
    annotation_counts = Counter(row.get("annotation_status", "missing") for row in annotations)
    if annotation_counts.get("approved", 0) != len(annotations):
        decisions = Counter(row["decision"]["status"] for row in predictions.values())
        report = {
            "kind": "screenrestore_private_development_score",
            "format_version": 1,
            "status": "BLOCKED: annotations_pending",
            "dataset_sha256": index["dataset_sha256"],
            "usage": "development_validation_only",
            "sample_count": len(annotations),
            "independent_group_count": index["independent_group_count"],
            "annotation_status_counts": dict(annotation_counts),
            "frozen_decision_counts": dict(decisions),
            "ground_truth_metrics": None,
            "release_verdict": "BLOCKED",
            "release_limitation": (
                "人工四角尚未全部批准；且开发集只有 13 个独立对象，少于 release gate 的 100 个 group"
            ),
        }
        _write_json(output / "scorecard.json", report)
        print(output / "scorecard.json")
        return 2
    rows = score_rows(predictions, annotations)
    overall = _aggregate(rows)
    development_gates = {
        name: passed for name, passed in overall["gates"].items() if name != "minimum_samples"
    }
    development_verdict = "PASS" if all(development_gates.values()) else "FAIL"
    by_domain = {
        domain: _aggregate([row for row in rows if row["domain"] == domain])
        for domain in TARGET_CLASS_BY_DOMAIN
    }
    by_view = {
        view: _aggregate([row for row in rows if row["capture_view"] == view])
        for view in CAPTURE_VIEW_BY_NUMBER.values()
    }
    report = {
        "kind": "screenrestore_private_development_score",
        "format_version": 1,
        "status": overall["status"],
        "dataset_sha256": index["dataset_sha256"],
        "usage": "development_validation_only",
        "overall": overall,
        "development_verdict": development_verdict,
        "development_gates": development_gates,
        "candidate_geometry": _candidate_geometry(rows),
        "candidate_geometry_by_domain": {
            domain: _candidate_geometry([row for row in rows if row["domain"] == domain])
            for domain in TARGET_CLASS_BY_DOMAIN
        },
        "candidate_geometry_by_capture_view": {
            view: _candidate_geometry([row for row in rows if row["capture_view"] == view])
            for view in CAPTURE_VIEW_BY_NUMBER.values()
        },
        "by_domain": by_domain,
        "by_capture_view": by_view,
        "release_verdict": (
            "PASS"
            if overall["status"] == "PASS"
            else "FAIL"
            if development_verdict == "FAIL"
            else "BLOCKED"
        ),
        "release_limitation": "开发集只有 13 个独立对象，少于 release gate 的 100 个 group",
        "rows": rows,
    }
    _write_json(output / "scorecard.json", report)
    print(output / "scorecard.json")
    return 0 if overall["status"] == "PASS" else 2


def _configure_pipeline(domain: str, corners: list[list[float]]) -> Any:
    pipeline = build_default_pipeline()
    apply_preset(pipeline, PRESET_BY_DOMAIN[domain])
    apply_processing_mode(pipeline, ProcessingMode.FIDELITY)
    params = pipeline.state("geometry").params.to_dict()
    params.update(
        {"corners": corners, "ratio_mode": AspectRatioMode.AUTO.value, "auto_crop": False}
    )
    pipeline.update_parameters("geometry", params)
    return pipeline


def _save_rgb(path: Path, image: np.ndarray) -> None:
    values = (
        image
        if image.dtype == np.uint8
        else np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    )
    Image.fromarray(values).save(path)


def _draw_quad(
    image_rgb: np.ndarray,
    quad: list[list[float]] | None,
    color: tuple[int, int, int],
    prefix: str,
) -> None:
    """在 RGB 图像上绘制归一化四角和固定角点顺序。"""
    if quad is None:
        return
    height, width = image_rgb.shape[:2]
    points = np.rint(np.asarray(quad, np.float32) * [width - 1, height - 1]).astype(np.int32)
    thickness = max(3, width // 900)
    cv2.polylines(image_rgb, [points], True, color, thickness)
    for index, point in enumerate(points):
        cv2.circle(image_rgb, tuple(point), thickness * 2, color, -1)
        cv2.putText(
            image_rgb,
            f"{prefix}{index}",
            tuple(point + np.array([6, -6])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            max(2, thickness - 1),
        )


def _review_tile(
    image_rgb: np.ndarray,
    truth: dict[str, Any],
    decision: dict[str, Any],
    *,
    tile_width: int = 480,
    tile_height: int = 640,
) -> Image.Image:
    """生成单帧人工/模型四角对照图；绿色 content、蓝色 outer、红色模型。"""
    overlay = image_rgb.copy()
    _draw_quad(overlay, truth.get("content_quad"), (35, 220, 75), "C")
    _draw_quad(overlay, truth.get("outer_quad"), (55, 145, 245), "O")
    _draw_quad(overlay, decision.get("corners"), (240, 65, 55), "P")
    preview = Image.fromarray(overlay)
    preview.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height + 42), "white")
    tile.paste(preview, ((tile_width - preview.width) // 2, (tile_height - preview.height) // 2))
    ImageDraw.Draw(tile).text(
        (8, tile_height + 10),
        f"view-{truth['capture_number']} | {truth['capture_view']} | {decision['status']}",
        fill="black",
    )
    return tile


def review(args: argparse.Namespace) -> int:
    """输出逐对象视觉复核板，不修改人工标注或冻结预测。"""
    data_root = args.data_root.expanduser().resolve()
    index, predictions, annotations = _load_bound_inputs(
        args.index.expanduser().resolve(),
        args.predictions.expanduser().resolve(),
        args.annotations.expanduser().resolve(),
        require_approved=True,
    )
    output = _ensure_output_directory(args.output_directory)
    frames_by_id = {row["image_id"]: row for row in index["frames"]}
    predictions_by_id = predictions
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for truth in annotations:
        grouped[truth["group_id"]].append(truth)
    reports: list[dict[str, Any]] = []
    for object_number, (group_id, truths) in enumerate(grouped.items(), start=1):
        truths.sort(key=lambda row: row["capture_number"])
        sheet = Image.new("RGB", (3 * 480, 2 * 682), "white")
        for position, truth in enumerate(truths):
            frame = frames_by_id[truth["image_id"]]
            path = _ensure_data_path(data_root / frame["image"], data_root, must_exist=True)
            _verify_frame(path, frame)
            image_rgb = load_image(path).original_rgb
            tile = _review_tile(image_rgb, truth, predictions_by_id[truth["image_id"]]["decision"])
            sheet.paste(tile, ((position % 3) * 480, (position // 3) * 682))
        destination = output / f"object-{object_number:02d}-review.jpg"
        sheet.save(destination, quality=92, subsampling=0)
        reports.append(
            {
                "group_id": group_id,
                "domain": truths[0]["domain"],
                "review_board": destination.name,
                "capture_count": len(truths),
            }
        )
        print(f"[视觉复核 {object_number}/{len(grouped)}] {destination.name}", file=sys.stderr)
    _write_json(
        output / "review-report.json",
        {
            "kind": "screenrestore_private_annotation_visual_review",
            "status": "READY_FOR_VISUAL_REVIEW",
            "dataset_sha256": index["dataset_sha256"],
            "legend": {
                "green": "human_content_quad",
                "blue": "human_outer_quad",
                "red": "frozen_model_candidate",
            },
            "annotation_mutated": False,
            "objects": reports,
        },
    )
    print(output / "review-report.json")
    return 0


def _render_restoration_contact_sheet(
    output: Path, reports: list[dict[str, Any]], *, columns: int = 4
) -> Path | None:
    """把每个对象的人工几何结果汇总成可快速人工复核的联系表。"""
    available = [row for row in reports if row["manual_geometry_output"] is not None]
    if not available:
        return None
    tile_width, tile_height, label_height = 480, 360, 36
    rows = (len(available) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    for position, row in enumerate(available):
        source = output / row["manual_geometry_output"]
        with Image.open(source) as opened:
            preview = opened.convert("RGB")
            preview.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (position % columns) * tile_width + (tile_width - preview.width) // 2
        y0 = (position // columns) * (tile_height + label_height)
        y = y0 + (tile_height - preview.height) // 2
        sheet.paste(preview, (x, y))
        # PIL 默认字体只保证 ASCII，因此标签使用稳定的对象编号和域名。
        ImageDraw.Draw(sheet).text(
            ((position % columns) * tile_width + 8, y0 + tile_height + 8),
            f"object-{position + 1:02d} | {row['domain']}",
            fill="black",
        )
    destination = output / "manual-geometry-contact-sheet.jpg"
    sheet.save(destination, quality=92, subsampling=0)
    return destination


def restore(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    index, predictions, annotations = _load_bound_inputs(
        args.index.expanduser().resolve(),
        args.predictions.expanduser().resolve(),
        args.annotations.expanduser().resolve(),
        require_approved=False,
    )
    output = _ensure_output_directory(args.output_directory)
    annotations_by_id = {row["image_id"]: row for row in annotations}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in index["frames"]:
        grouped[frame["group_id"]].append(frame)
    reports = []
    for position, (group_id, frames) in enumerate(grouped.items(), start=1):
        frames.sort(key=lambda row: row["capture_number"])
        representative = frames[0]
        truth = annotations_by_id[representative["image_id"]]
        object_directory = output / f"object-{position:02d}"
        object_directory.mkdir()
        path = _ensure_data_path(data_root / representative["image"], data_root, must_exist=True)
        image_rgb = load_image(path).original_rgb
        manual_output = None
        if (
            truth.get("annotation_status") == "approved"
            and truth["in_scope"]
            and truth.get("content_quad") is not None
        ):
            _verify_frame(path, representative)
            manual_pipeline = _configure_pipeline(truth["domain"], truth["content_quad"])
            manual_geometry = manual_pipeline.process(
                image_rgb,
                ProcessingContext(preview=False),
                source_id=f"private-manual-geometry:{truth['image_sha256']}",
            )
            manual_path = object_directory / "manual-geometry-archive.png"
            _save_rgb(manual_path, manual_geometry)
            provenance = ProvenanceReport(
                ArchiveVariant.ARCHIVE,
                ProvenanceMap.observed(manual_geometry.shape),
                geometry={"source": "human_annotation", "image_id": truth["image_id"]},
                notes=("单帧几何与受约束摄影校正；不声称恢复已饱和或被遮挡的内容。",),
            )
            _write_json(object_directory / "manual-geometry-provenance.json", provenance.to_dict())
            manual_output = str(manual_path.relative_to(output))
        accepted = [
            frame for frame in frames if predictions[frame["image_id"]]["decision"]["accepted"]
        ]
        # 自动代表帧只依据冻结预测置信度选择，不读取几何真值，避免 cherry-pick。
        automatic_frame = max(
            accepted,
            key=lambda frame: predictions[frame["image_id"]]["decision"]["confidence"],
            default=None,
        )
        automatic_output = None
        if automatic_frame is not None:
            decision = predictions[automatic_frame["image_id"]]["decision"]
            automatic_path = _ensure_data_path(
                data_root / automatic_frame["image"], data_root, must_exist=True
            )
            _verify_frame(automatic_path, automatic_frame)
            automatic_rgb = load_image(automatic_path).original_rgb
            automatic_pipeline = _configure_pipeline(automatic_frame["domain"], decision["corners"])
            restored = automatic_pipeline.process(
                automatic_rgb,
                ProcessingContext(preview=False),
                source_id=f"private-auto:{automatic_frame['image_sha256']}",
            )
            destination = object_directory / "e2e-auto-archive.png"
            _save_rgb(destination, restored)
            automatic_provenance = ProvenanceReport(
                ArchiveVariant.ARCHIVE,
                ProvenanceMap.observed(restored.shape),
                geometry={
                    "source": "frozen_automatic_prediction",
                    "image_id": automatic_frame["image_id"],
                    "confidence": decision["confidence"],
                },
                notes=("自动定位通过正式拒绝策略后执行单帧 Archive 流水线。",),
            )
            _write_json(
                object_directory / "e2e-auto-provenance.json", automatic_provenance.to_dict()
            )
            automatic_output = str(destination.relative_to(output))
        reports.append(
            {
                "group_id": group_id,
                "domain": representative["domain"],
                "manual_geometry_representative_capture": 1,
                "manual_geometry_output": manual_output,
                "e2e_auto_capture": automatic_frame["capture_number"] if automatic_frame else None,
                "e2e_auto_output": automatic_output,
                "e2e_auto_status": "restored" if automatic_frame else "rejected",
            }
        )
        print(
            f"[完整恢复 {position}/{len(grouped)}] {reports[-1]['e2e_auto_status']}",
            file=sys.stderr,
        )
    contact_sheet = _render_restoration_contact_sheet(output, reports)
    _write_json(
        output / "restoration-report.json",
        {
            "kind": "screenrestore_private_development_restoration",
            "status": (
                "COMPLETE"
                if all(row.get("annotation_status") == "approved" for row in annotations)
                else "PARTIAL: annotations_pending"
            ),
            "dataset_sha256": index["dataset_sha256"],
            "object_count": len(reports),
            "e2e_auto_restored": sum(row["e2e_auto_status"] == "restored" for row in reports),
            "manual_geometry_outputs": sum(
                row["manual_geometry_output"] is not None for row in reports
            ),
            "restoration_stack": {
                "geometry": "human_annotation_or_frozen_automatic_prediction",
                "pipeline": "product_runtime_archive_classic",
                "learned_restoration_models": [],
                "note": (
                    "当前完整训练的 P3 Fidelity/Photometric 只有训练 checkpoint，"
                    "尚未作为通过部署验证的产品运行时模型接入本次输出。"
                ),
            },
            "contact_sheet": (
                str(contact_sheet.relative_to(output)) if contact_sheet is not None else None
            ),
            "objects": reports,
        },
    )
    print(output / "restoration-report.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="建立不可变索引和待标注清单")
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--image-directory", type=Path, required=True)
    prepare_parser.add_argument("--output-directory", type=Path, required=True)
    prepare_parser.add_argument("--annotations", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze", help="在读取真值前冻结当前模型预测")
    freeze_parser.add_argument("--data-root", type=Path, required=True)
    freeze_parser.add_argument("--index", type=Path, required=True)
    freeze_parser.add_argument("--quad-model", type=Path, required=True)
    freeze_parser.add_argument(
        "--calibrator",
        type=Path,
        help="仅使用公开 validation 冻结的 correctness calibrator",
    )
    freeze_parser.add_argument(
        "--classic-agreement-snap",
        action="store_true",
        help="实验性：运行公开 validation 冻结的模型/传统候选双分支几何门",
    )
    freeze_parser.add_argument("--classic-agreement-min-iou", type=float, default=0.70)
    freeze_parser.add_argument("--classic-agreement-max-corner-nce", type=float, default=0.12)
    freeze_parser.add_argument("--output-directory", type=Path, required=True)

    label_parser = subparsers.add_parser("label", help="逐帧人工标注 content/outer 四角")
    label_parser.add_argument("--data-root", type=Path, required=True)
    label_parser.add_argument("--index", type=Path, required=True)
    label_parser.add_argument("--annotations", type=Path, required=True)
    label_parser.add_argument(
        "--relabel-approved",
        action="store_true",
        help="重新打开已批准帧；默认断点续标时跳过",
    )

    score_parser = subparsers.add_parser("score", help="用已冻结预测进行 e2e_auto 评分")
    score_parser.add_argument("--index", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--annotations", type=Path, required=True)
    score_parser.add_argument("--output-directory", type=Path, required=True)

    review_parser = subparsers.add_parser("review", help="生成人工与模型四角的逐对象视觉复核板")
    review_parser.add_argument("--data-root", type=Path, required=True)
    review_parser.add_argument("--index", type=Path, required=True)
    review_parser.add_argument("--predictions", type=Path, required=True)
    review_parser.add_argument("--annotations", type=Path, required=True)
    review_parser.add_argument("--output-directory", type=Path, required=True)

    restore_parser = subparsers.add_parser(
        "restore", help="每个对象输出一次人工几何恢复与允许的自动恢复"
    )
    restore_parser.add_argument("--data-root", type=Path, required=True)
    restore_parser.add_argument("--index", type=Path, required=True)
    restore_parser.add_argument("--predictions", type=Path, required=True)
    restore_parser.add_argument("--annotations", type=Path, required=True)
    restore_parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "prepare": prepare,
        "freeze": freeze,
        "label": label,
        "score": score,
        "review": review,
        "restore": restore,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
