#!/usr/bin/env python3
"""P4-G3.7 阶段编排，无 all/test，无下载，输出拒绝覆盖。

用法：source .venv/bin/activate && which python
python scripts/run_p4_g37.py preflight --data-root /data --run-root /runs --output-root /runs/p4-g37
各阶段 provenance.json 保存命令、代码、模型和清单身份；失败阶段不可当作成功输入。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_p4_acceptance as acceptance  # noqa: E402
from scripts import evaluate_p4_geometry_trajectory as trajectory  # noqa: E402
from scripts.audit_p4_dataset_isolation import (  # noqa: E402
    audit,
    markdown,
    read_development,
    sha256,
)
from scripts.prepare_p4_target_domain import render_overlays, validate_target  # noqa: E402
from training.quadlocator.train import _assert_public_training_manifest  # noqa: E402

B0_SHA = "3344ad62709abf9e413a1cdebbdb82b1c0add0655ffdaeadcd9ddcc6bac86746"
STAGES = (
    "preflight",
    "audit-data",
    "baseline",
    "sanity512",
    "prepare-target",
    "acceptance-dev",
    "report",
)
METRIC = {"metric_version": 2, "nce_normalization": "target_quad_bbox_diagonal"}


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def comparable_config(metadata: dict) -> dict:
    """拒绝把 loss/增强变化伪装为分辨率单变量；不自动启动补充对照。"""
    required = {
        "image_size": 256,
        "learning_rate": 1e-6,
        "loss_profile": "content_coordinate_only",
        "trainable_scope": "content_head",
        "train_augmentation": "full",
        "seed": 20260902,
        "batch_size": 16,
        "hard_sampling": False,
        "validation_split": "validation",
    }
    mismatch = [k for k, v in required.items() if metadata.get(k) != v]
    for k in ("train_samples", "validation_samples", "scheduler_t_max"):
        if not isinstance(metadata.get(k), int) or metadata[k] < 1:
            mismatch.append(k)
    if mismatch:
        raise ValueError("missing_comparable_256_arm: " + ",".join(mismatch))
    return {
        **required,
        "image_size": 512,
        "epochs": 1,
        **{k: metadata[k] for k in ("train_samples", "validation_samples", "scheduler_t_max")},
    }


def resolve_incumbent(output_root: Path, b0: Path) -> Path:
    """只有已通过 trajectory 的冻结身份能替换 B0；拒绝半成品或失配文件。"""
    result_path = output_root / "sanity512/result.json"
    if not result_path.is_file():
        return b0
    result = json.loads(result_path.read_text())
    if result["status"] in ("BLOCKED", "NO_ELIGIBLE_CHECKPOINT"):
        return b0
    if result["status"] != "FROZEN":
        raise ValueError("unknown geometry freeze state")
    selection = output_root / "sanity512/selection"
    summary = json.loads((selection / "trajectory-summary.json").read_text())
    frozen = selection / "frozen-geometry.pt"
    if (
        summary["selection"]["status"] != "FROZEN"
        or sha256(frozen) != summary["frozen_checkpoint"]["sha256"]
    ):
        raise ValueError("frozen challenger identity mismatch")
    return frozen


def baseline(args: argparse.Namespace, output: Path, checkpoint: Path) -> dict:
    """三个开发切片共同 512 重算；照片推理完成后才物化对应 GT。"""
    import torch

    specs = (
        trajectory._dataset_spec(
            "internal_validation",
            args.data_root / "manifests/p2-public/stage-b.geometry.jsonl",
            args.data_root,
            max_samples=1000,
            seed=20260902,
        ),
        trajectory._dataset_spec(
            "calibration",
            args.data_root / "manifests/p2-public/calibration-public.geometry.jsonl",
            args.data_root,
        ),
        trajectory._dataset_spec(
            "smartdoc_validation",
            args.data_root / "manifests/smartdoc.geometry.jsonl",
            args.data_root,
        ),
    )
    paths = tuple(sorted({p for spec in specs for p in spec.paths}))
    predictions = acceptance._infer_acceptance_features(
        checkpoint, paths, device=torch.device(args.device), batch_size=8
    )
    reports = {}
    for spec in specs:
        # 只读取 validation GT，再按冻结 schedule 精确匹配。
        truths = acceptance._truth_by_path(spec.manifest, args.data_root, "validation")
        rows = [acceptance._score_case(predictions[p], truths[p]) for p in spec.paths]
        reports[spec.name] = {
            "sample_count": len(rows),
            "independent_group_count": len({r["group_id"] for r in rows}),
            "geometry": acceptance._geometry_report(rows),
            "boundary": acceptance._boundary_report(rows),
            "hard_gate": acceptance._policy_metrics(rows, "hard_accepted"),
            "rejection_reasons": dict(
                Counter(reason for r in rows for reason in r["hard_reasons"])
            ),
        }
        write(output / f"{spec.name}.json", {**METRIC, **reports[spec.name]})
    parity = evaluator_parity(args, checkpoint, specs)
    return {
        **METRIC,
        "status": "PASS" if parity["passed"] else "BLOCKED: evaluator_parity_failed",
        "checkpoint_sha256": sha256(checkpoint),
        "evaluation_image_size": 512,
        "datasets": reports,
        "parity": parity,
        "test_data_used": False,
    }


def evaluator_parity(args: argparse.Namespace, checkpoint: Path, specs: tuple) -> dict:
    """固定每 slice 32 个样本检查 raw/decoded parity 与两条 evaluator；不拟合任何参数。"""
    import onnxruntime as ort
    import torch
    from scripts.audit_p4_geometry_parity import _prediction_from_raw
    from training.quadlocator.dataset import QuadDataset
    from training.quadlocator.metrics import ValidationMetrics
    from training.quadlocator.model import QuadLocatorS, load_quadlocator_state_dict

    from screenrestore.geometry.decoder import decode_corner_logits
    from screenrestore.geometry.detector import _letterbox_tensor
    from screenrestore.io.image_loader import load_image

    ort.disable_telemetry_events()
    session = ort.InferenceSession(
        str(checkpoint.with_name("quadlocator-s.onnx")), providers=["CPUExecutionProvider"]
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = QuadLocatorS(float(payload["width_multiplier"])).to(args.device).eval()
    load_quadlocator_state_dict(model, payload["state_dict"])
    raw_max, coordinate_max, reports = 0.0, 0.0, {}
    for spec in specs:
        # QuadDataset 只接收临时 validation-only 清单，避免原 loader 物化 test。
        records = read_development(spec.manifest, "validation")
        allowed = {str(p.resolve()) for p in spec.paths}
        records = [r for r in records if str((args.data_root / r["image"]).resolve()) in allowed]
        indices = sorted(
            np.random.default_rng(20260902).choice(
                len(records), min(32, len(records)), replace=False
            )
        )
        records = [records[int(i)] for i in indices]
        safe_manifest = args.output_root / "baseline" / f"{spec.name}.parity-input.jsonl"
        with safe_manifest.open("x") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")
        dataset = QuadDataset(
            safe_manifest,
            split="validation",
            image_size=512,
            dataset_root=args.data_root,
            augment=False,
        )
        training_metrics, frozen_nce, frozen_iou = ValidationMetrics(), [], []
        for index, record in enumerate(records):
            batch = {k: v.unsqueeze(0).to(args.device) for k, v in dataset[index].items()}
            with torch.inference_mode():
                outputs = model(batch["image"])
            training_metrics.update(outputs, batch)
            rgb = load_image(args.data_root / record["image"]).original_rgb
            tensor, transform = _letterbox_tensor(rgb, 512)
            # 输入一致性必须验证，包含 EXIF、缩放与颜色空间。
            if not np.array_equal(tensor, batch["image"].cpu().numpy()):
                return {"passed": False, "reason": "training_runtime_preprocess_mismatch"}
            onnx = dict(
                zip(
                    trajectory.OUTPUT_NAMES,
                    session.run(
                        list(trajectory.OUTPUT_NAMES), {session.get_inputs()[0].name: tensor}
                    ),
                    strict=True,
                )
            )
            raw = {k: v.detach().cpu().numpy() for k, v in outputs.items()}
            raw_max = max(raw_max, *(float(np.max(np.abs(raw[k] - onnx[k]))) for k in raw))
            a = decode_corner_logits(raw["content_corner_heatmaps"]).coordinates
            b = decode_corner_logits(onnx["content_corner_heatmaps"]).coordinates
            if (a is None) != (b is None):
                return {"passed": False, "reason": "torch_onnx_candidate_mismatch"}
            if a is not None:
                coordinate_max = max(coordinate_max, float(np.max(np.abs(a - b))) / 127)
            prediction = _prediction_from_raw(raw, transform, rgb.shape, "decoder_v2")
            if record["present"]:
                score = acceptance._stage_metrics(
                    prediction.content_quad,
                    np.asarray(record["content_quad"], np.float32),
                    np.array([rgb.shape[1] - 1, rgb.shape[0] - 1], np.float32),
                )
                frozen_nce.append(score["corner_nce"])
                frozen_iou.append(score["quad_iou"])
            print(f"[parity {spec.name} {index + 1}/{len(records)}]", file=sys.stderr)
        m = training_metrics.compute()
        deltas = {
            "nce_median": abs(m["content_corner_nce_median"] - float(np.median(frozen_nce))),
            "nce_p95": abs(m["content_corner_nce_p95"] - float(np.percentile(frozen_nce, 95))),
            "iou_median": abs(m["content_iou_median"] - float(np.median(frozen_iou))),
            "iou_p05": abs(m["content_iou_p05"] - float(np.percentile(frozen_iou, 5))),
        }
        reports[spec.name] = {
            "sample_count": len(records),
            "deltas": deltas,
            "passed": all(
                deltas[k] <= limit
                for k, limit in [
                    ("nce_median", 0.002),
                    ("nce_p95", 0.005),
                    ("iou_median", 0.01),
                    ("iou_p05", 0.01),
                ]
            ),
        }
    return {
        "passed": raw_max <= 0.001
        and coordinate_max <= 0.001
        and all(r["passed"] for r in reports.values()),
        "torch_onnx_max_abs": raw_max,
        "decoded_max_normalized_delta": coordinate_max,
        "datasets": reports,
        "onnx_sha256": sha256(checkpoint.with_name("quadlocator-s.onnx")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--test-identity-index", type=Path)
    parser.add_argument("--comparable-run", type=Path)
    args = parser.parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    if args.output_root.is_relative_to(repo) or args.output_root.is_relative_to(args.data_root):
        raise ValueError("run output must be outside repository and data root")
    checkpoint = args.run_root / "p2-geometry-w1-20260829-110658/stage-b/best.pt"
    output = args.output_root / args.stage
    manifests = [
        args.data_root / p
        for p in (
            "manifests/p2-public/stage-b.geometry.jsonl",
            "manifests/p2-public/calibration-public.geometry.jsonl",
            "manifests/smartdoc.geometry.jsonl",
        )
    ]
    # 先审计输入身份，再创建阶段产物或执行任何模型选择。
    for manifest in manifests:
        _assert_public_training_manifest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        **METRIC,
        "stage": args.stage,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": [sys.executable, *sys.argv],
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": {p.name: sha256(p) for p in manifests},
        "independent_group_count": {
            p.name: len({r["group_id"] for r in read_development(p)}) for p in manifests
        },
        "test_data_used": False,
        "protocol_sha256": sha256(repo / "docs/P4_G37_PLAN.md"),
        "code_sha256": {
            str(p.relative_to(repo)): sha256(p)
            for base in ("scripts", "training", "src")
            for p in sorted((repo / base).rglob("*.py"))
        },
    }
    write(output / "provenance.json", provenance)
    try:
        if provenance["checkpoint_sha256"] != B0_SHA:
            raise ValueError("B0 checkpoint identity mismatch")
        kib = int(subprocess.check_output(["du", "-sk", str(args.data_root)], text=True).split()[0])
        if kib > 30 * 1024 * 1024:
            raise ValueError("data root exceeds 30 GiB")
        target = (
            args.target_manifest
            or args.data_root / "manifests/p4-public-target/annotations.geometry.jsonl"
        )
        if args.stage == "preflight":
            result = {
                "status": "PASS",
                "data_kib": kib,
                "target_manifest_exists": target.is_file(),
                "sanity_status": "BLOCKED: missing_comparable_256_arm",
                "acceptance_status": "BLOCKED: insufficient_target_domain_groups",
            }
        elif args.stage == "audit-data":
            result = audit(args.data_root, target, args.test_identity_index)
            write(output / "dataset-isolation.json", result)
            (output / "dataset-isolation.md").write_text(markdown(result))
            result["status"] = "COMPLETED_WITH_LIMITATIONS"
        elif args.stage == "baseline":
            result = baseline(args, output, checkpoint)
        elif args.stage in ("prepare-target", "acceptance-dev"):
            if args.stage == "acceptance-dev" and target.is_file():
                _assert_public_training_manifest(target)
            records = read_development(target) if target.is_file() else []
            isolation = audit(args.data_root, target, args.test_identity_index) if records else None
            result = validate_target(records, args.data_root, isolation)
            if args.stage == "prepare-target" and records:
                render_overlays(records, args.data_root, output / "overlays")
            if args.stage == "acceptance-dev" and result["status"] == "READY":
                from scripts.p4_acceptance_development import evaluate_development

                checkpoint = resolve_incumbent(args.output_root, checkpoint)
                provenance["checkpoint_sha256"] = sha256(checkpoint)
                provenance["manifest_sha256"]["target_development"] = sha256(target)
                write(output / "provenance.json", provenance)
                result = evaluate_development(records, args, checkpoint, output)
        elif args.stage == "sanity512":
            prior = json.loads((args.output_root / "baseline/result.json").read_text())
            if prior["status"] != "PASS" or prior["checkpoint_sha256"] != B0_SHA:
                raise ValueError("evaluator_parity_failed_or_stale_baseline")
            baseline_provenance = json.loads(
                (args.output_root / "baseline/provenance.json").read_text()
            )
            if baseline_provenance["manifest_sha256"] != provenance["manifest_sha256"]:
                raise ValueError("baseline_dataset_changed")
            if args.comparable_run is None:
                raise ValueError("missing_comparable_256_arm")
            config = comparable_config(json.loads((args.comparable_run / "run.json").read_text()))
            # 可比 arm 必须提供冻结数据身份，无法核实的历史日志不具备复现实验资格。
            control = json.loads((args.comparable_run / "provenance.json").read_text())
            if (
                control["checkpoint_sha256"] != B0_SHA
                or control["manifest_sha256"] != provenance["manifest_sha256"]
            ):
                raise ValueError("comparable_arm_identity_mismatch")
            from training.quadlocator.train import main as train

            safe = output / "development.geometry.jsonl"
            records = read_development(manifests[0])
            with safe.open("x") as handle:
                for r in records:
                    handle.write(json.dumps(r) + "\n")
            command = [
                "--manifest",
                str(safe),
                "--dataset-root",
                str(args.data_root),
                "--output-directory",
                str(output / "training"),
                "--init-checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--evaluate-init",
                "--checkpoint-epochs",
                "1",
            ]
            for key, value in config.items():
                if key == "hard_sampling":
                    continue
                command.extend(["--" + key.replace("_", "-"), str(value)])
            write(output / "training-command.json", command)
            train(command)
            status = trajectory.main(
                [
                    "--baseline",
                    str(checkpoint),
                    "--checkpoints",
                    str(output / "training/checkpoints/epoch-001.pt"),
                    "--internal-manifest",
                    str(manifests[0]),
                    "--calibration-manifest",
                    str(manifests[1]),
                    "--smartdoc-manifest",
                    str(manifests[2]),
                    "--dataset-root",
                    str(args.data_root),
                    "--evaluation-image-size",
                    "512",
                    "--device",
                    args.device,
                    "--output-directory",
                    str(output / "selection"),
                ]
            )
            result = {
                "status": "FROZEN" if status == 0 else "NO_ELIGIBLE_CHECKPOINT",
                "incumbent": "challenger" if status == 0 else "B0",
            }
        else:
            result = {
                "status": "REPORT",
                "stages": {
                    stage: json.loads((args.output_root / stage / "result.json").read_text())
                    for stage in STAGES
                    if stage != "report" and (args.output_root / stage / "result.json").is_file()
                },
            }
            from scripts.report_p4_g37 import render_reports

            render_reports(result["stages"], output)
        result.update(METRIC)
        write(output / "result.json", result)
        if "independent_groups_by_domain" in result:
            provenance["target_independent_groups_by_domain"] = result[
                "independent_groups_by_domain"
            ]
        write(output / "provenance.json", provenance)
        print(json.dumps({"stage": args.stage, "status": result.get("status")}, ensure_ascii=False))
        return (
            3
            if str(result.get("status", "")).startswith(
                ("BLOCKED", "CALIBRATION_FAILURE", "NO_ELIGIBLE")
            )
            else 0
        )
    except Exception as exc:
        write(
            output / "result.json",
            {**METRIC, "status": "BLOCKED", "reason": str(exc), "incumbent": "B0"},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
