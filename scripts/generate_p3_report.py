"""汇总 P3 run 的机器报告，并刷新正式结果文档中的实际状态。

使用示例：
    source .venv/bin/activate
    which python
    python scripts/generate_p3_report.py \
      --run-directory "$SCREENRESTORE_RUN_ROOT/$SCREENRESTORE_RUN_NAME" \
      --docs-directory docs

该脚本只汇总已存在的 JSON 和产物；缺失阶段写 PENDING/BLOCKED，不填推测指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--docs-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_directory.expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"run 目录不存在：{run}")
    report_directory = run / "report"
    if report_directory.exists():
        raise FileExistsError(f"拒绝覆盖已有 report：{report_directory}")
    report_directory.mkdir(parents=True)
    documents = sorted(run.rglob("*.json"))
    index: list[dict[str, Any]] = []
    for path in documents:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        index.append(
            {
                "path": str(path.relative_to(run)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": value.get("status") if isinstance(value, dict) else None,
                "kind": value.get("kind") if isinstance(value, dict) else None,
            }
        )
    stages = _stage_status(run)
    training_runs = _training_runs(run)
    model_artifacts = _model_artifacts(run)
    blockers = _blockers(run)
    evaluations = _evaluations(run)
    summary = {
        "format_version": 1,
        "kind": "p3_report_index",
        "run_directory": str(run),
        "stages": stages,
        "json_artifacts": index,
        "release_gate": _read_optional(run / "evaluate" / "release-gate.json"),
        "preflight": _read_optional(run / "preflight" / "preflight.json"),
        "training_runs": training_runs,
        "model_artifacts": model_artifacts,
        "blockers": blockers,
        "evaluations": evaluations,
    }
    (report_directory / "run-index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = _markdown(summary)
    (report_directory / "P3_RESULTS.md").write_text(markdown, encoding="utf-8")
    docs = args.docs_directory.expanduser().resolve()
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "P3_RESULTS.md").write_text(markdown, encoding="utf-8")
    (docs / "P3_GEOMETRY_RESULTS.md").write_text(
        _section_markdown(
            "P3 Geometry",
            stages,
            prefix="geometry",
            summary=summary,
        ),
        encoding="utf-8",
    )
    (docs / "P3_RESTORATION_RESULTS.md").write_text(
        _section_markdown(
            "P3 Restoration",
            stages,
            names=("fidelity", "photometric", "demoire", "reflection", "superres", "router"),
            summary=summary,
        ),
        encoding="utf-8",
    )
    print(report_directory / "P3_RESULTS.md")
    return 0


def _stage_status(run: Path) -> dict[str, str]:
    names = (
        "preflight",
        "smoke-cpu",
        "smoke-mps",
        "geometry-b0",
        "geometry-b1",
        "geometry-b2",
        "geometry-b3",
        "geometry-b4",
        "geometry-b5",
        "geometry-b6",
        "dewarp",
        "fidelity",
        "photometric",
        "demoire",
        "demoire-synthetic",
        "reflection",
        "reflection-synthetic",
        "superres",
        "router",
        "evaluate",
    )
    output: dict[str, str] = {}
    for name in names:
        path = run / name
        blocked = run / f"{name}-real-blocked.json"
        if blocked.is_file():
            output[name] = "BLOCKED"
        elif path.is_dir() and any(path.iterdir()):
            output[name] = "COMPLETED"
        else:
            output[name] = "PENDING"
    if (run / "demoire-real-blocked.json").is_file():
        output["demoire-real"] = "BLOCKED"
    if (run / "reflection-real-blocked.json").is_file():
        output["reflection-real"] = "BLOCKED"
    return output


def _read_optional(path: Path) -> object | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _training_runs(run: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(run.rglob("run.json")):
        value = _read_optional(path)
        if not isinstance(value, dict):
            continue
        output.append(
            {
                "stage": str(path.parent.relative_to(run)),
                "kind": value.get("kind"),
                "task": value.get("task"),
                "budget": value.get("budget"),
                "device": value.get("device"),
                "parameter_count": value.get("parameter_count"),
                "samples": value.get("samples"),
                "steps": value.get("steps"),
                "wall_time_seconds": value.get("wall_time_seconds"),
                "peak_memory_bytes": value.get("peak_memory_bytes"),
                "seed": value.get("seed"),
                "git_commit": value.get("git_commit"),
                "manifest_sha256": value.get("manifest_sha256"),
            }
        )
    return output


def _model_artifacts(run: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pattern in ("best.pt", "*.onnx", "correctness-calibrator.json"):
        for path in sorted(run.rglob(pattern)):
            output.append(
                {
                    "path": str(path.relative_to(run)),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return output


def _blockers(run: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(run.glob("*-blocked.json")):
        value = _read_optional(path)
        if isinstance(value, dict):
            output.append({"path": path.name, **value})
    return output


def _evaluations(run: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(run.rglob("evaluation.json")):
        value = _read_optional(path)
        if not isinstance(value, dict):
            continue
        output.append(
            {
                "stage": str(path.parent.relative_to(run)),
                "kind": value.get("kind", value.get("protocol")),
                "summary": value.get("summary", value.get("metrics")),
            }
        )
    return output


def _markdown(summary: dict[str, Any]) -> str:
    stages = summary["stages"]
    lines = [
        "# P3 结果",
        "",
        f"run：`{summary['run_directory']}`",
        "",
        "本文件只汇总实际存在的产物。未执行的正式训练保持 PENDING，外部数据缺失保持 BLOCKED。",
        "",
        "## 阶段状态",
        "",
        "| stage | status |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in stages.items())
    lines.extend(
        [
            "",
            "## 训练运行",
            "",
            "| stage | task | budget | device | parameters | wall seconds |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in summary["training_runs"]:
        lines.append(
            f"| {item['stage']} | {item.get('task')} | {item.get('budget')} | "
            f"{item.get('device')} | {item.get('parameter_count')} | {item.get('wall_time_seconds')} |"
        )
    lines.extend(
        [
            "",
            "## 模型产物",
            "",
            "| path | bytes | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    lines.extend(
        f"| {item['path']} | {item['size_bytes']} | `{item['sha256']}` |"
        for item in summary["model_artifacts"]
    )
    lines.extend(
        [
            "",
            "## 数据与阻塞",
            "",
            "```json",
            json.dumps(
                {"preflight": summary.get("preflight"), "blockers": summary.get("blockers")},
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## 评估摘要",
            "",
            "```json",
            json.dumps(summary.get("evaluations"), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Release gate",
            "",
            "```json",
            json.dumps(summary.get("release_gate"), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 结论",
            "",
            "release gate 以机器报告为准；FAIL 保持 FAIL，不调整阈值。",
            "",
        ]
    )
    return "\n".join(lines)


def _section_markdown(
    title: str,
    stages: dict[str, str],
    *,
    prefix: str | None = None,
    names: tuple[str, ...] = (),
    summary: dict[str, Any] | None = None,
) -> str:
    selected = {
        name: status
        for name, status in stages.items()
        if (prefix is not None and name.startswith(prefix)) or name in names
    }
    lines = [f"# {title}", "", "| stage | status |", "|---|---|"]
    lines.extend(f"| {name} | {status} |" for name, status in selected.items())
    lines.extend(
        [
            "",
            "## 实际评估与训练元数据",
            "",
            "```json",
            json.dumps(
                {
                    "evaluations": (summary or {}).get("evaluations", []),
                    "training_runs": (summary or {}).get("training_runs", []),
                    "model_artifacts": (summary or {}).get("model_artifacts", []),
                    "blockers": (summary or {}).get("blockers", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "正式指标仅在对应 FULL/evaluate 产物存在后写入；缺失值不以推测数字补齐。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
