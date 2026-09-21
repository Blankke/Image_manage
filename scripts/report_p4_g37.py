"""从阶段聚合 JSON 生成脱敏 P4-G3.7 报告，不读取照片或 test。

用法：bash scripts/run_p4_g37.sh report
报告只消费阶段 result.json，输出到 report 阶段目录；复制入 docs 前人工审阅。
"""

from __future__ import annotations

from pathlib import Path

from scripts.audit_p4_dataset_isolation import markdown


def render_reports(stages: dict, output: Path) -> None:
    lines = [
        "# P4-G3.7 结果",
        "",
        "nce_normalization = target_quad_bbox_diagonal；metric_version = 2。",
        "",
        "| 阶段 | 状态 |",
        "|---|---|",
    ]
    for name, result in stages.items():
        lines.append(f"| {name} | {result.get('status', 'UNKNOWN')} |")
    baseline = stages.get("baseline", {})
    if baseline.get("datasets"):
        lines += [
            "",
            "## B0 metric-v2 baseline",
            "",
            "| slice | candidates | NCE median | NCE P95 | IoU median | IoU P05 | strict | accepted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, result in baseline["datasets"].items():
            m = result["geometry"]["coarse"]
            lines.append(
                f"| {name} | {m['candidate_count']} | {m['corner_nce_median']:.8f} | "
                f"{m['corner_nce_p95']:.8f} | {m['quad_iou_median']:.8f} | "
                f"{m['quad_iou_p05']:.8f} | {m['strict_geometry_rate']:.8f} | "
                f"{result['hard_gate']['accepted_count']} |"
            )
        lines += [
            "",
            "strict 为正例候选几何正确率；accepted 为完整 slice 的当前 hard gate 接受数。",
            f"Parity passed: {baseline.get('parity', {}).get('passed', False)}。",
        ]
    lines += [
        "",
        "## 决策与边界",
        "",
        "没有完整独立 acceptance release gate 时，保持 runtime fail-closed。",
        f"Target readiness: {stages.get('prepare-target', {}).get('status', 'NOT_RUN')}。",
        f"Acceptance: {stages.get('acceptance-dev', {}).get('status', 'NOT_RUN')}。",
        f"Native-512: {stages.get('sanity512', {}).get('status', 'NOT_RUN')}；"
        f"{stages.get('sanity512', {}).get('reason', '')}。",
        "Fidelity、Photometric、Super-resolution、Router 保持冻结。真实 demoire/reflection 保持 BLOCKED。",
        "本阶段不访问 SmartDoc test、不启动长训、不下载外部数据。",
        "",
    ]
    (output / "P4_G37_RESULTS.md").write_text("\n".join(lines))
    if "audit-data" in stages:
        (output / "P4_DATA_ISOLATION_AUDIT.md").write_text(markdown(stages["audit-data"]))
