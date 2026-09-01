"""从 validation 几何评估产物提取严格正确率校准特征。

使用范例：
    source .venv/bin/activate
    which python
    python scripts/prepare_geometry_calibration.py \
      --evaluation runs/calibration/evaluation.json \
      --output runs/calibration/features.jsonl --split validation

输入必须声明所有预测先于 oracle 加载。本脚本拒绝 test split，只输出运行时可得数值特征、
``strict_correct`` 标签和样本标识，不复制图片或清单中的其它 GT。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from screenrestore.geometry import CORRECTNESS_FEATURE_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args(argv)
    evaluation = json.loads(args.evaluation.expanduser().resolve().read_text(encoding="utf-8"))
    if evaluation.get("oracle_loaded_after_all_predictions") is not True:
        raise ValueError("校准输入必须证明所有预测在 oracle 加载前冻结")
    cases = evaluation.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation.json 缺少非空 cases")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖校准特征：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            if not isinstance(case, dict):
                raise ValueError("evaluation case 必须是对象")
            decision = case.get("decision")
            metrics = case.get("metrics")
            if not isinstance(decision, dict) or not isinstance(metrics, dict):
                raise ValueError("evaluation case 缺少 decision/metrics")
            diagnostics = decision.get("diagnostics")
            if not isinstance(diagnostics, dict):
                _progress(index, len(cases))
                continue
            missing = [name for name in CORRECTNESS_FEATURE_NAMES if name not in diagnostics]
            if missing:
                # 无候选/非法 quad 会在进入校准器前直接拒绝，不具备完整 policy 特征。
                _progress(index, len(cases))
                continue
            record = {
                "id": str(case.get("case", index)),
                "split": args.split,
                "strict_correct": bool(metrics.get("strict_correct", False)),
                "features": {
                    name: float(diagnostics[name]) for name in CORRECTNESS_FEATURE_NAMES
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            _progress(index, len(cases))
    if written == 0:
        output.unlink(missing_ok=True)
        raise ValueError("evaluation 中没有可用于 policy 校准的完整候选")
    return 0


def _progress(done: int, total: int) -> None:
    width = 24
    filled = round(width * done / max(1, total))
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} 提取校准特征",
        end="\n" if done >= total else "\r",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
