"""对冻结的 Fidelity checkpoint 做可复现 DIV2K HR 定量验证。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.restoration.evaluate \\
        --checkpoint "$SCREENRESTORE_RUN_ROOT/restoration/fidelity-smoke/best.pt" \\
        --hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \\
        --output "$SCREENRESTORE_RUN_ROOT/restoration/fidelity-smoke/evaluation.json"

评估只读取公开 HR 验证图片；在线退化种子固定，因此不同 checkpoint 可公平比较。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import Div2kHrDataset
from .degradation import CameraDegradationConfig
from .losses import fidelity_loss
from .metrics import fidelity_metrics
from .model import BoundedResidualNet
from .train import _accumulate, _batch_to_device, _device, _mean, _progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hr-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.samples < 0:
        raise ValueError("batch-size 必须大于 0，samples 不能为负数")
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "BoundedResidualNet":
        raise ValueError("checkpoint 不是 BoundedResidualNet")
    model = BoundedResidualNet(
        int(checkpoint["channels"]),
        int(checkpoint["blocks"]),
        float(checkpoint["max_delta"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = _device(args.device)
    model.to(device).eval()
    dataset = Div2kHrDataset(
        args.hr_directory,
        patch_size=int(checkpoint["patch_size"]),
        degradation=CameraDegradationConfig(**checkpoint["degradation"]),
        seed=args.seed,
        max_samples=args.samples,
    )
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    totals: dict[str, float] = {}
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            _progress(index - 1, len(loader), "冻结 checkpoint 验证")
            degraded, target = _batch_to_device(batch, device)
            restored = model(degraded)
            clean_restored = model(target)
            loss, parts = fidelity_loss(
                restored,
                target,
                clean_restored,
                target,
                identity_weight=float(checkpoint["identity_weight"]),
                edge_weight=float(checkpoint["edge_weight"]),
            )
            parts["loss"] = float(loss)
            _accumulate(totals, parts)
            _accumulate(totals, fidelity_metrics(restored, target, clean_restored))
    _progress(len(loader), len(loader), "冻结 checkpoint 验证")
    result = {
        "format_version": 1,
        "kind": "fidelity_restoration_evaluation",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "hr_directory": str(args.hr_directory.expanduser().resolve()),
        "samples": len(dataset),
        "device": str(device),
        "metrics": _mean(totals, len(loader)),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
