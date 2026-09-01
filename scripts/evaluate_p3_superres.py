"""比较 bicubic、P1 保守 SR 与可选 P3 SR checkpoint。

使用示例：
    source .venv/bin/activate
    which python
    python scripts/evaluate_p3_superres.py \
      --data-root "$SCREENRESTORE_DATA_ROOT" \
      --p1-run /Users/caozichen/screenrestore-runs/p1-full-20260828-133513 \
      --output /tmp/p3-sr-evaluation.json

该脚本只读取 validation 配对，默认最多 20 张固定裁剪；不会重训或覆盖权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# 允许按文档直接执行 ``python scripts/evaluate_p3_superres.py``，无需依赖 editable install。
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from training.superres.dataset import Div2kPairedSuperResolutionDataset  # noqa: E402
from training.superres.model import ConservativeSuperResolutionNet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--p1-run", type=Path, required=True)
    parser.add_argument("--p3-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args(argv)
    if args.samples < 1:
        raise ValueError("samples 必须大于 0")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 SR evaluation：{output}")
    root = args.data_root.expanduser().resolve()
    dataset = Div2kPairedSuperResolutionDataset(
        root / "manifests" / "div2k.restoration.jsonl",
        root,
        split="validation",
        variant="x2",
        patch_size=192,
        seed=20260830,
        max_samples=args.samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    models: dict[str, ConservativeSuperResolutionNet | None] = {
        "bicubic": None,
        "p1": _load_model(args.p1_run / "superres-x2" / "best.pt"),
        "p3": _load_model(args.p3_checkpoint) if args.p3_checkpoint else None,
    }
    totals = {
        name: {"psnr": 0.0, "ssim": 0.0, "overshoot": 0.0, "ringing": 0.0}
        for name, model in models.items()
        if name == "bicubic" or model is not None
    }
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            _progress(index - 1, len(loader))
            source, target = batch["input"], batch["target"]
            baseline = F.interpolate(source, size=target.shape[-2:], mode="bicubic", align_corners=False)
            for name, model in models.items():
                if name != "bicubic" and model is None:
                    continue
                restored = baseline if model is None else model(source)
                values = _metrics(restored, target, baseline)
                for key, value in values.items():
                    totals[name][key] += value
    _progress(len(loader), len(loader))
    count = len(loader)
    metrics = {
        name: {key: value / count for key, value in values.items()}
        for name, values in totals.items()
    }
    report = {
        "format_version": 1,
        "kind": "p3_superres_comparison",
        "variant": "x2",
        "samples": count,
        "metrics": metrics,
        "p3_status": "EVALUATED" if models["p3"] is not None else "NOT_AVAILABLE",
        "retrain_decision": "PENDING_OPERATOR_REVIEW",
        "lpips": "NOT_AVAILABLE: optional metric dependency is not installed by default",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


def _load_model(path: Path | None) -> ConservativeSuperResolutionNet:
    if path is None:
        raise ValueError("checkpoint 路径不能为空")
    resolved = path.expanduser().resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    model = ConservativeSuperResolutionNet(
        int(checkpoint["scale"]),
        int(checkpoint["channels"]),
        int(checkpoint["blocks"]),
        float(checkpoint["max_delta"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def _metrics(
    restored: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> dict[str, float]:
    mse = torch.mean((restored - target).square()).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    mean_x = restored.mean(dim=(2, 3), keepdim=True)
    mean_y = target.mean(dim=(2, 3), keepdim=True)
    variance_x = ((restored - mean_x) ** 2).mean(dim=(2, 3))
    variance_y = ((target - mean_y) ** 2).mean(dim=(2, 3))
    covariance = ((restored - mean_x) * (target - mean_y)).mean(dim=(2, 3))
    ssim = ((2 * mean_x.flatten(1) * mean_y.flatten(1) + 0.01**2) * (2 * covariance + 0.03**2) / ((mean_x.flatten(1).square() + mean_y.flatten(1).square() + 0.01**2) * (variance_x + variance_y + 0.03**2))).mean()
    overshoot = (F.relu(restored - 1.0).mean() + F.relu(-restored).mean())
    high_restored = restored - F.avg_pool2d(restored, 5, 1, 2)
    high_target = target - F.avg_pool2d(target, 5, 1, 2)
    high_baseline = baseline - F.avg_pool2d(baseline, 5, 1, 2)
    ringing = torch.mean(torch.abs(high_restored - high_target)) - torch.mean(torch.abs(high_baseline - high_target))
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "overshoot": float(overshoot),
        "ringing": float(ringing),
    }


def _progress(done: int, total: int) -> None:
    width = 24
    filled = round(width * done / max(1, total))
    print(
        f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} SR evaluate",
        end="\n" if done >= total else "\r",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
