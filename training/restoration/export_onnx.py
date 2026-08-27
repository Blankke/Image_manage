"""导出 BoundedResidualNet 为通用 image-to-image ONNX。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.restoration.export_onnx \\
        --checkpoint "$SCREENRESTORE_RUN_ROOT/restoration/fidelity-smoke/best.pt" \\
        --output "$SCREENRESTORE_RUN_ROOT/restoration/fidelity-smoke/fidelity-residual.onnx"

导出模型只接收和返回 N×3×H×W 的 float32 RGB [0,1]；它可由现有 ONNX restoration
后端配合模型清单使用。权重产物不会写入仓库。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .model import BoundedResidualNet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    print("[########----------------] 1/3 加载 Fidelity checkpoint", file=sys.stderr)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "BoundedResidualNet":
        raise ValueError("checkpoint 不是 BoundedResidualNet")
    model = BoundedResidualNet(
        int(checkpoint["channels"]),
        int(checkpoint["blocks"]),
        float(checkpoint["max_delta"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    patch_size = int(checkpoint["patch_size"])
    example = torch.zeros((1, 3, patch_size, patch_size), dtype=torch.float32)
    print("[################--------] 2/3 导出 ONNX", file=sys.stderr)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["image"],
        output_names=["restored_rgb"],
        opset_version=args.opset,
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "restored_rgb": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,
    )
    print(f"[########################] 3/3 完成：{output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
