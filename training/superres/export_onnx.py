"""导出独立 x2 或 wild-x4 超分 checkpoint 为 ONNX。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.superres.export_onnx --checkpoint best.pt --output superres.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import ConservativeSuperResolutionNet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "ConservativeSuperResolutionNet":
        raise ValueError("checkpoint 不是 ConservativeSuperResolutionNet")
    model = ConservativeSuperResolutionNet(
        int(checkpoint["scale"]),
        int(checkpoint["channels"]),
        int(checkpoint["blocks"]),
        float(checkpoint["max_delta"]),
    ).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["rgb_lr"],
        output_names=["rgb_hr"],
        dynamic_axes={"rgb_lr": {0: "batch", 2: "height", 3: "width"}, "rgb_hr": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
