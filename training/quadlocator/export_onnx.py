"""将 QuadLocator-S checkpoint 导出为正式运行时 ONNX 契约。

使用范例：
    source .venv/bin/activate
    which python
    python -m training.quadlocator.export_onnx --checkpoint runs/quadlocator-s/best.pt \
        --output models/weights/quadlocator-s.onnx

需要训练环境额外安装 ``onnx``。脚本显示导出进度，并用固定输出名对齐
``OnnxQuadDetector``；不会下载或打包模型权重。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from training.quadlocator.model import QuadLocatorExportWrapper, QuadLocatorS

OUTPUT_NAMES = [
    "content_corner_heatmaps",
    "outer_corner_heatmaps",
    "content_mask_logits",
    "boundary_logits",
    "presence_logits",
    "outer_presence_logits",
    "class_logits",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    print("[####--------------------] 1/3 加载 checkpoint", file=sys.stderr)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise RuntimeError("只允许导出 format_version=2 的 7-output QuadLocator checkpoint")
    model = QuadLocatorS(float(checkpoint["width_multiplier"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    wrapper = QuadLocatorExportWrapper(model)
    image_size = int(checkpoint["image_size"])
    example = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print("[############------------] 2/3 导出 ONNX", file=sys.stderr)
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["image"],
        output_names=OUTPUT_NAMES,
        opset_version=args.opset,
        dynamic_axes={"image": {0: "batch"}},
        dynamo=False,
    )
    print(f"[########################] 3/3 完成：{output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
