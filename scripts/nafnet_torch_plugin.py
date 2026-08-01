"""使用可选 PyTorch 与本地 NAFNet 权重执行去模糊/降噪。

使用范例（Windows PowerShell）：
    .\.venv\Scripts\Activate.ps1
    python scripts/nafnet_torch_plugin.py `
      --input input.png --output output.png `
      --weights models/weights/NAFNet-GoPro-width32.pth `
      --width 32 --tile 256

说明：脚本不联网、不依赖 CUDA，作为 ScreenRestore 外部模型插件运行。网络结构移植
自 NAFNet ``basicsr/models/archs/NAFNet_arch.py`` 与 ``arch_util.py``；NAFNet 部分
为 MIT、Copyright (c) 2022 megvii-model，BasicSR 部分为 Apache-2.0、
Copyright 2018-2020 BasicSR Authors；修改与权重来源见 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from screenrestore.core.operator import ProcessingContext
from screenrestore.inference.tiled_inference import tiled_inference


def _load_torch():  # type: ignore[no-untyped-def]
    """延迟导入可选 PyTorch。"""

    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as exc:
        raise RuntimeError(
            "未安装可选 PyTorch；请按 MODEL_PLUGINS.md 安装 CPU 版本"
        ) from exc
    return torch, nn, functional


def _build_nafnet(width: int):  # type: ignore[no-untyped-def]
    """构造与官方 GoPro/SIDD 权重兼容的 NAFNet。"""

    torch, nn, functional = _load_torch()

    class LayerNorm2d(nn.Module):
        """逐像素沿通道归一化；推理公式等价于上游自定义 autograd 实现。"""

        def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.epsilon = epsilon

        def forward(self, values):  # type: ignore[no-untyped-def]
            mean = values.mean(1, keepdim=True)
            variance = (values - mean).pow(2).mean(1, keepdim=True)
            normalized = (values - mean) / (variance + self.epsilon).sqrt()
            return (
                self.weight.view(1, -1, 1, 1) * normalized
                + self.bias.view(1, -1, 1, 1)
            )

    class SimpleGate(nn.Module):
        """把通道一分为二并逐元素相乘。"""

        def forward(self, values):  # type: ignore[no-untyped-def]
            first, second = values.chunk(2, dim=1)
            return first * second

    class NAFBlock(nn.Module):
        """无显式非线性激活的空间/通道混合残差块。"""

        def __init__(self, channels: int) -> None:
            super().__init__()
            expanded = channels * 2
            self.conv1 = nn.Conv2d(channels, expanded, 1)
            self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
            self.conv3 = nn.Conv2d(expanded // 2, channels, 1)
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded // 2, expanded // 2, 1),
            )
            self.sg = SimpleGate()
            self.conv4 = nn.Conv2d(channels, channels * 2, 1)
            self.conv5 = nn.Conv2d(channels, channels, 1)
            self.norm1 = LayerNorm2d(channels)
            self.norm2 = LayerNorm2d(channels)
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
            self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
            self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

        def forward(self, values):  # type: ignore[no-untyped-def]
            output = self.conv1(self.norm1(values))
            output = self.conv2(output)
            output = self.sg(output)
            output = output * self.sca(output)
            output = self.dropout1(self.conv3(output))
            residual = values + output * self.beta
            output = self.conv4(self.norm2(residual))
            output = self.conv5(self.sg(output))
            return residual + self.dropout2(output) * self.gamma

    class NAFNet(nn.Module):
        """官方 width32/64、[1,1,1,28] 编码布局。"""

        def __init__(self) -> None:
            super().__init__()
            encoder_counts = (1, 1, 1, 28)
            decoder_counts = (1, 1, 1, 1)
            self.intro = nn.Conv2d(3, width, 3, padding=1)
            self.ending = nn.Conv2d(width, 3, 3, padding=1)
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            channels = width
            for count in encoder_counts:
                self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
                self.downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
                channels *= 2
            self.middle_blks = nn.Sequential(NAFBlock(channels))
            for count in decoder_counts:
                self.ups.append(
                    nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2))
                )
                channels //= 2
                self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
            self.padder_size = 2 ** len(self.encoders)

        def forward(self, values):  # type: ignore[no-untyped-def]
            original_height, original_width = values.shape[-2:]
            pad_height = (self.padder_size - original_height % self.padder_size) % self.padder_size
            pad_width = (self.padder_size - original_width % self.padder_size) % self.padder_size
            padded = functional.pad(values, (0, pad_width, 0, pad_height))
            output = self.intro(padded)
            skips = []
            for encoder, downsample in zip(self.encoders, self.downs, strict=True):
                output = encoder(output)
                skips.append(output)
                output = downsample(output)
            output = self.middle_blks(output)
            for decoder, upsample, skip in zip(
                self.decoders,
                self.ups,
                reversed(skips),
                strict=True,
            ):
                output = decoder(upsample(output) + skip)
            output = self.ending(output) + padded
            return output[..., :original_height, :original_width]

    return torch, NAFNet()


def main(argv: list[str] | None = None) -> int:
    """加载权重，以重叠 tile 运行同分辨率恢复并保存。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(32, 64), default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength 必须位于 0..1")
    for path in (args.input, args.weights):
        if not path.is_file():
            parser.error(f"文件不存在：{path}")

    print("[------------------------]   0.0% 加载 NAFNet", file=sys.stderr)
    torch, model = _build_nafnet(args.width)
    torch.set_num_threads(max(1, args.threads))
    checkpoint = torch.load(str(args.weights), map_location="cpu", weights_only=True)
    parameters = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(parameters, strict=True)
    model.eval()
    with Image.open(args.input) as opened:
        source = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()

    def infer_tile(tile_rgb: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.transpose(tile_rgb, (2, 0, 1)).copy()).float()
        tensor = tensor.unsqueeze(0) / 255.0
        with torch.inference_mode():
            inferred = model(tensor).squeeze(0).clamp_(0.0, 1.0)
            array = inferred.mul(255.0).round().to(torch.uint8).cpu().numpy()
        return np.transpose(array, (1, 2, 0)).copy()

    context = ProcessingContext(
        preview=False,
        progress=lambda fraction, message: _print_progress(fraction, message),
    )
    restored = tiled_inference(
        source,
        infer_tile,
        context,
        tile_size=args.tile,
        overlap=args.overlap,
        padding=args.padding,
    )
    if args.strength < 1.0:
        restored = np.clip(
            np.rint(
                source.astype(np.float32) * (1.0 - args.strength)
                + restored.astype(np.float32) * args.strength
            ),
            0,
            255,
        ).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.stem}.tmp{args.output.suffix}")
    Image.fromarray(restored, "RGB").save(temporary)
    temporary.replace(args.output)
    _print_progress(1.0, f"完成：{args.output}")
    return 0


def _print_progress(fraction: float, message: str) -> None:
    """输出固定宽度进度条。"""

    fraction = min(1.0, max(0.0, fraction))
    filled = round(fraction * 24)
    bar = "#" * filled + "-" * (24 - filled)
    print(f"\r[{bar}] {fraction * 100:5.1f}% {message}", end="", file=sys.stderr)
    if fraction >= 1.0:
        print(file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
