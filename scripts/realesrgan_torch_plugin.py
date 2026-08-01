"""使用可选 PyTorch 与本地 Real-ESRGAN general-x4v3 权重处理图像。

使用范例（Windows PowerShell）：
    .\.venv\Scripts\Activate.ps1
    python scripts/realesrgan_torch_plugin.py `
      --input input.png --output output.png `
      --weights models/weights/realesr-general-x4v3.pth `
      --weak-weights models/weights/realesr-general-wdn-x4v3.pth `
      --denoise-strength 0.5 --outscale 2

说明：该脚本是可选外部插件，不会被核心 GUI 导入。它不联网下载权重，支持中文
路径、CPU、tile 加权融合和终端进度。网络结构移植自 Real-ESRGAN 的
``realesrgan/archs/srvgg_arch.py``，BSD-3-Clause，Copyright (c) 2021 Xintao Wang。
来源和修改记录见 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from screenrestore.core.operator import ProcessingContext
from screenrestore.inference.tiled_inference import tiled_inference


def _load_torch():  # type: ignore[no-untyped-def]
    """延迟导入可选 PyTorch，缺失时给出可执行安装提示。"""

    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as exc:
        raise RuntimeError(
            "未安装可选 PyTorch；请按 MODEL_PLUGINS.md 安装 CPU 版本"
        ) from exc
    return torch, nn, functional


def _build_model():  # type: ignore[no-untyped-def]
    """构造上游 SRVGGNetCompact x4/32-conv 网络。"""

    torch, nn, functional = _load_torch()

    class SRVGGNetCompact(nn.Module):
        """Real-ESRGAN 的紧凑 VGG 超分网络，末端 PixelShuffle x4。"""

        def __init__(self) -> None:
            super().__init__()
            features = 64
            body: list[object] = [nn.Conv2d(3, features, 3, 1, 1), nn.PReLU(features)]
            for _ in range(32):
                body.extend((nn.Conv2d(features, features, 3, 1, 1), nn.PReLU(features)))
            body.append(nn.Conv2d(features, 3 * 4 * 4, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(4)

        def forward(self, values):  # type: ignore[no-untyped-def]
            output = values
            for layer in self.body:
                output = layer(output)
            output = self.upsampler(output)
            return output + functional.interpolate(values, scale_factor=4, mode="nearest")

    return torch, SRVGGNetCompact()


def _load_parameters(
    torch,  # type: ignore[no-untyped-def]
    strong_path: Path,
    weak_path: Path | None,
    denoise_strength: float,
):  # type: ignore[no-untyped-def]
    """读取官方 params_ema；可按上游 DNI 方式线性混合强/弱降噪权重。"""

    strong = torch.load(str(strong_path), map_location="cpu", weights_only=True)
    strong_parameters = strong.get("params_ema", strong.get("params", strong))
    if weak_path is None or denoise_strength >= 1.0:
        return strong_parameters
    weak = torch.load(str(weak_path), map_location="cpu", weights_only=True)
    weak_parameters = weak.get("params_ema", weak.get("params", weak))
    return {
        key: strong_parameters[key] * denoise_strength
        + weak_parameters[key] * (1.0 - denoise_strength)
        for key in strong_parameters
    }


def main(argv: list[str] | None = None) -> int:
    """解析参数、分块推理并原子保存 RGB 输出。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weak-weights", type=Path)
    parser.add_argument("--denoise-strength", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--outscale", type=float, default=2.0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 0.0 <= args.denoise_strength <= 1.0:
        parser.error("--denoise-strength 必须位于 0..1")
    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength 必须位于 0..1")
    if not 1.0 <= args.outscale <= 4.0:
        parser.error("--outscale 必须位于 1..4")
    for path in (args.input, args.weights):
        if not path.is_file():
            parser.error(f"文件不存在：{path}")
    if args.denoise_strength < 1.0 and (
        args.weak_weights is None or not args.weak_weights.is_file()
    ):
        parser.error("降噪强度小于 1 时必须提供 --weak-weights")

    print("[------------------------]   0.0% 加载 Real-ESRGAN", file=sys.stderr)
    torch, model = _build_model()
    torch.set_num_threads(max(1, args.threads))
    parameters = _load_parameters(
        torch,
        args.weights,
        args.weak_weights,
        args.denoise_strength,
    )
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
    output = tiled_inference(
        source,
        infer_tile,
        context,
        tile_size=args.tile,
        overlap=args.overlap,
        padding=args.padding,
    )
    target_size = (
        max(1, round(source.shape[1] * args.outscale)),
        max(1, round(source.shape[0] * args.outscale)),
    )
    if output.shape[1::-1] != target_size:
        output = cv2.resize(output, target_size, interpolation=cv2.INTER_LANCZOS4)
    if args.strength < 1.0:
        baseline = cv2.resize(source, target_size, interpolation=cv2.INTER_LANCZOS4)
        output = np.clip(
            np.rint(
                baseline.astype(np.float32) * (1.0 - args.strength)
                + output.astype(np.float32) * args.strength
            ),
            0,
            255,
        ).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.stem}.tmp{args.output.suffix}")
    Image.fromarray(output, "RGB").save(temporary)
    temporary.replace(args.output)
    _print_progress(1.0, f"完成：{args.output}")
    return 0


def _print_progress(fraction: float, message: str) -> None:
    """输出固定宽度进度条，供用户和外部后端捕获。"""

    fraction = min(1.0, max(0.0, fraction))
    filled = round(fraction * 24)
    bar = "#" * filled + "-" * (24 - filled)
    print(f"\r[{bar}] {fraction * 100:5.1f}% {message}", end="", file=sys.stderr)
    if fraction >= 1.0:
        print(file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
