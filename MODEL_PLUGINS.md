# 可选模型插件

模型插件不是 ScreenRestore 的必需组件。经典 CPU 流水线不导入 ONNX Runtime、OpenVINO、PyTorch、TensorFlow、CUDA 或 Vulkan。模型算子默认禁用；启用后填写清单路径即可从 GUI、项目或 CLI 的同一流水线运行。

## 清单

把 JSON 清单放在 `models/` 的任意子目录。仓库中的 `models/examples/realesrgan-x2.json` 只是未安装示例，不包含程序或权重。

```json
{
  "id": "realesrgan-x2",
  "name": "Real-ESRGAN x2",
  "type": "external_process",
  "executable": "../../plugins/realesrgan/realesrgan-ncnn-vulkan.exe",
  "arguments": ["-i", "{input}", "-o", "{output}", "-s", "2"],
  "supports_tiling": true,
  "license": "MIT",
  "homepage": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan",
  "timeout_seconds": 600
}
```

占位符：`{input}`、`{output}`、`{temp}`、`{manifest_dir}`。参数以字符串数组传入，后端固定 `shell=False`。`{manifest_dir}` 便于清单可靠引用仓库内脚本和权重，不依赖应用启动目录。`required_files` 可列出相对清单目录的权重/参数文件，使设置页在缺权重时明确显示不可用。

## 可复现安装

Windows PowerShell：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin torch
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin ncnn
```

安装器只写入 Git 忽略的 `models/weights/` 与 `plugins/realesrgan/`，并核验项目维护的
SHA-256。它不会改变核心 `pyproject.toml` 依赖，也不会把权重打入 PyInstaller 包。

安装后，在“可选模型恢复/超分”步骤中填写以下清单之一：

```text
models/examples/nafnet-gopro-width32-torch.json
models/examples/realesrgan-general-x4v3-torch.json
models/examples/realesrgan-x2.json
```

前两个示例按 Windows `.venv\Scripts\python.exe` 配置；Linux/WSL 验证时把清单的
`executable` 改为 `../../.venv/bin/python`。

## NAFNet 与 Real-ESRGAN PyTorch 插件

NAFNet 使用 GoPro width32 官方权重，输出保持原分辨率，适合实验性去模糊。它对
本仓库影院样例改善温和、电影质感较自然，但不能消除已经覆盖人物的宽光幕。

Real-ESRGAN 使用 general-x4v3 强/弱降噪官方权重，以 DNI 控制降噪，再通过强度
与 Lanczos 基线混合。它对印刷插画线条和头发轮廓提升明显，但可能产生绘画感和原图
不存在的纹理。默认清单只给 0.65 强度，不应把其输出描述为真实原始像素。

可绕过 GUI 直接复现：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/nafnet_torch_plugin.py --input input.png --output nafnet.png `
  --weights models/weights/NAFNet-GoPro-width32.pth --width 32 --strength 0.65
python scripts/realesrgan_torch_plugin.py --input input.png --output realesrgan.png `
  --weights models/weights/realesr-general-x4v3.pth `
  --weak-weights models/weights/realesr-general-wdn-x4v3.pth `
  --denoise-strength 0.35 --strength 0.6 --outscale 2
```

## 外部 ncnn-vulkan

1. 从项目官方发布页自行下载 Windows 版本。
2. 独立核查程序和模型权重许可证；程序许可证不一定覆盖权重。
3. 解压到例如 `plugins/realesrgan/`，不要把二进制或权重提交到仓库；也可使用上面的校验安装器。
4. 调整清单的相对 `executable` 和参数。
5. 在“设置/模型插件”查看可用性。

外部后端把输入复制到 ASCII 临时目录，可处理原始中文路径；捕获 stdout/stderr、检查退出码、支持超时和取消，并在结束后清理临时文件。

## ONNX Runtime

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[inference-onnx]"
```

清单使用 `"type": "onnx"` 和 `"model_path": "model.onnx"`。当前通用骨架面向单输入 NCHW RGB 图像到图像模型；特殊归一化、多输入或动态输出模型需要专用适配器。

## OpenVINO

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[inference-openvino]"
```

清单使用 `"type": "openvino"` 和 `"model_path"`。默认编译到 CPU。

## Tile 融合

通用工具支持 tile size、overlap、padding、边缘反射填充、整数倍率输出、线性加权融合、进度和取消。测试覆盖不规则边缘 tile 的 1× 原样合并与 2× 放大，要求逐像素一致。

## 安全提示

模型和外部可执行文件具有本机代码权限。仅从可信官方来源安装并校验哈希；ScreenRestore 不自动下载、升级或执行未由清单明确指定的文件。
