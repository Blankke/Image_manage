# ScreenRestore

ScreenRestore 是一款完全离线的屏幕照片恢复工具，用于处理手机斜拍显示器、电子海报、投影幕布、电影院银幕、LED 大屏及普通文档/PPT。它以平面透视恢复为起点，继续处理条带、摩尔纹、噪声、色偏、曝光和局部对比度，默认保留彩色内容、渐变、暗部与电影光影。

应用不会上传图片，不包含遥测，不调用云端 API。没有独立显卡、模型文件、PyTorch 或 CUDA 时，经典 CPU 流水线仍可完整启动和运行。

## 当前能力

- PNG、JPEG、WebP、BMP、TIFF；EXIF Orientation；中文路径；最长边 1600 像素代理预览。
- 自动四边形候选评分；贴边银幕的稳健直线细化；四角拖动、覆盖层、局部放大镜、边界约束、键盘微调、重置与重新检测。
- 带弱相机标定的 AUTO 物理比例恢复；自动/估计/自由、16:9、16:10、4:3、1.85:1、2.39:1 和自定义比例；纵向纸张不会被强制横向拉伸。
- 非破坏式可序列化流水线；算子启停与安全重排；节点 LRU 缓存；撤销/重做；异步 debounce 预览和取消。
- 周期条带与影院宽光幕校正、区域自适应色彩去摩尔纹、实验性 Gaussian notch、摩尔纹热图、反光蒙版、小区域修复及实验性梯度 DCT 反光抑制。
- 影院预设以中等强度启用梯度 DCT 来压低宽水平光幕；它可能同时压低真实低对比纹理，界面明确标为实验性并允许关闭。
- Gray World、White Patch、手动 RGB 增益、中性灰点选取；完整曝光/色调；LAB CLAHE；光照场校正；四种降噪；Unsharp Mask；实验性 Wiener 反卷积。
- 显示器、电子海报、电影院/投影、LED 大屏、文档/PPT、自定义预设。
- 原图/结果左右并排和可拖分割条；按住空格临时看原图；Fit/100%/200%；RGB/HSV；直方图、热图和频谱。
- `.screenrestore.json` 项目；相对源图路径、SHA-256 核对和重新定位；全分辨率后台 PNG/JPEG/WebP/TIFF 导出。
- 与 GUI 共用流水线的 CLI；JSON 诊断；滚动日志；PyInstaller Windows 配置。
- 真正位于流水线中的可选模型算子、外部程序、ONNX Runtime、OpenVINO 后端及无接缝 tile 融合；提供 NAFNet/Real-ESRGAN 移植脚本和可复现安装器。核心包不附带权重、PyTorch 或外部二进制。

## Windows 10/11 安装与运行

需要 64 位 Python 3.11。建议在 Windows 原生 PowerShell 中执行：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 install
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 run
```

等价的手工命令：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m screenrestore.app
```

打开一张初始图片：

```powershell
screenrestore-gui "D:\照片\倾斜屏幕.jpg"
```

## CLI

```powershell
screenrestore "D:\照片\输入.jpg" `
  --preset display `
  --corners auto `
  --output "D:\照片\输出.png" `
  --json-diagnostics
```

显式四角可使用像素或 `[0,1]` 归一化坐标，顺序会自动规范化：

```powershell
screenrestore input.jpg --corners 120,80 1810,65 1880,1010 90,1040 -o output.tiff
```

从项目处理：

```powershell
screenrestore --project work.screenrestore.json --output output.webp --quality 94
```

## 测试与打包

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 test
PowerShell -ExecutionPolicy Bypass -File scripts/windows.ps1 package
```

或手工执行：

```powershell
.\.venv\Scripts\Activate.ps1
python -m ruff check src tests scripts
python -m pytest
python -m PyInstaller --noconfirm packaging/screenrestore.spec
```

Windows 构建结果位于 `dist\ScreenRestore\ScreenRestore.exe`。PyInstaller 必须在 Windows 原生环境运行；WSL/Linux 构建不会产生 Windows 可执行文件。

## 可选模型

核心程序不会下载模型。在“设置/模型插件”中可查看本地清单状态；启用“可选模型
恢复/超分”步骤并填写清单路径后，GUI/CLI 会从当前流水线实际调用同一后端。

Windows 一键安装官方权重与 CPU PyTorch：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin torch
```

安装官方 Real-ESRGAN ncnn-vulkan Windows 包：

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install_optional_models.ps1 -Plugin ncnn
```

所有下载都校验 SHA-256，并保存到 Git 忽略目录。详细清单字段、直接运行命令、
效果取舍和安全注意事项见 [MODEL_PLUGINS.md](MODEL_PLUGINS.md)。

## 明确限制

- 当前几何仍以平面单应性为主；能细化直线边界，但不处理弯曲银幕或镜头畸变网格校正。
- 单张照片中已经饱和的反光、被遮挡内容、过曝高光、纯黑欠曝和采样混叠后丢失的细节无法真实恢复。
- 严重运动模糊、失焦和滚动快门形变只能有限改善；Wiener 反卷积是实验功能，参数不匹配会产生振铃。
- 传统去摩尔纹会在抑制伪色与保留细线之间取舍；自动频域陷波默认关闭。
- 当前不实现视频/多帧对齐恢复和批量队列；模型插件可实际运行，但不随核心软件分发模型。
- 学习模型会生成统计意义上的细节，不能保证等于原始数字帧；影院画面应优先比较 NAFNet、Real-ESRGAN 与经典结果后再导出。
- ScreenRestore 不把反光抑制或 inpainting 描述为对大面积丢失内容的真实恢复。

## 隐私与日志

所有图像处理均在本机完成。日志不会记录图片像素或诊断图，只记录应用/系统/依赖版本、后端、算子耗时和异常堆栈。Windows 日志位于：

```text
%LOCALAPPDATA%\ScreenRestore\logs\screenrestore.log
```

## 文档

- [架构](ARCHITECTURE.md)
- [算法与失败场景](ALGORITHMS.md)
- [模型插件](MODEL_PLUGINS.md)
- [开发记录](docs/DEVELOPMENT.md)
- [参考项目清单](docs/REFERENCE_PROJECTS.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)

截图统一放在 `docs/screenshots/`，便于后续 Windows 原生构建时更新，不将用户输入图片作为截图素材。
