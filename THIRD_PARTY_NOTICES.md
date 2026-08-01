# Third-Party Notices

ScreenRestore 的运行时 Python 依赖由各自许可证授权，具体锁定版本以构建环境生成的依赖清单为准。

## Python 运行时依赖

| 组件 | 用途 | 许可证（包元数据/上游） | 修改 |
| --- | --- | --- | --- |
| NumPy | 数组与频域计算 | BSD-3-Clause 及其分发中列出的兼容许可证 | 未修改源码 |
| SciPy | 一维 Gaussian 趋势平滑 | BSD-3-Clause | 未修改源码 |
| Pillow | 图像编解码和 EXIF | MIT-CMU | 未修改源码 |
| PySide6 / Qt for Python | Windows 桌面 GUI | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | 动态使用官方 wheel，未修改源码 |
| OpenCV Python wheel | 几何、滤波、颜色空间和修复 | Apache-2.0 | 未修改源码 |

## 开发与打包依赖

| 组件 | 用途 | 许可证 | 分发说明 |
| --- | --- | --- | --- |
| pytest | 测试 | MIT | 不进入运行时 |
| Ruff | 静态检查 | MIT | 不进入运行时 |
| PyInstaller | Windows 打包 | GPL-2.0-or-later，带允许分发构建应用的特殊例外 | 仅构建工具 |
| Hatchling | Python 包构建 | MIT | 仅构建工具 |

可选 `onnxruntime` 通常为 MIT；OpenVINO 为 Apache-2.0。它们默认不安装、不导入，也不随核心包分发。真正分发可选运行时或模型前必须对选定版本和权重另行核查。

## OpenCV

- 项目：https://github.com/opencv/opencv
- 许可证：Apache License 2.0
- 使用方式：通过 `opencv-python-headless`/`opencv-python` Python 包调用图像处理 API。
- 修改：未修改或复制 OpenCV 源码。

## 已移植或修改的代码

### fast-reflection-removal

- 来源：https://github.com/JanPalasek/fast-reflection-removal
- 上游文件：`src/python/frr/base.py`、`src/python/frr/core.py`
- 上游版本：commit `1f987741440afd686c48a703e07b5e29cc8df459`。
- 许可证：MIT，Copyright (c) 2024 Jan Palasek。
- 许可证全文：https://github.com/JanPalasek/fast-reflection-removal/blob/1f987741440afd686c48a703e07b5e29cc8df459/LICENSE
- 本项目文件：`src/screenrestore/operators/reflection_dct.py`。
- 使用与修改：移植梯度、散度、Laplacian、二维 DCT/逆 DCT 与 Poisson 求解；移除
  Matplotlib 和调试文件写入；修复上游 `remove_reflection` 重复求解；加入逐通道稳健
  分位数配准、混合强度、取消检查、float32 与除零保护。实验模式默认不启用。

### Real-ESRGAN

- 来源：https://github.com/xinntao/Real-ESRGAN
- 上游文件：`realesrgan/archs/srvgg_arch.py`，并参考 `realesrgan/utils.py` 的 DNI
  权重混合和 tile 推理约定。
- 上游版本：commit `a4abfb2979a7bbff3f69f58f58ae324608821e27`。
- 许可证：BSD-3-Clause，Copyright (c) 2021 Xintao Wang。
- 许可证全文：https://github.com/xinntao/Real-ESRGAN/blob/a4abfb2979a7bbff3f69f58f58ae324608821e27/LICENSE
- 本项目文件：`scripts/realesrgan_torch_plugin.py`、
  `models/examples/realesrgan-general-x4v3-torch.json`。
- 使用与修改：移植 `SRVGGNetCompact` x4/32-conv 结构；去除 BasicSR 依赖；改为
  本地权重、CPU、通用加权 tile、中文路径、终端进度、输出倍率与有限强度混合。
- 权重：安装脚本可从上游 v0.2.5.0 下载 `realesr-general-x4v3.pth` 与
  `realesr-general-wdn-x4v3.pth` 并校验 SHA-256。权重和 PyTorch 不进入核心包。

### NAFNet

- 来源：https://github.com/megvii-research/NAFNet
- 上游文件：`basicsr/models/archs/NAFNet_arch.py` 与 `arch_util.py` 中的
  `LayerNorm2d`。
- 上游版本：commit `2b4af71ebe098a92a75910c233a3965a3e93ede4`。
- 许可证：NAFNet 部分为 MIT，Copyright (c) 2022 megvii-model；其 `arch_util.py`
  同时携带 BasicSR 的 Apache-2.0 声明，Copyright 2018-2020 BasicSR Authors。
- 许可证全文：https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/LICENSE
- 本项目文件：`scripts/nafnet_torch_plugin.py`、`scripts/export_nafnet_onnx.py`、
  `models/examples/nafnet-gopro-width32-torch.json`、
  `models/examples/nafnet-gopro-width32-onnx.json`。
- 使用与修改：移植 width32/64 NAFNet 推理结构；去除 BasicSR、训练和 CUDA 依赖；
  用纯推理 LayerNorm 代替自定义反向传播；加入本地权重、CPU、通用加权 tile、中文
  路径、终端进度与有限强度混合；ONNX 转换器复用该已记录结构导出用户本地
  权重，不另行复制上游实现。
- 权重：安装脚本可从上游 README 指向的官方 Google Drive 下载
  `NAFNet-GoPro-width32.pth` 并校验 SHA-256。权重和 PyTorch 不进入核心包。

## 研究参考

用户指定的完整参考项目、许可证核查、实际移植范围和未采用原因见
`docs/REFERENCE_PROJECTS.md`；针对 CLEAR、RIFLE、BRACE、BinaryDemoire、ESDNet
等屏摄候选的代码/权重状态见 `docs/SCREEN_AI_RESEARCH.md`。除上面逐文件列出的三项
移植外，其余参考仓库未复制源码、权重、素材或二进制。

若未来引入第三方代码或分发外部组件，必须在本文件补充：来源仓库与版本、原始文件、版权声明、许可证、修改说明和分发方式。
