# 参考项目清单

核查日期：2026-08-01。许可证以仓库顶层 `LICENSE`、官方发布页和本地浅克隆为
依据。用户明确声明其拥有所列论文源码及权重并授权当前论文项目直接利用；项目仍
逐项保留公开许可证、文件来源和修改记录。没有公开许可证且本轮没有必要复制的项目
继续只作思想参考，避免未来分发时权利边界不清。

本轮实际检出的关键 commit：fast-reflection-removal `1f987741`、DDA `bae419bf`、
MCNN `468b2677`、UnDeM `73cb9406`、NAFNet `2b4af71e`、Real-ESRGAN
`a4abfb29`、Real-ESRGAN-ncnn-vulkan `37026f49`、BIPNet `661fb9ee`。

| 项目 | 参考方向 | 核查到的许可证 | ScreenRestore 使用方式 |
| --- | --- | --- | --- |
| [OpenCV-Document-Scanner](https://github.com/andrewdcampbell/OpenCV-Document-Scanner) | 四边形检测、拖点交互 | 未发现明确许可证 | 仅理解思路，禁止复制代码 |
| [makeacopy](https://github.com/egdels/makeacopy) | 离线扫描产品结构 | 顶层 LICENSE 仅声明所含 OpenCV 文件为 Apache-2.0，项目整体授权不清晰 | 仅理解思路，禁止复制代码 |
| [OpenCV](https://github.com/opencv/opencv) | 轮廓、透视、滤波、色彩转换 | Apache-2.0 | 通过 PyPI 二进制依赖调用公开 API，不复制源码 |
| [HandyView](https://github.com/xinntao/HandyView) | 图片查看、缩放与比较交互 | MIT | 仅研究交互，不复制代码 |
| [fast-reflection-removal](https://github.com/JanPalasek/fast-reflection-removal) | 去反光插件方向 | MIT | 已移植梯度/DCT Poisson 核心并做数值、取消和依赖改造；影院预设中等强度启用，其他预设关闭，始终标为实验性 |
| [Awesome-Demoireing](https://github.com/rebeccaeexu/Awesome-Demoireing) | 去摩尔纹论文索引 | MIT | 仅作研究索引，不复制实现 |
| [DDA](https://github.com/zyxxmu/DDA) | 动态计算量分配思想 | 未发现明确许可证 | 采用“热度决定局部处理强度”思想；原测试硬编码 CUDA，未复制到核心 |
| [MoirePhotoRestoration-MCNN](https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN) | 多尺度去摩尔纹 | MIT | 已核对网络/测试代码；缺少适合当前实拍场景的即用权重，暂不迁入 |
| [UnDeM](https://github.com/zysxmu/UnDeM) | 摩尔纹合成与恢复 | 未发现明确许可证 | 已核对结构；本轮不引入训练依赖和数据集 |
| [VideoDemoireing](https://github.com/CVMI-Lab/VideoDemoireing) | 视频时序去摩尔纹 | Apache-2.0 | 仅为未来多帧方向调研，不复制代码 |
| [NAFNet](https://github.com/megvii-research/NAFNet) | 去噪与去模糊模型参考 | MIT | 已移植纯推理结构，提供 GoPro width32 权重安装与外部插件；默认不安装/不启用 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 可选超分辨率 | BSD-3-Clause | 已移植紧凑 general-x4v3 推理结构和 DNI，提供强/弱权重安装；默认不安装/不启用 |
| [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | 外部 ncnn-vulkan 后端 | MIT | 提供官方 Windows 包下载校验脚本和外部进程清单，不进入核心包 |
| [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) | 外部 ncnn-vulkan 后端 | MIT | 仅定义可选外部进程接入方式，不附带二进制 |
| [BIPNet](https://github.com/akshaydudhane16/BIPNet) | 多帧恢复 | 未发现明确许可证 | 仅理解论文思路，禁止复制代码 |
| [flicker_remover](https://github.com/flyeyesport/flicker_remover) | 多帧闪烁校正 | 未发现明确许可证 | 仅理解论文思路，禁止复制代码 |

## 使用原则

- 已有明确许可证不等于自动引入：真正复制、改写或分发代码时仍记录文件级来源、
  版权信息、修改内容和许可证要求。
- 用户的权利人授权只适用于当前项目，不被表述为这些仓库面向公众的通用许可证。
- 核心经典算法仍基于 OpenCV/NumPy/SciPy，三项模型/参考移植都可禁用且不污染
  无模型启动路径。
