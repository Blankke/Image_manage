# 参考项目清单

核查日期：2026-08-23。许可证以仓库顶层 `LICENSE`、官方发布页和本地浅克隆为
依据。用户明确声明其拥有所列论文源码及权重并授权当前论文项目直接利用；项目仍
逐项保留公开许可证、文件来源和修改记录。没有公开许可证且本轮没有必要复制的项目
继续只作思想参考，避免未来分发时权利边界不清。

本轮实际检出的关键 commit：fast-reflection-removal `1f987741`、DDA `bae419bf`、
MCNN `468b2677`、UnDeM `73cb9406`、NAFNet `2b4af71e`、Real-ESRGAN
`a4abfb29`、Real-ESRGAN-ncnn-vulkan `37026f49`、BIPNet `661fb9ee`、DocAligner
`3275b0f0`。

| 项目 | 参考方向 | 核查到的许可证 | ScreenRestore 使用方式 |
| --- | --- | --- | --- |
| [OpenCV-Document-Scanner](https://github.com/andrewdcampbell/OpenCV-Document-Scanner) | 四边形检测、拖点交互 | 未发现明确许可证 | 仅理解思路，禁止复制代码 |
| [DocAligner](https://github.com/DocsaidLab/DocAligner) | 角点热图、边缘和存在性监督的文档四角基线 | Apache-2.0 | 通过独立 PyPI 依赖运行未修改上游模型；不复制源码或权重。LCNet100 在四场景完整四角 0/4，FastViT-SA24 为 2/4，后者已检出样本平均 NCE 2.95%、IoU 0.8849；无语义 boundary 的纯梯度精修为 3.18%/0.8775。模型不具备 content/outer 层级和产品接受置信度 |
| [makeacopy](https://github.com/egdels/makeacopy) | 离线扫描产品结构 | 顶层 LICENSE 仅声明所含 OpenCV 文件为 Apache-2.0，项目整体授权不清晰 | 仅理解思路，禁止复制代码 |
| [OpenCV](https://github.com/opencv/opencv) | 轮廓、透视、滤波、色彩转换 | Apache-2.0 | 通过 PyPI 二进制依赖调用公开 API，不复制源码 |
| [HandyView](https://github.com/xinntao/HandyView) | 图片查看、缩放与比较交互 | MIT | 仅研究交互，不复制代码 |
| [fast-reflection-removal](https://github.com/JanPalasek/fast-reflection-removal) | 去反光插件方向 | MIT | 已移植梯度/DCT Poisson 核心并做数值、取消和依赖改造；所有 Fidelity 预设默认关闭，始终标为实验性 |
| [Awesome-Demoireing](https://github.com/rebeccaeexu/Awesome-Demoireing) | 去摩尔纹论文索引 | MIT | 仅作研究索引，不复制实现 |
| [ESDNet / UHDM](https://github.com/CVMI-Lab/UHDM) | 4K 屏幕照片去摩尔纹候选 | 代码仓库 Apache-2.0；权重分发条款仍需独立核查 | 代码许可明确，但尚未通过 CPU/ONNX 和 8 组配对筛选；暂不复制结构或分发权重 |
| [Moiré Zero / MZNet](https://github.com/sngryongLee/Moire-Zero) | 高分辨率高效去摩尔纹候选 | 核查时仓库未提供顶层许可证 | 仅记录论文与部署候选；禁止复制代码、配置或权重，等待上游明确授权 |
| [CLEAR](https://github.com/libozhu03/CLEAR) | 联合去闪烁条带与摩尔纹 | 当前仓库仅项目页，未发现代码/权重许可证 | 仅研究论文和数据设计，禁止复制或宣称已接入 |
| [RIFLE](https://github.com/libozhu03/RIFLE) | 扩散式闪烁条带去除 | 安装、权重与推理仍为 TBD，未发现仓库许可证 | 仅研究退化模拟与掩膜评估，不进入 CPU 默认路径 |
| [BRACE](https://github.com/ZZH-qwq/BRACE) | 多曝光 RAW 条带恢复 | 安装、数据、权重与推理仍为 TODO，未发现仓库许可证 | 仅作为未来多曝光 RAW 协议参考 |
| [BinaryDemoire](https://github.com/zhengchen1999/BinaryDemoire) | 1-bit 轻量去摩尔纹 | 当前尚未发布代码/权重且无许可证 | 继续监测，不能接入或分发 |
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

## P2 geometry 数据参考

| 数据源 | 官方入口 | 许可核查 | 本项目选择规则 |
| --- | --- | --- | --- |
| MIDV-500 | `ftp://smartengines.com/midv-500/` | 官方 `license.txt` 为 CC-BY-SA-2.5 | 固定 10/50 document types；同一 document 与相邻视频帧不跨 split |
| MIDV-Holo | `ftp://smartengines.com/midv-holo/` 与 SmartEngines 官方仓库 | 官方 `license.txt` 为 CC-BY-SA-2.5，并提示 Generated Photos attribution | 按 sample kind×A-E 光照×设备×ID/passport 每格固定抽取 clip；同一虚构 document 的 original/fraud 变体同组 |
| The Met Collection API | `https://metmuseum.github.io/` | Open Access 数据为 CC0；图像逐对象要求 `isPublicDomain=true` | painting/print/photograph 轮转，要求 primary image，保存 object ID 与来源 metadata |
| COCO 2017 | `https://cocodataset.org/#download` | 数据集条款与 image-level Flickr license 分别适用 | 仅官方 val2017 5K 背景池，不下载 18 GB train2017 |

上述数据、归档、来源 metadata 和生成后的合成图片均只保存在
`$SCREENRESTORE_DATA_ROOT`，由 `.gitignore` 排除。仓库仅维护 schema、下载与清单生成规则。

## 使用原则

- 已有明确许可证不等于自动引入：真正复制、改写或分发代码时仍记录文件级来源、
  版权信息、修改内容和许可证要求。
- 用户的权利人授权只适用于当前项目，不被表述为这些仓库面向公众的通用许可证。
- 核心经典算法仍基于 OpenCV/NumPy/SciPy，三项模型/参考移植都可禁用且不污染
  无模型启动路径。
