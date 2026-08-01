# 屏摄恢复 AI 候选核查

核查日期：2026-08-01。目标是为 ScreenRestore 的本地、CPU 可选插件寻找针对屏摄
摩尔纹、闪烁条带和复合退化的模型；“有论文或项目页”不等于“已有可分发代码与权重”。

## 当前结论

| 候选 | 任务与价值 | 代码/权重现状 | 许可与接入决定 |
| --- | --- | --- | --- |
| [CLEAR](https://arxiv.org/abs/2602.01559) | 同时去闪烁条带与摩尔纹；MIRAGE 数据和联合频域建模与本项目 3–5 号样本最匹配 | [官方仓库](https://github.com/libozhu03/CLEAR) 当前只有项目页素材，没有推理代码、权重或模型导出 | 未发现可用于代码/权重的明确许可证；只跟踪，不能复制或声称已接入 |
| [RIFLE](https://arxiv.org/abs/2509.24644) | 潜空间扩散去闪烁条带，提供条带先验和真实配对评估思路 | [官方仓库](https://github.com/libozhu03/RIFLE) 的安装、权重、推理仍为 TBD | 扩散模型不适合作为默认 CPU 路径；且代码/权重尚不可用，仅研究其合成退化和掩膜评估 |
| [BRACE](https://arxiv.org/abs/2606.29845) | 多曝光 RAW 融合去闪烁条带，物理假设比单帧推断更可靠 | [官方仓库](https://github.com/ZZH-qwq/BRACE) 当前安装、数据、权重和推理仍为 TODO | 未发现明确许可证；保留为未来 RAW 包围曝光输入协议参考，不复制代码 |
| [BinaryDemoire](https://arxiv.org/abs/2602.03176) | 1-bit 权重/激活的轻量去摩尔纹，理论上适合 CPU | [官方仓库](https://github.com/zhengchen1999/BinaryDemoire) 当前 TODO 仍是发布代码与预训练模型 | 仓库无许可证且没有代码/权重；不能接入，继续监测发布状态 |
| [ESDNet/UHDM](https://arxiv.org/abs/2207.09935) | 面向 4K、多尺度摩尔纹；比通用去模糊模型更贴合屏幕照片 | [官方代码](https://github.com/CVMI-Lab/UHDM) 已公开，代码仓库为 Apache-2.0 | 候选优先级最高；权重分发条款、CPU 内存、ONNX 动态尺寸和 8 组实拍收益仍须逐项验收后才能成为插件 |
| NAFNet GoPro width32 | 通用同分辨率去模糊 | 本地已有官方权重和可选 PyTorch/ONNX 导出路径 | 合成屏幕筛选出现明显干净内容漂移，运动/失焦 PSNR 均未改善；维持实验 restoration，不默认启用 |
| Real-ESRGAN general-x4v3 | 感知恢复与超分 | 本地已有官方强/弱权重和 CPU tile 插件 | 只作为 enhancement；8 组基准需与 Fidelity 并列，不能覆盖或冒充原始数字帧 |

## 接入门槛

1. 仓库和权重都有可核验的分发许可；未知许可只允许论文层面研究。
2. 无网络、无 GPU和无模型时，经典 Fidelity 仍完整可用。
3. 清单必须声明 `role=restoration` 或 `role=enhancement` 及明确 `task`。
4. 先通过合成干净输入保护、摩尔纹、条带、运动/失焦筛选，再运行 8 组配对实拍。
5. Restoration 至少不能系统性降低 PSNR、SSIM、梯度相关或扩大色差；Enhancement
   允许生成细节，但网页、文件名和诊断必须持续显示该声明。
6. 优先导出单输入动态尺寸 ONNX CPU；若必须通过外部进程，仍使用服务器清单白名单、
   `shell=False`、超时、取消和 ASCII 临时桥。

## 下一候选的可执行顺序

ESDNet 是当前唯一同时具备屏摄任务匹配和明确代码许可证的优先候选。实际接入前应先
核查官方权重条款，并用现有 `scripts/screen_model_plugins.py` 与
`scripts/evaluate_paired.py` 记录失败而非只展示挑选过的成功样例。CLEAR、RIFLE、
BRACE 和 BinaryDemoire 在代码、权重或许可证完备之前不会进入核心或模型安装脚本。
