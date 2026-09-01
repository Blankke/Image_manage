# P3 数据、许可与磁盘审计

审计时间：2026-08-30（Asia/Shanghai）。

## 当前容量

- 数据根：`/Users/caozichen/screenrestore-data`，约 23 GiB。
- P3 硬上限：30 GiB；剩余预算约 7 GiB。
- 当前磁盘可用：约 53 GiB。
- 主要占用：geometry 约 15 GiB、superres 约 6.5 GiB、backgrounds 约 787 MiB、textures 约 179 MiB、private 约 136 MiB、manifests 约 94 MiB。
- P3 默认不下载新大型数据，不生成 augmentation cache。

## 已有可用清单

| 清单 | 样本 | 独立 group | 切分摘要 | P3 用途 |
|---|---:|---:|---|---|
| P2 all geometry | 56,340 | 24,084 | train 45,034 / validation 5,624 / test 5,682 | B0/B1/B3/B5 几何 |
| P2 calibration | 5,624 | 2,411 | validation only | correctness calibrator |
| SmartDoc | 24,889 | 30 | train 20,760 / validation 2,450 / test 1,679 | postcard content 几何；独立 group 偏少 |
| MIDV-500 | 3,000 | 10 | train 2,400 / validation 300 / test 300 | 文档透视与 partial |
| MIDV-Holo | 4,405 | 20 | train 2,648 / validation 464 / test 1,293 | 反光条件下文档几何；不等同 reflection paired restoration |
| private geometry | 46 | 24 | train 26 / validation 10 / test 10 | 画作/屏幕/多目标本地回归 |
| DIV2K restoration | 900 | 当前旧清单 group 字段不满足 P3 | train 800 / validation 100 | Fidelity/SR 内容源；需重建 P3 schema 清单 |

## 许可状态

- SmartDoc：沿用仓库现有来源与使用审计，仅用于几何清单所声明用途。
- MIDV-500、MIDV-Holo：本地数据集 `license.txt` 为 CC-BY-SA-2.5；MIDV-Holo 还需保留 Generated Photos attribution 提示。现有约束记录在 `THIRD_PARTY_NOTICES.md` 与 `docs/REFERENCE_PROJECTS.md`。
- DIV2K：沿用 NTIRE/DIV2K 官方条款；只复用已下载 HR/LR，不重新下载。
- private：本地受控使用，`private_local_only`，不得提交或上传。

## 阻塞数据

### FHDMi 或其他真实 paired 去摩尔纹数据：BLOCKED

- 缺少路径：`/Users/caozichen/screenrestore-data/demoire/fhdmi`。
- 所需格式：P3 restoration JSONL，包含数字 clean reference、实拍 moire input、subject/group/session、设备/屏幕参数、alignment 与许可字段。
- 许可：必须核对图片、标注和再分发条款；代码许可不能替代数据许可。
- 磁盘：必须先确认保留量落在剩余约 7 GiB 预算内。
- 获取方式：仅从数据集官方发布页或作者明确授权渠道手动获取。
- 指标影响：P3 只能完成 synthetic Demoire smoke，真实周期能量抑制与 clean texture 保留结论为 BLOCKED。

### 真实 paired reflection restoration：BLOCKED

- 缺少路径：`/Users/caozichen/screenrestore-data/reflection/paired`；当前 reflection 目录为空。
- 所需格式：单帧或多帧 observed input、对齐 reference、reflection mask、unresolved mask、alignment quality 和 provenance 字段。
- 许可：必须为明确授权采集或可用于训练/评估的公开许可。
- 磁盘：建议先采集小型本地验证集，预计保留量控制在 1–2 GiB。
- 获取方式：优先按 `docs/P3_DATA_CAPTURE_GUIDE.md` 本地采集，不抓取来源不明图片。
- 指标影响：单帧反射真实恢复、mask-localized change 和 saturated unresolved 的真实集结论为 BLOCKED；现有 MIDV-Holo 只可做几何/反光场景诊断。

### 真实 mild dewarp paired data：BLOCKED

- 缺少路径：`/Users/caozichen/screenrestore-data/dewarp/paired`。
- 所需格式：弯曲输入、正面参考、逆 dense grid/可靠配准、直线标记与许可元数据。
- 许可与获取：优先使用自制授权测试卡和本地采集；本轮不引入 Doc3D。
- 磁盘：建议首批小于 1 GiB。
- 指标影响：D2 只验证合成 grid、正 Jacobian、直线与 ONNX/MPS smoke，真实弯曲能力为 BLOCKED。

## 下载门

任何后续下载必须先打印并人工确认：下载压缩量、解压峰值、最终保留量、数据根预计总量、许可来源和删除临时 archive 的时点。预计总量超过 30 GiB 时硬失败。
