# P2 自动几何训练结果（2026-08-30）

## 结论

P2 的数据准备、private 人工标注、Stage A～D、ONNX 导出、SmartDoc `e2e_auto`、分数据源
冻结评估和 overlay 人工复核均已完成。正式 run 为：

```text
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658
```

训练工程状态为 **完成**，产品自动几何验收状态为 **FAIL**。本轮修复了 P1 的关键模型契约：
QuadLocator 现在显式输出并监督 `outer_presence_logits`，运行时不会再把无依据的 outer heatmap
强行解释为外框。Stage B 第 12 轮是本轮最好的通用检查点，SmartDoc 自动接受从 P1 的 0 提升到
253 / 1,679，accepted precision 达到 34.78%，IoU 中位数达到 0.8844，说明训练方向取得了实质
进展。

该结果距离发布门槛仍很远。Stage B 的 253 个 accepted 中只有 88 个满足严格正确条件，另外
165 个均为几何误差；boundary F1 仍为 0，SmartDoc NCE P95 为 13.54%，且 SmartDoc test 只有
2 个独立 group。Stage D 无法在 99% 最低 precision 下为 corner、boundary、layer 或 combined
找到非空阈值，最终产品代理仍为 0 accepted。因此：

- Stage B `best.pt` 只作为下一轮开发与消融基线，不进入无人值守产品路径。
- Stage C 在仅 5 个 private validation group 上选模，泛化表现较 Stage B 退化，不作为最终候选。
- Stage D 的阈值针对 Stage C，且 combined gate 为空，不能移植到 Stage B 或作为部署配置。
- 正式 release gate 保持原值，不通过降低阈值换取覆盖率。
- 本轮没有重训 Fidelity、restoration 或 super-resolution，P1 恢复模型保持不变。

## P1 → P2 产品指标

下表使用各轮正式 SmartDoc test `e2e_auto` 报告。P1 没有 accepted，因此它的 accepted NCE 与
IoU 是保守空集值；这些列用于产品状态对照，不能解释成等价样本上的连续提升。

| 指标 | P1 | P2 Stage B | release gate | 结论 |
| --- | ---: | ---: | ---: | --- |
| 独立 test group | 2 | 2 | ≥100 | FAIL，证据不足 |
| 自动接受 | 0 / 1,679 | 253 / 1,679 | — | 覆盖开始恢复 |
| accepted precision | 0 | 0.347826 | ≥0.99 | FAIL |
| in-scope coverage | 0 | 0.150685 | ≥0.90 | FAIL |
| accepted NCE P95 | 1.0 | 0.135372 | ≤0.01 | FAIL |
| accepted IoU median | 0 | 0.884393 | ≥0.97 | FAIL |
| accepted IoU P05 | 0 | 0.538319 | ≥0.93 | FAIL |
| wrong layer rate | 0 | 0 | ≤0.005 | 数值 PASS；样本证据不足 |

P1 对所有 proposed quad 的诊断 NCE median / P95 为 0.003218 / 0.125125，IoU median / P05
为 0.988417 / 0.454818。P2 恢复自动接受后暴露出真实产品风险：中位几何已经可用，尾部仍会
产生高置信度错误，当前不能无人值守导出。

## 模型契约与产物

| 项目 | P1 | P2 |
| --- | ---: | ---: |
| 参数量 | 99,503 | 99,632（+129，约 +0.13%） |
| 输入尺寸 | 512×512 | 512×512 |
| ONNX 输出 | 6 | 7，新增 outer presence |
| checkpoint format | v1 | v2 |
| 训练设备 | Apple MPS | Apple MPS |

P1 warm start 共加载 99,503 个参数；P2 新增 `outer_presence_head.weight` 与
`outer_presence_head.bias`，其余参数按名称与 shape 加载。

本轮三个 ONNX 均已成功导出，文件大小均为 428,860 字节：

| 阶段 | 文件 | SHA-256 |
| --- | --- | --- |
| A | `stage-a/quadlocator-s.onnx` | `404e213f1816099c9567efb195d6902418d0c00edc68fee99a557ea809c77786` |
| B | `stage-b/quadlocator-s.onnx` | `e73ce6912205c210fbbbdbd66ddefc0c9fba27cdd5c41853019badec98c1ab48` |
| C | `stage-c/quadlocator-s.onnx` | `080ea7a8990fb8e323e2f325df089df31c025644f83038c0951e7a3870eab50f` |

推荐保留的开发基线为：

```text
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/best.pt
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/quadlocator-s.onnx
```

其 SHA-256 分别为：

```text
3344ad62709abf9e413a1cdebbdb82b1c0add0655ffdaeadcd9ddcc6bac86746  best.pt
e73ce6912205c210fbbbdbd66ddefc0c9fba27cdd5c41853019badec98c1ab48  quadlocator-s.onnx
```

## 数据规模、分布与磁盘

P2 全量清单含 56,340 个样本、24,084 个 group。Stage B 不读取 private，使用 56,294 个公开
与合成样本；Stage C 使用 private train 与受控 public replay，validation/test 只保留 private
对应 split。

| 数据源 | 样本 | group | train | validation | test | 类别分布 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SmartDoc | 24,889 | 30 | 20,760 | 2,450 | 1,679 | postcard 24,889 |
| MIDV-500 | 3,000 | 10 | 2,400 | 300 | 300 | postcard 1,896；none 1,104 |
| MIDV-Holo | 4,405 | 20 | 2,648 | 464 | 1,293 | postcard 3,966；none 439 |
| synthetic | 24,000 | 24,000 | 19,200 | 2,400 | 2,400 | artwork 5,041；postcard 5,997；screen 5,246；none 7,716 |
| private-labeled | 46 | 24 | 26 | 10 | 10 | artwork 28；postcard 5；none 13 |

Stage B 中有 13,726 个 outer 正样本，全部来自 synthetic；synthetic 另含 1,207 个多目标或
层级歧义拒绝样本，其中 validation 112 个。private 的 46 张图片按内容指纹合并为 24 个 group：
train 14、validation 5、test 5。人工标注的多目标同框共有 8 张、4 个 group，各 split 分别为
train 4 张、validation 2 张、test 2 张；它们统一使用 `ambiguous=true`、`present=false`。

Stage C 共 2,446 个样本、888 个 group，其中 private 46 张、SmartDoc replay 1,043 张、
MIDV-500 replay 276 张、MIDV-Holo replay 223 张、synthetic replay 858 张。它的 validation
只有 private 10 张 / 5 group，类别为 artwork 8、none 2，没有 postcard 和 screen。这是 Stage C
选模不稳定的直接原因。

| 目录 | 占用 |
| --- | ---: |
| 全部 `screenrestore-data` | 24,085,884 KiB，约 22.97 GiB |
| SmartDoc | 1,714,900 KiB，约 1.64 GiB |
| MIDV-500 | 9,918,076 KiB，约 9.46 GiB |
| MIDV-Holo | 2,165,056 KiB，约 2.06 GiB |
| synthetic | 2,246,012 KiB，约 2.14 GiB |
| private | 139,308 KiB，约 0.13 GiB |
| 当前 P2 run | 64,132 KiB，约 62.63 MiB |
| 全部 run | 83,600 KiB，约 81.64 MiB |
| 当前磁盘可用 | 56,021,144 KiB，约 53.43 GiB |

P1 报告记录的数据快照为 8.3 GiB；按两次快照估算，P2 净增加约 14.7 GiB。当前总数据仍低于
30 GiB 上限，约有 7.0 GiB 余量。许可证与权利边界已记录在 `THIRD_PARTY_NOTICES.md` 和
`docs/REFERENCE_PROJECTS.md`：MIDV-500 / MIDV-Holo 为 CC-BY-SA-2.5，Met 仅使用
`isPublicDomain=true` 的 CC0 对象，COCO 按图片级许可约束，训练图片不提交仓库。

## 分阶段训练与学习曲线

| 阶段 | 数据角色 | 配置 | 完成轮数 | 选中轮 | selection score | checkpoint val loss | 耗时 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | P1 warm start + outer 契约修复 | LR 3e-4，batch 4 | 3 / 3 | 2 | 0.528830 | 0.567940 | 4,449.44 秒（1:14:09） |
| B | public mixed-domain 正式训练 | LR 2e-4，patience 5 | 17 / 18 | 12 | 0.541263 | 0.710569 | 69,332.92 秒（19:15:33） |
| C | private train + public replay | LR 7.5e-5，patience 3 | 4 / 8 | 1 | 0.518010 | 3.280998 | 969.08 秒（0:16:09） |
| D | 冻结 Stage C，只做 validation calibration | minimum precision 0.99 | — | — | — | — | 约 157 秒 |

A～C 记录训练耗时合计 74,751.45 秒，约 20 小时 45 分 51 秒；按 Stage C 后处理完成到
calibration 文件落盘时间估算，含 Stage D 共约 20 小时 48 分 28 秒。

Stage A 学习曲线：

| epoch | train loss | val loss | score | NCE P95 | IoU median / P05 | mask IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.5454 | 0.5771 | 0.5210 | 0.1695 | 0.4897 / 0.1496 | 0.7585 |
| 2 | 1.3822 | 0.5679 | **0.5288** | 0.1688 | 0.5214 / 0.1362 | 0.7852 |
| 3 | 1.2519 | 0.5143 | 0.5154 | 0.1659 | 0.4941 / 0.1088 | 0.8037 |

Stage B 学习曲线：

| epoch | train loss | val loss | score | NCE P95 | IoU median / P05 | mask IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.4723 | 0.8813 | 0.5346 | 0.1595 | 0.5960 / 0.1459 | 0.7138 |
| 2 | 1.2896 | 0.8751 | 0.5119 | 0.1604 | 0.5505 / 0.1001 | 0.7266 |
| 3 | 1.2275 | 0.8059 | 0.5163 | 0.1591 | 0.5855 / 0.0652 | 0.7541 |
| 4 | 1.1931 | 0.8034 | 0.5169 | 0.1605 | 0.6010 / 0.0480 | 0.7585 |
| 5 | 1.1547 | 0.8002 | 0.5169 | 0.1573 | 0.6058 / 0.0419 | 0.7526 |
| 6 | 1.1350 | 0.7234 | 0.5358 | 0.1624 | 0.6525 / 0.0649 | 0.7630 |
| 7 | 1.0792 | 0.7134 | 0.5373 | 0.1567 | 0.6648 / 0.0489 | 0.7625 |
| 8 | 1.0457 | 0.7271 | 0.5309 | 0.1580 | 0.6571 / 0.0346 | 0.7559 |
| 9 | 1.0255 | 0.7078 | 0.5277 | 0.1576 | 0.6423 / 0.0359 | 0.7749 |
| 10 | 1.0099 | 0.7271 | 0.5327 | 0.1530 | 0.6570 / 0.0331 | 0.7536 |
| 11 | 1.0040 | 0.6855 | 0.5236 | 0.1560 | 0.6397 / 0.0180 | 0.7763 |
| 12 | 0.9824 | 0.7106 | **0.5413** | 0.1504 | 0.6938 / 0.0289 | 0.7710 |
| 13 | 0.9725 | 0.7055 | 0.5239 | 0.1539 | 0.6437 / 0.0182 | 0.7633 |
| 14 | 0.9724 | 0.7163 | 0.5326 | 0.1537 | 0.6689 / 0.0243 | 0.7745 |
| 15 | 0.9657 | 0.6806 | 0.5353 | 0.1539 | 0.6788 / 0.0196 | 0.7689 |
| 16 | 0.9570 | 0.6676 | 0.5365 | 0.1546 | 0.6742 / 0.0293 | 0.7776 |
| 17 | 0.9547 | 0.6867 | 0.5407 | 0.1529 | 0.6949 / 0.0204 | 0.7792 |

Stage B 的 train loss 持续下降，但 NCE P95 长期停留在 0.15 左右，IoU P05 也没有随 loss
改善。这说明继续堆 epoch 不能解决尾部几何错误，early stopping 在第 17 轮终止是合理的。

Stage C 学习曲线：

| epoch | train loss | val loss | score | NCE P95 | IoU median / P05 | mask IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.6930 | 3.2810 | **0.5180** | 0.1889 | 0.5476 / 0.2680 | 0.6276 |
| 2 | 1.3743 | 3.4424 | 0.4686 | 0.1943 | 0.5425 / 0.2555 | 0.5135 |
| 3 | 1.2777 | 3.4066 | 0.5001 | 0.1884 | 0.5460 / 0.2734 | 0.5586 |
| 4 | 1.1685 | 3.0521 | 0.4693 | 0.1888 | 0.5450 / 0.2739 | 0.5732 |

在固定的 1,000 张 Stage B public validation 子集上重新比较，Stage B / Stage C 的 selection
score 为 0.5819 / 0.5754，IoU median 为 0.6995 / 0.6896；Stage C 虽把 IoU P05 从 0.0355
提高到 0.0638，却使 none recall 从 0.9841 降到 0.9312，多目标拒绝率从 1.0 降到 0.9130。
因此 Stage C 没有发生全面灾难性遗忘，但总体泛化略退化。

## Stage B 混合 validation 指标

Stage B 最佳检查点的 validation 共 5,614 张。内容与 outer 指标如下：

| 指标 | 数值 |
| --- | ---: |
| content NCE median / P95 | 0.034168 / 0.150410 |
| content IoU median / P05 | 0.693840 / 0.028918 |
| content mask IoU | 0.770986 |
| boundary F1 | **0** |
| outer presence precision / recall | 0.990152 / 0.966716 |
| outer presence FPR / Brier / ECE | 0.003050 / 0.007771 / 0.005565 |
| outer NCE median / P95 | 0.034742 / 0.067854 |
| outer IoU median / P05 | 0.806193 / 0.629878 |
| no-candidate rate | 0.185964 |
| layer-ambiguous rate | 0.001425 |
| strict accepted precision / coverage proxy | 0.000222 / 0.000213 |

outer presence 契约修复有效，P1 的随机 outer 响应问题已被消除。outer quad 真值只来自 synthetic，
所以 outer 四角精度仍不能代表真实画框、卡纸或屏幕外框泛化。

类别混淆矩阵的行是真值、列是预测，顺序均为 artwork、postcard、screen、none：

```text
[[ 412,   51,   0,  30],
 [  11, 3565,   1,  94],
 [   0,    0, 520,   0],
 [   2,    9,   1, 918]]
```

recall：artwork 0.835700、postcard 0.971125、screen 1.000000、none 0.987097。artwork 是当前
最弱类别，主要混为 postcard 或 none。

## 分数据源冻结评估

以下结果对同一个 Stage B `best.pt` 做只读推理，未更新权重。每个数据源使用其完整 validation
split；MIDV 和 SmartDoc 的 group 数过少，只能作为域差诊断。

| 数据源 | 样本 / group | NCE median / P95 | IoU median / P05 | mask IoU | 关键分类 recall | strict precision / coverage |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| MIDV-Holo | 464 / 2 | 0.02366 / 0.05077 | 0.78164 / 0.50613 | 0.75211 | postcard 0.8804；none 1.0000 | 0 / 0 |
| MIDV-500 | 300 / 1 | 0.03493 / 0.08894 | 0.69128 / 0.37591 | 0.48990 | postcard 0.7875；none 0.9500 | 0 / 0 |
| SmartDoc | 2,450 / 3 | 0.02937 / 0.15871 | 0.66514 / 0.01714 | 0.81370 | postcard 1.0000 | 0 / 0 |
| synthetic | 2,400 / 2,400 | 0.04923 / 0.10843 | 0.70660 / 0.44663 | 0.77295 | artwork 0.8357；postcard 0.9717；screen 1.0000；none 0.9936 | 0.00065 / 0.00062 |

synthetic outer presence precision / recall 为 0.9902 / 0.9667，outer IoU median / P05 为
0.8062 / 0.6299。四个数据源的 boundary F1 均为 0，这是跨域共同故障，而非某一个数据集的
偶发问题。

## SmartDoc test release gate

| 阶段 | accepted | precision | coverage | NCE P95 | IoU median / P05 | wrong layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 250 | 0 | 0.148898 | 0.175810 | 0.663796 / 0.406866 | 0 |
| B | 253 | **0.347826** | **0.150685** | **0.135372** | **0.884393 / 0.538319** | 0 |
| C | 249 | 0.136546 | 0.148303 | 0.151837 | 0.725270 / 0.578228 | 0 |

三个阶段均为 **FAIL**。Stage B 的 accepted 253 张中，88 张满足 class、layer、NCE ≤0.01 与
IoU ≥0.93；165 张 class 和 layer 正确，但四角几何未达门槛。拒绝原因按多标签计数为：

- `boundary_uncertain`：1,280
- `invalid_quad`：146
- `out_of_scope`：138

模型对 SmartDoc 的类别和 content 层级均判断正确，主要瓶颈已经从 P1 的 outer 层级契约转移到
boundary 与角点尾部。SmartDoc 的 2 个 test group 仍远低于 100-group release 最低要求。

正式报告：

```text
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/smartdoc-test-e2e.json
```

## private validation / test 与人工复核

private validation 和 test 各只有 10 张、5 个独立 group，结论统一标记为
**insufficient evidence**。它们可以揭示具体错误，不能支持可靠性或发布结论。

| checkpoint | split | accepted | 单目标类别正确 | 多目标拒绝 |
| --- | --- | ---: | ---: | ---: |
| Stage B | validation | 3 / 10 | 8 / 8 | 2 / 2 |
| Stage B | test | 2 / 10 | 3 / 8 | 2 / 2 |
| Stage C | validation | 2 / 10 | 6 / 8 | 2 / 2 |
| Stage C | test | 4 / 10 | 4 / 8 | 2 / 2 |

人工复核确认，多目标同框样本在 Stage B 与 C 上均被拒绝，符合当前产品语义。单幅画作已有一部分
样本能给出接近内容层的四角；仍存在把局部内区当作完整画芯、boundary 支持不足、明信片被判
`none`、相似缩略图与 HD 决策不一致等问题。Stage C 没有稳定改善这些错误。

主要 contact sheet：

```text
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/overlays-public-validation/contact-sheet.jpg
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/overlays-private-validation/contact-sheet.jpg
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/overlays-private-test/contact-sheet.jpg
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-c/overlays-public-validation/contact-sheet.jpg
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-c/overlays-private-validation/contact-sheet.jpg
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-c/overlays-private-test/contact-sheet.jpg
```

## Stage D calibration

Stage D 冻结 Stage C 模型，只读 5,624 张 validation、2,411 个 group，明确记录
`test_split_read=false`。在最低 precision 0.99 下：

| 分量 | threshold | precision | recall | accepted |
| --- | ---: | ---: | ---: | ---: |
| content presence | 0.660000 | 0.990505 | 0.978261 | 4,634 |
| outer presence | 0.698572 | 0.990260 | 0.897059 | 1,232 |
| class | 0.996628 | 0.990005 | 0.786810 | 4,302 |
| corner | 1.0 | 0 | 0 | 0 |
| boundary | 1.0 | 0 | 0 | 0 |
| layer | 1.0 | 0 | 0 | 0 |
| combined | 1.0 | 0 | 0 | 0 |

presence、outer presence 和 class 能找到高 precision 阈值；几何分量无法找到任何满足条件的
非空集合，product proxy 最终为 accepted 0、precision 0、coverage 0。该文件是失败证据，不能
作为产品阈值配置：

```text
/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-d/calibration.json
```

## 失败分类与根因判断

1. **boundary 头没有形成可用信号。** 所有 validation 数据源的 boundary F1 都是 0，SmartDoc
   有 1,280 张因 `boundary_uncertain` 被拒绝。当前一像素边界监督及类别不平衡很可能使网络倾向
   全背景；需要先检查正负概率分布，再升级监督表示和 loss。
2. **角点尾部错误仍是高置信度误接受主因。** Stage B 已接受 253 张，但其中 165 张严格几何
   不正确；训练 loss 下降并未带动 NCE P95 和 IoU P05 改善。
3. **真实独立 group 严重不足。** SmartDoc test 2 group、MIDV validation 1～2 group、private
   validation/test 各 5 group，视频帧数量不能替代独立作品、设备、场地与拍摄 session。
4. **真实 outer 监督缺失。** outer presence 在合成集上表现良好，outer quad 也只在合成集有真值，
   真实画框、卡纸和屏幕边框的域差尚未被量化。
5. **Stage C 选模集过小且类别缺失。** 5 个 group 无法稳定选择 checkpoint，validation 不含
   postcard 和 screen，最终 Stage C 在 SmartDoc 与固定 public 子集上均退化。
6. **多目标拒绝逻辑方向正确，证据量不足。** private validation/test 各只有 1 个多目标 group；
   2 / 2 拒绝只能作为 smoke regression。

## 下一轮建议

下一轮应继续聚焦 geometry，按以下顺序推进：

1. 先修 boundary 监督与诊断。把细线标签升级为受控窄带或距离场，使用适合极端前景不平衡的
   Dice/Focal 组合；逐 epoch 记录边界正样本、负样本概率与多阈值 F1，并增加数值回归测试。
2. 将选模指标改为直接惩罚高置信度坏四角，单独跟踪 accepted 中 NCE >1% 或 IoU <0.93 的数量；
   在该指标改善前，不扩大模型宽度，也不单纯增加 epoch。
3. 建立至少 100 个独立真实 validation/test group，优先补 artwork、screen、真实 outer、卡纸、
   嵌套画框、不完整目标、相邻多画与复杂背景；按作品、设备、地点和 capture session 隔离。
4. private 达到足够规模前，把 Stage C 降为人工域适配诊断，不允许它覆盖通用 Stage B 候选。
   后续恢复 Stage C 时，需做类别分层的独立 validation，并保留固定 public regression gate。
5. 先由冻结候选通过原图分辨率 ONNX + 精修 validation gate，再执行 calibration。Stage D 应校准
   最终候选本身，且输出阈值必须通过独立 test 验收；不能把 Stage C 阈值套到 Stage B。
6. 持续保留 99% accepted precision、90% coverage、NCE P95 1%、IoU median/P05 0.97/0.93 和
   wrong layer 0.5% 的正式门槛。

Stage B 已证明 P2 契约修复与 mixed-domain 训练方向有效；下一轮的重点是 boundary 监督、真实
独立 group 与尾部错误控制。当前证据不支持继续盲目训练 width=1.5，也不支持开始恢复模型重训。

## 最终完整性检查

| 检查 | 结果 |
| --- | --- |
| checkpoint 加载与参数量 | PASS，99,632 |
| Stage B ONNX Runtime 会话与输出契约 | PASS，7 个预期输出 |
| 关键 run / calibration / e2e JSON 解析 | PASS |
| `XDG_STATE_HOME=/tmp/screenrestore-p2-final-state python -m pytest -q` | PASS，157 passed |
| `python -m ruff check .` | PASS |
| `python -m compileall -q src training benchmarks scripts tests` | PASS |
| `git diff --check` | PASS |

测试只有 PyTorch legacy ONNX exporter 的 2 条弃用警告，不影响本轮 ONNX 导出与运行时加载；后续
应在独立维护任务中迁移到 `torch.export` exporter。

## 复核命令

以下命令只读取训练产物并校验 JSON、checkpoint、ONNX 与文档，不会启动训练：

```bash
cd /Users/caozichen/Image_manage
source .venv/bin/activate
which python

python -m json.tool \
  /Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-d/calibration.json \
  >/dev/null

python -m json.tool \
  /Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/smartdoc-test-e2e.json \
  >/dev/null

shasum -a 256 \
  /Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/best.pt \
  /Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/quadlocator-s.onnx
```
