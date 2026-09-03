# P4 Geometry G1/G2/G3 统一比较

## 口径与范围

所有 SmartDoc 数值均来自 `decoder_v2` 的 **raw/coarse、all present samples**：1679/1679
候选先在仅照片阶段生成，之后才读取冻结标注计分。它们不等同于 refined 指标、final
proposal 或 accepted subset。B0 使用 P2 stage-b 的冻结 checkpoint；G1/G2/G3 共享同一
warm start、manifest、5000 train samples、1000 validation samples、seed=20260901、512
输入、batch=4、3 epochs 与 learning rate=1e-4。

> 关键修正（P4 G3.5）：后续 augmentation 分量对照发现，训练数据集在增强前过早构造了
> `image_tensor`。因此 historical `geometric`/`full` augmentation 更新了 quad 标签，却没有把
> 增强后的图像送入模型；`photometric` augmentation 也完全没有作用。该错位已在
> `QuadDataset.__getitem__` 修复，并有 image/quad 同步回归测试。G1/G2/G3 与此前 R1 的
> “训练后退化”仍是事实，但不能再被解释为多任务、学习率或 augmentation 质量的独立证据；它们
> 均受错位监督污染。B0 evaluator parity 与所有冻结 checkpoint 的推理审计不受此 bug 影响。

G1 的早期运行在本轮平衡式 `best_geometry` 选模策略落地前完成。为避免使用其 IoU P05
已经归零的 epoch 2 checkpoint，本文将 epoch 1 的 `best_product.pt` 作为 G1 的最优可比较
checkpoint；它已独立完成 SmartDoc audit。G2/G3 均使用当前 `best_geometry.pt`（epoch 2）。

## Validation：固定 1000 样本

| Experiment | checkpoint | NCE med | NCE P95 | IoU med | IoU P05 | strict | heatmap conf. | mask IoU | boundary AUPRC | boundary best F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 warm start | epoch 0 | 0.03029 | 0.15404 | 0.74608 | 0.15254 | 0.00736 | 0.90767 | 0.77515 | 0.25120 | 0.38182 |
| G1 content-only | epoch 1 | 0.03394 | 0.15505 | 0.67673 | 0.08837 | 0.01104 | 0.89479 | 0.78004 | 0.25570 | 0.38758 |
| G2 content+mask | epoch 2 | 0.03276 | 0.16103 | 0.71978 | 0.20150 | 0.00613 | 0.89433 | 0.78785 | 0.25585 | 0.37968 |
| G3 content+boundary | epoch 2 | 0.03436 | 0.16499 | 0.69663 | 0.12815 | 0.00982 | 0.89049 | 0.60327 | 0.25925 | 0.36901 |

G2 相对 epoch 0：NCE median `+0.00247`、NCE P95 `+0.00699`、IoU median `-0.02631`、IoU
P05 `+0.04896`、strict `-0.00123`、mask IoU `+0.01270`。mask 改善了一个 validation tail
指标，却没有让总体 corner geometry 超过 B0。

## SmartDoc test：raw/coarse、all present samples

| Experiment | checkpoint | NCE med | NCE P95 | IoU med | IoU P05 | strict | candidates | score ambiguous | refinement accepted | final accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | frozen P2 stage-b | 0.11699 | 0.31836 | 0.68598 | 0.21309 | 0.00000 | 1679 | 1608 | 0.00000 | 0 |
| G1 | epoch 1 / best_product | 0.13776 | 0.32178 | 0.55645 | 0.27907 | 0.00000 | 1679 | 1606 | 0.00000 | 0 |
| G2 | epoch 2 / best_geometry | 0.12736 | 0.32090 | 0.61713 | 0.28952 | 0.00000 | 1679 | 1606 | 0.00000 | 0 |
| G3 | epoch 2 / best_geometry | 0.12984 | 0.32104 | 0.57154 | 0.31052 | 0.00000 | 1679 | 1566 | 0.00000 | 0 |

G2 相对 B0：NCE median `+0.01037`、NCE P95 `+0.00254`、IoU median `-0.06886`、IoU P05
`+0.07642`。因此它不满足“raw NCE median 不高于 B0 且 raw IoU median 不低于 B0”的继续条件。
G3 虽将 `score_ambiguous` 从 1608 降至 1566，但仍未转化为任何 final accepted sample，且 raw
NCE median 与 raw IoU median 都继续劣于 B0。

## Training dynamics

| Experiment | epoch | train loss | val loss | NCE med | NCE P95 | IoU med | IoU P05 | strict | LR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | 1 | 0.19456 | 0.11340 | 0.03394 | 0.15505 | 0.67673 | 0.08837 | 0.01104 | 7.5e-5 |
| G1 | 2 | 0.19046 | 0.11715 | 0.03865 | 0.15057 | 0.65218 | 0.00000 | 0.00736 | 2.5e-5 |
| G1 | 3 | 0.19139 | 0.12048 | 0.04073 | 0.15466 | 0.63315 | 0.00000 | 0.00613 | 0 |
| G2 | 1 | 0.82114 | 0.44042 | 0.03279 | 0.15881 | 0.69296 | 0.17993 | 0.01227 | 7.5e-5 |
| G2 | 2 | 0.80127 | 0.45488 | 0.03276 | 0.16103 | 0.71978 | 0.20150 | 0.00613 | 2.5e-5 |
| G2 | 3 | 0.80438 | 0.46685 | 0.03593 | 0.15719 | 0.68572 | 0.12376 | 0.00982 | 0 |
| G3 | 1 | 1.08706 | 0.85847 | 0.03259 | 0.16944 | 0.70417 | 0.00000 | 0.00982 | 7.5e-5 |
| G3 | 2 | 0.99913 | 0.85654 | 0.03436 | 0.16499 | 0.69663 | 0.12815 | 0.00982 | 2.5e-5 |
| G3 | 3 | 1.00376 | 0.86344 | 0.03751 | 0.16406 | 0.65370 | 0.00000 | 0.00613 | 0 |

G1/G3 从第一个 epoch 起已在 NCE P95 或 IoU median 上回退，且随后出现明显 tail collapse；G2
的 IoU P05 在 epoch 1/2 提升，但 NCE P95、NCE median 与 IoU median 在第一轮即没有共同改善。
现有证据指向“当前 recipe 一发生梯度更新就容易损伤 raw geometry”，尚不能区分 content loss、
trainable scope、learning rate、sampler、augmentation 或坐标/decoder 训练语义的具体责任。

## 决策

- G4：**NO**。没有一个 auxiliary 策略同时满足 validation 与 SmartDoc raw geometry 的 B0 比较条件。
- geometry FULL：**BLOCKED**。
- 多任务冲突：证据不足以确认。G1 仅排除了“删除其它 task loss 后当前 recipe 会自动恢复”的假设；
  G2/G3 结果说明 mask/boundary 在当前 recipe 下也未构成有效修复，尚未记录 shared-gradient cosine。
- G3.5 root-cause ablation：**YES**。首先运行 R1（content head only）与 learning-rate matrix，
  冻结所有 backbone/FPN/非 content head 的参数及 BatchNorm running stats；仅在 R1 稳定后再进入 R2。

## G3.5 R0/R1：head-only learning-rate matrix

R0 冻结 replay 与 B0 一致，Torch MPS/ONNX CPU 最大绝对误差为 `1.89e-4`。R1 只训练
`content_corner_head` 的 5188 个参数，实际参与损失仅为 `content_heatmap` 与
`content_corner_geometry`；encoder、FPN、其他 heads 及其 BatchNorm running stats 都被冻结。

| R1 LR | validation NCE med delta | NCE P95 delta | IoU med delta | IoU P05 delta | SmartDoc NCE med | SmartDoc IoU med |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-5 | +0.00183 | -0.00028 | -0.01923 | +0.00721 | 0.12838 | 0.65716 |
| 3e-5 | +0.00283 | +0.00540 | -0.03029 | -0.03865 | 0.13308 | 0.63655 |
| 1e-4 | +0.00452 | +0.00636 | -0.04940 | -0.11967 | 0.13539 | 0.61627 |

学习率明确影响退化幅度：`1e-4` 最差，`1e-5` 最接近 B0。但 R1 即使在 `1e-5` 仍让 SmartDoc
raw NCE median 从 B0 的 `0.11699` 升至 `0.12838`，IoU median 从 `0.68598` 降至 `0.65716`。
因此 R1 不满足稳定条件，暂停 R2；下一步是关闭增强的固定 64 样本 overfit，检查 content loss、
标签、坐标变换和 decoder 训练语义。

## G3.5 augmentation control

在同一随机 64 样本、content heatmap-only、content head-only、40 epochs、1e-4 的严格对照中，
仅将训练增强从 `none` 改为历史默认 `full`。两次训练与 SmartDoc test 均不存在 image、group_id 或
capture_session 重叠。

| train augmentation | SmartDoc NCE med | NCE P95 | IoU med | IoU P05 |
|---|---:|---:|---:|---:|
| none | 0.06487 | 0.31941 | 0.81065 | 0.36223 |
| full | 0.12446 | 0.31553 | 0.67134 | 0.23105 |

`full` augmentation 使 IoU median 从 `0.81065` 降至 `0.67134`，低于冻结 B0 的 `0.68598`。增强是
当前最强根因候选，但尚需将同步 homography 与 photometric 退化分别复现，不能把两者混为同一原因。

## 修复图像/标签错位后的 aligned R1

修复后，full augmentation 同时进入模型输入与四角标签。以 5000 train / 1000 validation、只训练
content head、1 epoch 重跑 R1：

| LR | SmartDoc NCE med | NCE P95 | IoU med | IoU P05 |
|---:|---:|---:|---:|---:|
| B0 | 0.11699 | 0.31836 | 0.68598 | 0.21309 |
| 1e-5 | 0.11506 | 0.32027 | 0.68768 | 0.21961 |
| 3e-5 | 0.12324 | 0.33822 | 0.68704 | 0.17184 |
| 1e-4 | 0.09888 | 0.35009 | 0.72718 | 0.28843 |

`1e-4` 的 raw median 与 IoU P05 改善最明显，但 NCE P95 从 `0.31836` 升为 `0.35009`，因此它仅可
进入带 geometry 选模、tail-collapse watchdog 与早停的受控长程 recovery，仍不满足 geometry FULL
或发布级 gate。

## 受控长程 recovery：tail collapse 复现

`p4-geometry-recovery-night-20260902-104344` 使用 20000 train / 1000 validation、content
head-only、content-only loss、正确同步的 full augmentation、1e-4。watchdog 在第 4 epoch 停止：
validation NCE P95 `0.22493` 超过 epoch 0 的 `1.30×` 上限 `0.20025`，且 IoU P05 `0.03408`
低于 `0.65×` 下限 `0.09915`，连续两轮触发。

SmartDoc best checkpoint 的 raw/coarse NCE median 从 B0 的 `0.11699` 降至 `0.03680`，IoU median
从 `0.68598` 升至 `0.87808`；但 NCE P95 为 `0.34096`、IoU P05 为 `0.19132`，均未同时优于 B0。
refined attempt 的 IoU median 为 `0.93736`，却只产生 1 个 final accepted 样本（coverage `0.00060`）。

结论：本轮证明对齐修复后 content head 可显著改善 median geometry，但 1e-4 长程更新仍会伤害 tail。
当前 `best_geometry.pt` 与 `best_product.pt` 都是 watchdog 判定的 epoch 4，不可作为后续 warm start。
下一步必须将 best-geometry 选择加入相对 epoch 0 的 tail eligibility，随后用低学习率进行短程
tail-preservation recovery；geometry FULL 继续 BLOCKED。

## Coordinate-only 受控长训：16 epochs

`p4-geometry-coordinate-recovery-20260902-210628` 使用修复后的同步 `full` augmentation、
20000 train / 1000 validation、`content_coordinate_only`、content head-only（5188 个可训练参数）和
`1e-5`。训练完整跑完 16 个 epoch，未触发 geometry tail watchdog；几何与产品选模均选择 epoch 14。

本轮自身 validation 的 epoch 0 到 epoch 14，NCE P95 从 `0.23674` 降至 `0.18431`，IoU P05 从
`0.26500` 升至 `0.41862`。这说明低学习率下的 coordinate-only 更新可以稳定改善该验证切片的 tail，
没有复现 1e-4 的 tail collapse。

SmartDoc 仍按 decoder v2 的 raw/coarse、1679 个 present sample 与冻结 B0 比较：

| checkpoint | NCE med | NCE P95 | IoU med | IoU P05 | refined NCE P95 | refined IoU P05 | final accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 frozen P2 | 0.11699 | 0.31836 | 0.68598 | 0.21309 | — | — | 0 |
| coordinate-only epoch 14 | 0.13064 | 0.26284 | 0.67455 | 0.45448 | 0.22507 | 0.51939 | 0 |

相对 B0，长训将 NCE P95 降低 `0.05552`、IoU P05 提高 `0.24138`，但 NCE median 增加
`0.01365`、IoU median 降低 `0.01144`。这是一项明确的 tail 改善与典型样本轻微退化的交换，不能
描述为总体 raw geometry 胜过 B0。短训 epoch 1 曾四项 raw/coarse 同时优于 B0，因此当前证据更支持
“存在较短有效训练窗口；16 epoch 的 validation-tail 选模没有保住 SmartDoc median 最优点”，而非直接
扩大训练时长。

最终自动接受仍为 0：本轮只更新 content corner head，冻结的 presence、class、boundary、mask 与
接受策略没有获得对应改善；decoder v2 的 1676/1679 个样本仍被 `score_ambiguous` 拒绝。并且该
SmartDoc 集只有 2 个独立 group，不能作为 release 级精度证据。

### 决策更新

- geometry FULL：**NO / BLOCKED**。release gate 的 accepted precision、in-scope coverage、NCE P95、
  IoU median 和 IoU P05 均未通过，最终接受覆盖率仍为 0。
- coordinate-only：保留为有效的 geometry-tail recovery 方向，但不再直接按 16 epoch 扩大规模。
- 下一步：先做保留 epoch checkpoint 的短程轨迹评估（例如 epoch 1/2/4/8），在同一 SmartDoc raw/coarse
  审计中找出 median 与 tail 可同时改善的 checkpoint；确定几何 checkpoint 后，再单独处理
  presence/class/boundary/acceptance 的产品链路。
