# P4-G3.7 冻结协议

阶段：Data Isolation + Native-512 Geometry Sanity + Acceptance Re-audit。
代码起点：`90cadb3988679b4772fec1de4a9f4d3a3d422e87`，已核对 origin/main。
执行时额外记录工作树代码摘要、命令和每个输入文件 SHA256。

## Incumbent 与 metric-v2

B0：`$SCREENRESTORE_RUN_ROOT/p2-geometry-w1-20260829-110658/stage-b/best.pt`。
实测 SHA256：`3344ad62709abf9e413a1cdebbdb82b1c0add0655ffdaeadcd9ddcc6bac86746`。
checkpoint epoch=12，width_multiplier=1，原生输入 512。

`metric_version = 2`；`nce_normalization = target_quad_bbox_diagonal`。
NCE 为对应四角欧氏距离均值除以目标 quad 包围盒对角线。冻结 evaluator 使用原图像素坐标；
训练 evaluator 使用等比例 letterbox 坐标。归一化标注先乘 `(W-1,H-1)` 转到原图。
旧整图对角线日志不参与 eligibility。所有比较重新运行 B0 与共同 512 evaluator。
候选几何分母与全体正例 strict 分母分别记录，拒绝样本不得丢失。

## 数据角色与读取边界

- P2 stage-b train：只用于既有 recipe 复现；全量用于元数据隔离审计。
- internal validation：固定 seed=20260902 的 1000 样本；历史开发诊断。
- P2 calibration 与 SmartDoc validation：开发诊断，显式报告相互重叠，不能当作独立证据。
- 新公开 target-domain validation：仅已授权真实照片及人工标注与审核；按 source-family group 固定 hash
  划分 fit/selection/evaluation（60/20/20，复用已有 SHA256 前八位模十协议）。
- SmartDoc test：本阶段不读取图片、GT 或历史 test 结果，不执行 test。
  仅允许消费事先生成的、完全不含 GT/图片路径的身份 hash 索引作集合审计；缺少索引时记为
  `test_isolation_unverified`，不把未检查解释为独立。混合清单仅投影 split，跳过 test 行。
- 允许读取上述开发集照片、标注、checkpoint、run/history 与 validation audit。
  禁止下载；数据根硬上限 30 GiB；清单、overlay、模型和 run 写到仓库外。

## Native-512 的唯一变量与停止条件

唯一核心变化为 train image_size 256 → 512。B0、数据清单、样本选取、seed、augmentation、
loss、trainable scope、AdamW、weight_decay、cosine horizon 与 effective batch 必须一致。
目标 recipe 为 content_coordinate_only / content_head / aligned full augmentation / LR=1e-6，
仅一轮，不做 sweep 或长训。物理 batch 如需缩小，必须显式记录 accumulation 和末批语义。

实际最近 1e-6 run 是 content_only / none / 5000 / seed=20260902 / batch=16 / horizon=4。
现有 coordinate-only/full run 使用 1e-5 或 2e-6，尚无满足目标 recipe 的 256 对照。
因此严格实验 preflight 必须拒绝不可比 arm；不得悄悄同时改 loss 与增强后归因于输入尺寸。
若操作者明确选择多变量诊断，则单独修订协议，结果不得声称是分辨率因果验证。

先运行 B0 parity：相同照片、512 输入，Torch/ONNX raw 最大绝对误差 ≤0.001，
decoded corner 最大归一化偏差 ≤0.001；训练/冻结指标 median NCE 偏差 ≤0.002、
P95 NCE ≤0.005、IoU median/P05 ≤0.01。比较须说明 clipping/order 与无候选处理差异。
parity 失败立即阻止训练，先修 evaluator/decoder/坐标语义。

Geometry eligibility 复用 trajectory：三个 slice 四项均满足相对新 B0 的绝对容差
NCE median +0.002、P95 +0.005、IoU median -0.01、P05 -0.01。
无合格 challenger 输出 `NO_ELIGIBLE_CHECKPOINT`，保留 B0，停止 coordinate-only 搜索与 FULL。
通过则冻结 SHA256，只进入独立 target-domain validation，不宣称 release 完成。

## Acceptance eligibility

目标 48 subjects，postcard/artwork/screen/poster 各 12；poster 映射 artwork 并保留场景类别。
每 subject ≥2 sessions，每 session 6–10 positives，额外约 25% hard negatives。
partial/multi-target/曲面/严重遮挡/极端视角必须拒绝。
启用要求：四类各 ≥8 独立 groups，source/subject/group/session 完整，人工审核 overlay，
source family 固定 group，无内部泄漏，与既有 calibration/SmartDoc validation/test 无交集。
缺项统一 `BLOCKED: insufficient_target_domain_groups`，同时列出具体原因。

fit 仅拟合与 normalization；selection 仅选 threshold/policy 与预注册消融；evaluation
仅对已选 policy 最终评估一次，不回调。比较 current hard gate、去 margin、去 boundary、
combined calibrated evidence、boundary/refinement evidence。消融在 selection 比较信息贡献。
全体 evaluation 样本作为 policy 分母；报告 TP/FP/FN/TN、precision、recall、in-scope coverage。
完整 gate：precision ≥99%、coverage ≥90%、wrong-layer <0.5%、NCE P95 ≤1%、
IoU median ≥0.97、P05 ≥0.93，并满足现有独立 group 发布要求。
selection 成功而 evaluation 失败标记 calibration failure，保持 runtime fail-closed。

## 运行时边界

本阶段默认仅生成研究 evidence。未通过独立完整 gate 的 checkpoint、calibrator 和阈值
不得进入 runtime。即使通过，也仅提出 G4 candidate，发布声明需另行审阅完整证据。
Fidelity/Photometric/Super-resolution/Router 保持冻结；真实 demoire/reflection 无合规 paired
数据保持 BLOCKED。本阶段不提供 all/test，不扩大训练预算。

## 训练前 evaluator 修复记录

B0 的首次 parity 检查发现：训练验证按 heatmap channel 对应，冻结 benchmark 按循环/反向最优
对应；部署另有 letterbox padding 裁剪。已将训练验证复用冻结 corner_metrics 的匹配方式，
并传入 image_bounds 同步裁剪。NCE 分母仍为 target_quad_bbox_diagonal；这项修复只改变验证
统计，不改变 loss、augmentation、模型或 runtime。修复前的训练侧 tail 不可与修复后直接比较。
三 slice 各固定 32 张的复查通过后，方可解除 evaluator 阻塞。
