# P4-G3.7 结果

## 决策

保持 P2 B0 与当前 runtime fail-closed policy。目标域 readiness 为
`BLOCKED: insufficient_target_domain_groups`，属于本阶段决策 C。
指定的 native-512 单变量实验另受 `missing_comparable_256_arm` 阻塞，未训练新 challenger，
因此不能把本阶段记为“512 challenger 失败”或 `NO_ELIGIBLE_CHECKPOINT`。

代码起点为 `90cadb3988679b4772fec1de4a9f4d3a3d422e87`，已核对 main 与 origin/main。
本轮未提交 commit。B0 SHA256 经实际 checkpoint 验证为
`3344ad62709abf9e413a1cdebbdb82b1c0add0655ffdaeadcd9ddcc6bac86746`，epoch=12、原生 512。

## G3.6 与既有 native-512 证据

已读取 G3.6 run.json、history、validation trajectory 与 acceptance audit，确认历史三组
validation 重叠，geometry/acceptance 均不支持替换 B0。旧 trajectory format v1 使用各自
checkpoint 输入尺寸，不作为共同 512 eligibility 的权威证据。

最近的 1e-6 对照实际为 content_only/content_head/none augmentation、5000 train、1000 val、
batch=16、seed=20260902、cosine horizon=4。已有 native-512 单轮也采用该 recipe；
checkpoint SHA256 为 `edc31d7a668ce3199adcdd7b33050928da9f75fd475a6253cf47927ab7bcbab2`。
其共同 512 trajectory format v2 状态确为 `NO_ELIGIBLE_CHECKPOINT`：internal IoU median
下降 0.01384，SmartDoc validation IoU median 下降 0.01463，超过 0.01 容差。

这组旧实验仅支持 content_only/none 配方的观察。它不能替代本阶段要求的
content_coordinate_only/full augmentation 单变量实验。实际 coordinate-only/full 历史 arm
使用 1e-5 或 2e-6，尚无符合要求的 1e-6、256 arm。本轮停止搜索，保留这些差异为明确 blocker。

## metric-v2 与 evaluator 修复

`metric_version = 2`

`nce_normalization = target_quad_bbox_diagonal`

NCE 为匹配后四角平均像素误差除以目标 quad 包围盒对角线。归一化标注先映射到原图像素坐标。
旧训练日志采用整图对角线时的数字，不用于当前 eligibility。

首次 B0 parity 发现新的尾部统计差异：训练验证按 heatmap channel 直接对应，且未裁掉
letterbox padding；冻结 benchmark 使用循环/反向最优对应，并将角点裁回原图。
首次检查三 slice NCE P95 差分别为 0.12715、0.09030、0.26333，训练被门控阻止。

已修复训练验证：复用权威 corner_metrics 的匹配与 IoU，实现 image_bounds 裁剪。
训练 loss、增强、网络和 runtime 保持原行为。修复前的训练-side tail 也不可与修复后直接比较。

修复后每 slice 固定 32 张进行同输入 parity，结果如下；它是有界数值检查，不等于全数据逐项 parity：

| slice | NCE median 差 | NCE P95 差 | IoU median 差 | IoU P05 差 |
|---|---:|---:|---:|---:|
| internal validation | 0.00001521 | 0.00000719 | 0.00000009 | 0.00000004 |
| calibration | 0.00001853 | 0.00011457 | 0.00000003 | 0.00000016 |
| SmartDoc validation | 0.00000272 | 0.00000822 | 0.00000017 | 0.00000009 |

全部通过预注册容差。Torch MPS/ONNX CPU raw 最大绝对差 0.00016785，decoded corner 最大
归一化差 0.00000018；预处理输入完全一致。该检查只使用开发照片。

## 新 B0 metric-v2 baseline

所有照片在 512 输入完成预测后，再按对应 validation GT 评分。以下为 raw/coarse、正例候选；
strict 为 NCE ≤0.01 且 IoU ≥0.93 的候选比例。runtime acceptance 使用完整 slice 分母。

| slice | samples | groups | candidates | NCE median | NCE P95 | IoU median | IoU P05 | strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| internal validation | 1000 | 436 | 842 | 0.08925764 | 0.28627879 | 0.75816710 | 0.32773528 | 0.00118765 |
| calibration | 5624 | 2411 | 4692 | 0.09105009 | 0.29213908 | 0.74828795 | 0.30091166 | 0.00042626 |
| SmartDoc validation | 2450 | 3 | 2450 | 0.12597178 | 0.33261892 | 0.66860528 | 0.18027871 | 0 |

| slice | refinement improve | worsen | fail | refinement accepted | runtime accepted |
|---|---:|---:|---:|---:|---:|
| internal validation | 475 | 145 | 222 | 9 | 0 |
| calibration | 2662 | 767 | 1263 | 83 | 0 |
| SmartDoc validation | 912 | 368 | 1170 | 0 | 0 |

完整 coarse/refined-attempt/rollback/final 指标、拒绝原因分布和 policy precision/recall/coverage/FP/FN
保存在每个 slice 的 JSON。全部接受数为零，不能宣称 acceptance 已解决。

## 数据隔离与 target readiness

实际图片 SHA256 确认 internal/calibration 重叠 1000 张，internal/SmartDoc validation
重叠 430 张，calibration/SmartDoc validation 重叠 2450 张。所有既有清单缺失 digital_source_id
和 subject_id，source-family 独立性仍未证明，详见 `docs/P4_DATA_ISOLATION_AUDIT.md`。

target-domain 私有目录和人工标注尚不存在，四类独立 group 均为 0，fit/selection/evaluation
未启用。没有预先生成的无 GT test 身份索引，test isolation 记为 unverified。本轮没有打开
SmartDoc test 图片、GT 或历史 test run，不使用它作阈值或模型决策。

已实现元数据/SHA256 重叠审计、source/group/session 泄漏检测、确定性 group 分区、
overlay 与角点编号、pending/approved/rejected 人工审核字段、照片/标注 SHA256 审核绑定、
审核清单合并与 insufficient-group gate。数据保持本地，30 GiB 上限检查通过。

## Acceptance 与 restoration

acceptance-dev preflight 实际返回 `BLOCKED: insufficient_target_domain_groups`。
已实现 fit 拟合、selection 消融/选择、选定策略冻结后单次 evaluation 的独立流程。
比较 hard gate、去绝对 margin veto、去 boundary veto、combined evidence、boundary/refinement
模型及特征消融。evaluation 使用完整分母，source-family 单次锁阻止改输出目录后重复评估。
代码通过合成单元测试；正式 calibration、evaluation 与真实域效果均未执行。

完整 release gate 仍要求 precision ≥99%、coverage ≥90%、wrong-layer <0.5%、NCE P95 ≤1%、
IoU median ≥0.97、P05 ≥0.93，且独立 evaluation groups ≥100。首批 48 subjects 是 development
数据计划，不能替代发布级测试规模。运行时置信度策略未修改。

Fidelity/Photometric/Super-resolution/Router 保持冻结。真实 demoire/reflection 仍缺合规 paired
数据，保持 BLOCKED；本轮未启动 restoration 训练或外部下载。

## 产物与下一阶段

权威修复后结果：`$SCREENRESTORE_RUN_ROOT/p4-g37-metric-v2`。
首次 parity 失败记录保留在 `$SCREENRESTORE_RUN_ROOT/p4-g37`，不覆盖历史产物。
每阶段 provenance 保存命令、commit、代码摘要、B0 SHA256、manifest SHA256 和 group 数。

Go：按采集协议建立真实独立 target-domain 数据，补齐来源身份与人工审核。
No-go：geometry FULL、继续 LR/epoch 搜索、正式 acceptance 部署及 G4 发布声明。
单变量 512 实验需先解决缺少可比 256 arm 的协议矛盾，不自动改为多变量试验。
复制执行命令、缺失数据位置与下一步见 `docs/P4_NEXT_COMMANDS.md`。

## 验证

已在项目 Python 3.11 `.venv` 中执行：

```bash
python -m compileall -q training scripts tests src
python -m ruff check .
XDG_STATE_HOME=/tmp/screenrestore-p4-g37-state python -m pytest -q
bash -n scripts/run_p4_g37.sh
```

全部通过：238 tests passed；8 条为现有 TorchScript ONNX 导出的弃用提示。
新增回归覆盖源身份/字节/路径重叠、group/source/session 泄漏、确定性分区、
真实 group 门、审核失效、NCE 坐标与排列/裁剪、完整 policy 分母、单次 evaluation、
B0 fallback、512 recipe 重现与 report 生成。工作区原有 G3.6 未提交记录已保留。
