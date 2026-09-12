# P4-G3.6 Acceptance Feature Audit

## 结论

P4-G3.6 没有找到可替代 B0 的 coordinate-only checkpoint。epoch 12/14/16 位于 Pareto front，
但所有 milestone 都未同时守住 B0 的 NCE median 与 IoU median；本轮冻结的 incumbent 仍为 P2 B0：

- checkpoint：`/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/best.pt`
- SHA256：`3344ad62709abf9e413a1cdebbdb82b1c0add0655ffdaeadcd9ddcc6bac86746`
- coordinate challenger winner：无，状态为 `NO_ELIGIBLE_CHECKPOINT`

acceptance 证据也不支持修改正式 `ConfidencePolicy`。`candidate_margin` 不适合作为独立硬门；当前
calibrated evidence 在 selection 上得到的 100% precision 未泛化到隔离 evaluation。继续保持运行时
fail-closed，不降低 0.06，不启用校准器，也不做 confidence-shaping 微调。

## 数据与隔离

trajectory 选择只使用 validation 数据：固定 internal 1000 张/436 groups、P2 calibration 5624 张/
2411 groups、SmartDoc validation 2450 张/3 groups。SmartDoc test 未参与 checkpoint、阈值、校准器或
超参数选择。

三个 slice 并非独立数据源：internal 与 calibration 重叠 1000 张，internal 与 SmartDoc validation
重叠 430 张，calibration 包含全部 2450 张 SmartDoc validation。报告保留各 slice，避免把重叠样本
误作额外证据。SmartDoc validation 只有 3 个独立 group，状态为
`insufficient_target_domain_groups`；P4 target-domain protocol 已追加到
`docs/P3_DATA_CAPTURE_GUIDE.md`。

acceptance calibration 按 `group_id` 确定性分成：

| partition | samples | groups | strict correct |
|---|---:|---:|---:|
| fit | 3838 | 1432 | 26 |
| selection | 501 | 500 | 21 |
| evaluation | 1285 | 479 | 14 |

strict target 要求 target class 正确、content-layer quad、NCE ≤0.01 且 IoU ≥0.93。无目标、无候选及
错误语义均为负例。policy 比较统一使用完整的 1285 条 evaluation 分母。

## Candidate margin

`candidate_margin` 是四个 content corner 中最弱角的 `peak1 - peak2`。0.06 来自初始提交
`c5483ae5e8d8360f4047a38c47a71af262acda24`（2026-08-23）；原注释明确称其为阻止“有候选就采用”
的保守默认值，并要求未来在独立真实验证集校准。仓库没有找到支持 0.06 的独立 calibration 记录。

本轮 5342 个完整候选的 margin 分布为：median 0.002429、P95 0.010665、maximum 0.045784。
因此 0.06 位于全部观测值之外：accepted=0、precision=0、recall=0、coverage=0、FN=61、FP=0。

margin 与 correctness 的 Pearson 为 0.0680，Spearman 为 0.0818，average precision 为 0.0244。
threshold sweep 中最高 precision 仅 0.0355，对应 threshold 0.007693、coverage 0.0591、recall
0.3115、FP=516、FN=42。没有任何 margin threshold 同时形成合理 precision 与 coverage。

这 5342 个 `score_ambiguous` 的直接原因是输出 heatmap 的 peak difference 标度整体远低于未经校准
的 0.06。它不能解释为 5342 个 geometry 全部错误：其中存在 61 个 strict-correct 样本，同样全部
被 0.06 拒绝。

## 其它 heatmap 与 refinement evidence

单变量关联仍然很弱：

| feature | Pearson | Spearman | AP |
|---|---:|---:|---:|
| candidate margin | 0.0680 | 0.0818 | 0.0244 |
| peak ratio min | 0.0376 | 0.0695 | 0.0198 |
| entropy mean | 0.0447 | 0.0465 | 0.0156 |
| entropy max | 0.0458 | 0.0532 | 0.0155 |
| sharpness min | -0.0637 | -0.0873 | 0.0071 |
| boundary support | 0.1415 | 0.1416 | 0.0479 |
| continuous coverage | 0.0899 | 0.1008 | 0.0230 |
| normal alignment | 0.0886 | 0.1507 | 0.0646 |
| mask consistency | 0.1173 | 0.1268 | 0.0621 |

peak ratio 没有比 absolute peak difference 更稳定；移除 ratio 后 evaluation AP 从 0.2680 小幅升至
0.2743。移除 margin 后 AP 降至 0.2575，但最终 policy 结果相同，说明 margin 只能作为弱联合证据，
无法单独承担硬门。移除 entropy/sharpness 后 AP 为 0.2463，提供少量联合信息，仍不足以泛化。

boundary/refinement evidence 的信息量高于单个 heatmap shape feature。移除全部 boundary/refinement
特征后，selection 上不存在满足 99% precision 的非零 coverage 阈值，evaluation AP 从 0.2680 降至
0.1127。但这仍不足以证明当前 boundary hard gate 已成熟。

## Geometry 与 boundary audit

| stage | candidates | NCE median | NCE P95 | IoU median | IoU P05 | strict geometry |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 4692 | 0.09105 | 0.29214 | 0.74829 | 0.30091 | 0.00043 |
| refined attempt | 3429 | 0.05603 | 0.17631 | 0.84215 | 0.52842 | 0.10936 |
| rollback/final | 4692 | 0.09105 | 0.29214 | 0.74829 | 0.30091 | 0.01343 |

在 4692 个正样本候选中，refined attempt 同时改善 NCE/IoU 的有 2662 个，恶化 767 个，1263 个
无法拟合。按来源分别为：SmartDoc 912/368/1170、synthetic 1297/231/86、MIDV-Holo
345/114/1、MIDV500 104/50/6、private 4/4/0（顺序均为改善/恶化/无法拟合）。

当前 refinement/boundary hard gate 只通过 83/5624（1.48%）。它确实包含独立风险信息，也阻止了
大量错误放行；然而它的 coverage 极低，而且“尝试能改善”远多于“hard gate 接受”，说明现行门槛
过度保守。精修失败时 rollback 保留 coarse，几何输出没有被破坏；被影响的是 automatic acceptance。

运行时暂时保留 boundary hard gate 作为 fail-closed 安全措施，但不把它视为长期接受策略。只有新的
独立 target-domain validation 证明 evidence policy 达到 ≥99% precision 后，才考虑取消单项否决。

## Policy comparison

| policy（隔离 evaluation） | accepted | precision | recall | coverage | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| current hard gate | 0 | 0 | 0 | 0 | 0 | 14 |
| hard gate，移除 boundary | 0 | 0 | 0 | 0 | 0 | 14 |
| hard gate，移除 margin | 11 | 0.9091 | 0.7143 | 0.00970 | 1 | 4 |
| hard gate，移除 margin 与 boundary | 842 | 0.0119 | 0.7143 | 0.74162 | 832 | 4 |
| calibrated evidence | 1 | 0 | 0 | 0.00088 | 1 | 14 |

calibrated evidence 在 selection 上以 threshold 0.077714 接受 3 条，precision=1.0、coverage=0.00904；
在未参与阈值选择的 evaluation 上仅接受 1 条且为 FP。因此它没有在相同或更高 precision 下获得更合理
coverage，不允许进入 runtime。当前 `score_ambiguous` 大量出现主要由未校准的 0.06 绝对阈值造成，
而无法安全放宽的根因是整个 evidence stack 尚不能稳定识别极少数 strict-correct 样本。

## SmartDoc test 单次冻结评估

在 checkpoint 与第一版 validation calibrator 冻结后，本轮只执行了一次 SmartDoc test：1679 张、
2 个独立 group，且没有进行拟合或选择。该次运行发现新审计脚本错误地把 `[0,1]` 坐标传给只接受
像素坐标的 `corner_metrics`，因此该运行的 NCE 与 strict target 无效；IoU、refinement、rejection 和
acceptance 不受该 NCE 分母错误影响。问题已增加回归测试并修复，依照单次测试约束没有重跑 test。

可信的单次 test 字段为：coarse IoU median 0.68598、IoU P05 0.21309；refined attempt 826 个，IoU
median 0.76929、IoU P05 0.52784；rollback/final 回到 coarse；current hard gate 接受 0；第一版
calibrated policy 接受 1 条。由于该条的 strict target 来自错误 NCE 口径，不能声明 test precision。

同一 B0 checkpoint 的既有 decoder-v2 审计提供历史交叉检查：coarse NCE median 0.11699、NCE P95
0.31836、IoU median 0.68598、IoU P05 0.21309；refined attempt NCE median 0.08419、NCE P95
0.19285、IoU median 0.76883、IoU P05 0.52670；final accepted=0。该历史结果未参与本轮任何选择。

## 决策

- coordinate-only 同时改善 median + tail：否。它稳定改善 tail，却明显损害 B0 median。
- geometry freeze：保留 incumbent B0；没有 coordinate trajectory winner。
- 0.06：无数据依据，不降低，也不把它解释成 geometry correctness。
- calibrated policy：development experiment 失败，不部署。
- boundary：保留 fail-closed runtime gate；继续研究 evidence 化，精修失败始终 rollback 到 coarse。
- confidence-shaping：不做。margin/ratio 对 correctness 的预测价值不足，前提不成立。
- G4 / geometry FULL：不允许进入。
- 下一步：先按 P4 target-domain protocol 增加真正独立的 postcard/artwork/screen/poster development
  groups，修复 validation slices 的重叠，再重新做一次全程预注册的 geometry/acceptance audit。

## 权威产物

- trajectory：`/Users/caozichen/screenrestore-runs/p4-g36-manual-coordinate-trajectory-20260903/trajectory-selection-v2`
- acceptance：`/Users/caozichen/screenrestore-runs/p4-g36-manual-coordinate-trajectory-20260903/acceptance-validation-b0-v3`
- 单次 test：`/Users/caozichen/screenrestore-runs/p4-g36-manual-coordinate-trajectory-20260903/frozen-smartdoc-test-b0`

