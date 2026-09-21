# P4 数据隔离审计

| slice | samples | sources | groups | sessions |
|---|---:|---:|---:|---:|
| stage_b_train | 45008 | 0 | 19245 | 505 |
| internal_validation | 1000 | 0 | 436 | 246 |
| calibration | 5624 | 0 | 2411 | 280 |
| smartdoc_validation | 2450 | 0 | 3 | 15 |
| target_development | 0 | 0 | 0 | 0 |

0 个 source 可能代表身份缺失，不能解释为无泄漏。

| pair | path | SHA256 | source | group | session |
|---|---:|---:|---:|---:|---:|
| stage_b_train__internal_validation | 0 | 0 | 0 | 0 | 0 |
| stage_b_train__calibration | 0 | 0 | 0 | 0 | 0 |
| stage_b_train__smartdoc_validation | 0 | 0 | 0 | 0 | 0 |
| stage_b_train__target_development | 0 | 0 | 0 | 0 | 0 |
| internal_validation__calibration | 1000 | 1000 | 0 | 436 | 246 |
| internal_validation__smartdoc_validation | 430 | 430 | 0 | 3 | 15 |
| internal_validation__target_development | 0 | 0 | 0 | 0 | 0 |
| calibration__smartdoc_validation | 2450 | 2450 | 0 | 3 | 15 |
| calibration__target_development | 0 | 0 | 0 | 0 | 0 |
| smartdoc_validation__target_development | 0 | 0 | 0 | 0 | 0 |

Test isolation: test_isolation_unverified。
既有开发切片只作历史诊断；source 身份不完整或 test 独立性未验证时，禁止进入正式 fit/selection/evaluation。

## 身份完整性与角色

| slice | subject IDs | 缺失 source IDs | 缺失 subject IDs | 缺失 group/session | 允许用途 |
|---|---:|---:|---:|---:|---|
| P2 stage-b train | 0 | 45008 | 45008 | 0/0 | 既有训练数据诊断；source 独立性未证实 |
| internal validation | 0 | 1000 | 1000 | 0/0 | 重叠开发诊断 |
| P2 calibration | 0 | 5624 | 5624 | 0/0 | 历史 evidence；不能替代 target calibration |
| SmartDoc validation | 0 | 2450 | 2450 | 0/0 | 3 groups smoke slice |
| SmartDoc test | 未审计 | 未审计 | 未审计 | 未审计 | 无图片/GT 访问权限；等待预先生成的 hash-only 身份索引 |
| target development | 0 | 无数据 | 无数据 | 无数据 | BLOCKED；fit/selection/evaluation 均未启用 |

SHA256 检查覆盖实际开发图片字节，路径经过 resolve/Unicode NFC 后计算身份摘要；
source/subject/group/session 独立计算，未用图片路径代替 source family。
所有已列开发图像均存在；同一 source 被重新拍摄、打印或裁切的情况仍需可信 digital_source_id。
目前 source 字段缺失，零 source 交集代表 unknown，不代表证明没有 source-family leakage。

train 与这些 validation slices 没有已知图片、group 或 session 交集。internal/calibration/SmartDoc
之间存在明确 image leakage、group leakage、capture-session leakage。不得把三张表当作三份独立证据。

目标域启用前还必须验证 group/source/subject/session 不跨 fit/selection/evaluation，四类各 ≥8
独立 group，source-family 固定 group，人工 overlay 审核有效；检查采用失败关闭策略。
生成协议与审核命令见 `docs/P4_NEXT_COMMANDS.md`。

## 可复现产物

当前审计：`$SCREENRESTORE_RUN_ROOT/p4-g37-metric-v2/audit-data`。
JSON 包含各字段 missing 计数、split/target_class/scene_type 分布与所有开发集合两两交集；
provenance.json 保存 commit、命令、B0 SHA256、清单 SHA256 和代码摘要。
数据根为 24,085,884 KiB，低于 30 GiB 上限。
