# P4-G3.7 操作命令

## 环境与 preflight

在仓库目录运行。现有结果位于 `$HOME/screenrestore-runs/p4-g37-metric-v2`；
以下使用新的输出根复现。各阶段已有目录都会拒绝覆盖，请保留历史产物。

```bash
source .venv/bin/activate
which python
export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
export SCREENRESTORE_RUN_ROOT="$HOME/screenrestore-runs"
export P4_G37_RUN_DIRECTORY="$SCREENRESTORE_RUN_ROOT/p4-g37-$(date +%Y%m%d-%H%M%S)"
export P4_DEVICE=mps
bash scripts/run_p4_g37.sh preflight
```

## 数据审计

```bash
bash scripts/run_p4_g37.sh audit-data
```

当前既有清单缺少 digital_source_id，必须由已授权的来源记录人工补全，不能从照片路径或
训练 group 猜测。SmartDoc test 不在本轮允许访问范围。若数据管理员已提供预先生成的
无 GT 身份索引，可在审计时显式传入：

```bash
bash scripts/run_p4_g37.sh audit-data --test-identity-index "$P4_TEST_IDENTITY_INDEX"
```

两条 audit-data 命令二选一，在新的输出根执行。索引 JSON 契约为
`kind=geometry_identity_hash_index, split=test, rows=[...]`；每行只允许七个非空 SHA256 字段：
`image_path, normalized_path, image_sha256, digital_source_id, subject_id, group_id, capture_session`。
身份字符串按 Unicode NFC 后 UTF-8 SHA256；normalized_path 对数据根解析后的规范路径计算。
索引不得含照片路径、GT、模型预测或指标。当前没有该索引，test 独立性记为未验证；
本阶段脚本不提供读取 test 生成索引的命令。

## B0 baseline

```bash
bash scripts/run_p4_g37.sh baseline
```

重评 internal 1000、公开 calibration 5614、SmartDoc validation 2450；相同照片只推理一次。
随后每个 slice 用固定 32 张验证训练 evaluator、冻结 evaluator、Torch/ONNX parity。
parity 失败返回非零退出码，禁止训练。输出包含原生 512、metric-v2 与 checkpoint 身份。

## 512 sanity

当前状态：`BLOCKED: missing_comparable_256_arm`。
实际最近 1e-6 arm 为 content_only/none；它不满足 coordinate-only/full 的单变量前提。
此时下一条安全检查命令为：

```bash
bash scripts/run_p4_g37.sh sanity512
```

它会拒绝训练并留下阻塞记录。不要把最近 content-only run 冒充可比 arm，也不要自动补跑
256 或继续 learning-rate 搜索。需要先明确新的实验协议或提供真实存在的可比历史产物。
若未来已有合规 arm，入口支持 `--comparable-run`，并核对 run.json 的 recipe、
provenance.json 的 B0 与清单 SHA256；仅运行一轮，effective batch=16。

## Target-domain 采集、标注与审核

现在缺少全部 48 个真实 subjects：postcard、artwork、screen/display、poster 各 12 个。
每 subject 使用独立 digital_source_id、source-family group_id 和 subject_id，至少两个 sessions，
每 session 6–10 张正例，再额外约 25% hard negative。poster 的正例标为
`domain=poster, target_class=artwork, scene_type=poster`；screen/display 统一 `domain=screen`。

已授权公开照片放在 `$SCREENRESTORE_DATA_ROOT/geometry/p4-public-target/raw`，人工标注放在
`$SCREENRESTORE_DATA_ROOT/manifests/p4-public-target/annotations.geometry.jsonl`。
image 使用相对于数据根的路径。标注遵循 `datasets/schemas/geometry.schema.json`，额外完整填写
`domain/digital_source_id/subject_id/group_id/capture_session/scene_type/in_scope`，
并为每张照片填写 `source` 与 `source_license`。
`capture_conditions` 使用 `frontal, mild_perspective, moderate_perspective, near_border,
nested_layer, weak_edge, light_glare`，记录实际条件；每个 subject 合计覆盖这些条件。
partial/multi-target/curved/severe_occlusion/extreme_view 标为拒绝样本（present=false、
target_class=none、in_scope=false、content_quad=null），不能伪造正常 positive。

人工照片与标注尚未存在，现在可执行的下一条 readiness 命令为：

```bash
bash scripts/run_p4_g37.sh prepare-target
```

数据到位后，在新的输出根执行 prepare-target 生成带角点编号的 overlay 与 review.jsonl。
人工逐张查看 overlay，在 review.jsonl 填写 status=approved/rejected、reviewer、reviewed_at。
随后生成带审核绑定的新清单；命令中的环境变量应指向实际审核文件和外部输出位置：

```bash
python scripts/prepare_p4_target_domain.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --manifest "$SCREENRESTORE_DATA_ROOT/manifests/p4-public-target/annotations.geometry.jsonl" \
  --reviews "$P4_G37_RUN_DIRECTORY/prepare-target/overlays/review.jsonl" \
  --output-manifest "$SCREENRESTORE_DATA_ROOT/manifests/p4-public-target/reviewed.geometry.jsonl"
```

工具保留人工状态，不自动批准。审核同时绑定照片 SHA256 与整条标注 SHA256，修改任一后需重新审核。
四类各至少 8 groups、所有隔离检查和 overlay 审核通过后，才允许 acceptance。48 subjects 是
development 采集目标；完整发布仍要求至少 100 独立 evaluation groups，不能以首批 48 替代。

## 数据就绪后的 acceptance

以下命令仅在真实 reviewed manifest 与无 GT 身份索引已存在、既有 source 元数据已补齐时执行：

```bash
bash scripts/run_p4_g37.sh acceptance-dev \
  --target-manifest "$SCREENRESTORE_DATA_ROOT/manifests/p4-public-target/reviewed.geometry.jsonl" \
  --test-identity-index "$P4_TEST_IDENTITY_INDEX"
```

缺少上述条件会返回 `BLOCKED: insufficient_target_domain_groups`。fit 只拟合，selection 选策略和
阈值，evaluation 仅评估一次选定策略；source-family 单次锁保存在外部 run root。
当前 runtime policy 始终保持 fail-closed；研究脚本不会自动部署。
私人实拍集在公开权重与策略冻结后，使用 `scripts/run_private_validation.sh` 独立预测和评分，
不进入本阶段的 fit、selection 或阈值校准。

## 报告与检查

```bash
bash scripts/run_p4_g37.sh report
python -m compileall -q training scripts tests src
python -m ruff check .
XDG_STATE_HOME=/tmp/screenrestore-p4-g37-state python -m pytest -q
```

report 从阶段 JSON 生成外部结果与隔离 Markdown，审阅后再同步仓库正式文档。
