# P1 训练结果（2026-08-28）

## 结论

P1 的数据准备、四阶段正式训练、checkpoint 与 ONNX 导出均已完成。当前唯一保留的正式
run 为：

```text
/Users/caozichen/screenrestore-runs/p1-full-20260828-133513
```

训练工程状态为完成，产品自动几何验收状态为 **FAIL**。QuadLocator 在 SmartDoc test 与
private 无 GT 数据上均为 0 自动接受，当前权重只能用于问题分析和下一轮训练基线，不能作为
无人值守自动裁切模型发布。Fidelity 与两个超分模型完成了各自训练集内的验证；这些指标来自
DIV2K 构造退化或配对切片，尚不代表真实画作/屏摄恢复质量。

## 数据与运行环境

| 项目 | 结果 |
| --- | --- |
| SmartDoc 清单 | train 20,760；validation 2,450；test 1,679 |
| SmartDoc 独立 group | train 25；validation 3；test 2 |
| DIV2K | train 800；validation 100 |
| DIV2K wild x4 | train 3,200 个 LR 配对；validation 100 |
| private | 47 张可解码图片；其中 23 组含缩略图与 HD，视觉复核按 HD 优先得到 24 张 |
| 训练设备 | Apple MPS |
| 数据目录占用 | 8.3 GiB |
| 运行目录占用 | 19 MiB（含评估 JSON 与 private 效果图） |
| 当前磁盘可用 | 53 GiB |

private 的 47 张图片以 `--with-private-identity` 参与 Fidelity identity 保护，权重为 0.20。
因此它们属于 Fidelity 的训练输入，不能用于该模型的泛化质量结论。private 没有几何 GT，相关
输出只用于分布审计和人工复核。

## 四阶段训练结果

### QuadLocator-S

| 项目 | 数值 |
| --- | ---: |
| 参数量 | 99,503 |
| 输入 | 512×512 |
| batch / epoch | 4 / 12 |
| 学习率 | 0.002，CosineAnnealing |
| 训练耗时 | 10,693.03 秒（约 2 小时 58 分） |
| 最佳 validation loss | 0.085281，第 9 轮 |
| 最后一轮 validation loss | 0.087949 |

最佳 loss 相比第 1 轮的 0.105720 下降约 19.3%，第 9 轮之后出现小幅回升。loss 收敛只表示
多任务训练目标下降，不能替代 accepted/rejected、NCE、IoU 与层级选择评估。

### Fidelity BoundedResidualNet

| 项目 | 数值 |
| --- | ---: |
| 参数量 | 113,028 |
| patch / batch / epoch | 192 / 8 / 20 |
| 训练耗时 | 2,662.08 秒（约 44 分） |
| 最佳 validation loss | 0.059338，第 11 轮 |
| PSNR / SSIM | 22.9217 dB / 0.922448 |
| clean identity MAE | 0.000224 |
| edge correlation | 0.829000 |
| color error（0..255） | 24.5887 |

第 1 轮到最佳轮的变化很小：PSNR 约提升 0.006 dB，SSIM 约提升 0.00008。模型基本在早期进入
平台期，需要依靠固定退化切片和真实手机配对集判断模型是否学到有效恢复，不能仅用混合 loss
继续增加 epoch。

### Conservative Super Resolution

| 模型 | 最佳轮 | validation loss | PSNR | SSIM | edge correlation | color error（0..255） | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x2 bicubic | 16 | 0.019816 | 31.1448 dB | 0.982159 | 0.956144 | 6.8003 | 1,559.65 秒 |
| wild x4 | 13 | 0.087892 | 18.6588 dB | 0.702073 | 0.643239 | 36.1929 | 4,893.79 秒 |

x2 在训练过程中持续改善。wild x4 的 PSNR 仅比第 1 轮高约 0.27 dB，SSIM 提高约 0.022，
绝对质量仍弱；该权重适合作为后续域差与退化建模的基线，暂不适合作为默认增强模型。

四阶段累计记录训练耗时约 19,808.55 秒（约 5 小时 30 分）。

## 自动几何评估

### SmartDoc test（有 GT）

正式 `e2e_auto` 报告：

```text
/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/smartdoc-test-e2e.json
```

| 指标 | 当前结果 | release gate | 状态 |
| --- | ---: | ---: | --- |
| 独立 group | 2 | ≥100 | FAIL |
| 自动接受 | 0 / 1,679 | — | FAIL |
| in-scope coverage | 0 | ≥0.90 | FAIL |
| accepted precision | 0 | ≥0.99 | FAIL |
| accepted NCE P95 | 1.0（无 accepted 时的保守值） | ≤0.01 | FAIL |
| accepted IoU median / P05 | 0 / 0 | ≥0.97 / ≥0.93 | FAIL |
| wrong layer rate | 0 | ≤0.005 | PASS，但分母中没有 accepted |

拒绝原因按多标签计数：`layer_ambiguous` 1,573、`boundary_uncertain` 340、
`corner_uncertain` 47、`invalid_quad` 33、`no_candidate` 73。

为避免 0 accepted 隐藏角点头的实际数值，另对所有 proposed quad 做诊断统计：

| 诊断指标 | 数值 |
| --- | ---: |
| proposed NCE median / P95 | 0.003218 / 0.125125 |
| proposed IoU median / P05 | 0.988417 / 0.454818 |
| coarse candidate NCE median / P95 | 0.004496 / 0.125125 |
| coarse candidate IoU median / P05 | 0.983962 / 0.454818 |

中位角点结果较好，尾部误差远超 gate；更关键的是统一置信度策略拒绝了全部样本，产品覆盖率为 0。

### 主要失败原因

SmartDoc 清单的 1,679 个 test 样本都有 `content_quad`，没有 `outer_quad`。当前训练 loss 对
outer heatmap 使用 `outer_present` 加权；outer 缺失样本的 outer loss 权重为 0，outer head
没有获得“当前样本无 outer”的负监督。模型契约也没有 outer presence 输出。

ONNX 运行时仍会尝试解码 outer heatmap。随机或未校准的 outer 响应通过基础热图阈值后，会进入
content/outer 包含关系检查，使 `layer_confidence` 降到 0，最终形成大规模
`layer_ambiguous`。private 效果图中的紫色 outer 候选也呈现出相同问题。

下一轮几何训练需要一次性统一升级模型与运行时契约：

1. 增加显式 `outer_presence_logits`，对 outer 存在与缺失都提供监督；运行时只在 outer presence
   通过校准阈值后解码和使用 outer quad。
2. 为 content/outer 热图分别记录可解释的验证指标，训练结束自动运行 e2e gate，避免用总 loss
   代替产品验收。
3. 扩充 artwork、screen、嵌套 content/outer、hard negative 和 incomplete target 的真实标注，
   严格按作品、设备与 capture session 切分。SmartDoc 当前 2 个 test group 只能用于回归诊断。
4. 先修复 outer 契约并重训，再在独立真实集校准 fail-closed 阈值；不应通过降低层级阈值换取接受率。

## private 无 GT 审计与效果图

全量 47 张图片的审计报告：

```text
/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/private-unlabeled-audit.json
```

结果为 0 解码/推理失败、0 accepted、47 次预测类别均为 `postcard`。拒绝原因按多标签计数为
`no_candidate` 43、`corner_uncertain` 4、`layer_ambiguous` 4、`boundary_uncertain` 2。

HD 优先去重后的 24 张人工复核效果图：

```text
/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/private-effects/contact-sheet.jpg
/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/private-effects/previews/
/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/private-effects/report.json
```

24 张均被拒绝，其中 2 张存在 proposed quad；总览图用橙色标 content、紫色标 outer。由于没有
accepted 样本，`rectified/` 没有输出。这符合产品的自动拒绝约束，也直观显示当前模型无法处理
private 中的美术馆画作、多作品同框和复杂画框分布。

## 当前模型文件

| 模型 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `geometry/quadlocator-s.onnx` | 428,007 | `dcfa62e8b692557c6dd2607b0cd6b8481e3ef1d5affc89edb504c63f5b35a1c3` |
| `restoration/fidelity-residual.onnx` | 462,760 | `54d55d262868bd1c26515d38969acc428f992fb4dd16718610676080a367dd03` |
| `superres-x2/conservative-superres.onnx` | 463,045 | `339d4f3d600229dbe9c605445603396d2bf7f91b1697e20f7d44507cac6bd9cf` |
| `superres-wild-x4/conservative-superres.onnx` | 463,045 | `055d065e64f244e0b2cd1ed2ab02c5fc8c106a08e01c047f35dfe28e7de993e4` |

## 清理记录

删除前已确认没有训练或评估进程使用历史目录。以下目录已永久删除，无法从本机回收：

| 目录 | 删除原因 | 删除前大小 |
| --- | --- | ---: |
| `p1-20260827-203343` | 仅完成 geometry，已被正式 run 取代 | 504 KiB |
| `p1-full-20260827-204608` | wild x4 阶段中断，缺少 history/run/ONNX | 4.9 MiB |
| `p1-full-20260828-102819` | restoration 第 1 轮中断，后续阶段为空 | 2.2 MiB |
| `p1-full-all-data-20260827` | 只有 geometry/run.json 空壳 | 4 KiB |
| `p1-smoke-20260827-wild` | 旧 smoke，仅完成前两阶段 | 772 KiB |
| `p1-smoke-all-data-20260827` | 已完成 smoke，已被正式 run 取代 | 1.3 MiB |
| `phase1-baseline-20260827` | 基线空壳 | 4 KiB |

合计释放约 9.7 MiB。SmartDoc 下载日志与 PID 记录保持原样；当前 run、数据、模型权重和 private
原图均未删除。

## 复现命令

所有 Python 命令必须先进入项目虚拟环境并确认解释器：

```bash
cd /Users/caozichen/Image_manage
source .venv/bin/activate
which python
python -c 'import sys; print(sys.executable)'
```

复现 private 全量无 GT 审计：

```bash
python scripts/audit_unlabeled_geometry.py \
  --image-directory "/Users/caozichen/screenrestore-data/private" \
  --quad-model "/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/quadlocator-s.onnx" \
  --output "/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/private-unlabeled-audit.json"
```

复现 HD 优先的 private 效果图：

```bash
python scripts/render_unlabeled_geometry.py \
  --image-directory "/Users/caozichen/screenrestore-data/private" \
  --quad-model "/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/quadlocator-s.onnx" \
  --output-directory "/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/private-effects"
```

复现 SmartDoc test。脚本最终返回 1 是当前 release gate FAIL 的预期结果，JSON 仍会正常写入：

```bash
EVAL_DIRECTORY="$(mktemp -d /tmp/screenrestore-smartdoc-test.XXXXXX)"
jq -r 'select(.split == "test") | .image' \
  /Users/caozichen/screenrestore-data/manifests/smartdoc.geometry.jsonl |
while IFS= read -r relative_path; do
  link_path="$EVAL_DIRECTORY/$relative_path"
  mkdir -p "$(dirname "$link_path")"
  ln -s "/Users/caozichen/screenrestore-data/$relative_path" "$link_path"
done

python -m benchmarks.geometry_e2e.run \
  --data-directory "$EVAL_DIRECTORY" \
  --manifest /Users/caozichen/screenrestore-data/manifests/smartdoc.geometry.jsonl \
  --dataset-root /Users/caozichen/screenrestore-data \
  --split test \
  --quad-model /Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/quadlocator-s.onnx \
  --output /Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/smartdoc-test-e2e.json

find "$EVAL_DIRECTORY" -depth -delete
```

## 本次新增与验证

- 新增 `scripts/render_unlabeled_geometry.py`，用于可复现地生成匿名 private 几何复核图、总览图
  和有限 JSON 报告。
- 运行 private 全量无 GT 审计、SmartDoc test `e2e_auto`、效果图生成和目视检查。
- 使用 ONNX Runtime 验证三个图像模型：Fidelity 保持输入尺寸，x2 与 wild x4 分别输出 2×、
  4× 尺寸；输出均为有限值并位于 `[0, 1]`。QuadLocator ONNX 已由 1,679 张 SmartDoc test 与
  47 张 private 审计实际运行验证。

最终质量检查：

| 检查 | 结果 |
| --- | --- |
| `python -m pytest -q` | PASS，134 passed |
| `python -m ruff check src tests scripts training benchmarks` | PASS |
| `python -m compileall -q src training benchmarks scripts tests` | PASS |
| 新增脚本 `ruff format --check` | PASS |
| 仓库级 `ruff format --check` | FAIL，99 个既有文件未按当前 Ruff formatter 排版 |
| `git diff --check` | PASS |
| typecheck | NOT CONFIGURED |
| security / contract / race / vet 专项命令 | NOT CONFIGURED |

本次没有批量格式化 99 个既有文件，避免把无关机械改动混入 P1 结果整理。仓库级格式门仍需在
独立维护任务中统一处理。
