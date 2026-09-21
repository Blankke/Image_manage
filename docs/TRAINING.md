# 训练与模型使用

2026-08-28 正式 run 的训练指标、SmartDoc e2e gate、private 无 GT 效果图与历史产物清理记录见
[`P1_RESULTS.md`](P1_RESULTS.md)。

2026-08-30 P2 自动几何的 mixed-domain 训练、分数据源评估、private 标注复核与 Stage D 校准结论见
[`P2_GEOMETRY_RESULTS.md`](P2_GEOMETRY_RESULTS.md)。

## 数据角色

| 数据 | 可用于 | 不可用于 |
| --- | --- | --- |
| SmartDoc | QuadLocator 的 content 四角、mask、boundary、presence 与类别监督 | 宣称为画作/屏幕真实分布，或推断 outer 标签 |
| DIV2K HR + x2 bicubic | Fidelity 在线退化恢复监督与定量验证 | 替代真实手机退化验证 |
| DIV2K wild x4 | 真实 x4 退化的配对审计、后续独立 SR 域差验证 | 混入同尺寸 Fidelity，或作为当前 x2 bicubic 样本 |
| `private/` 无 GT | 显式 identity 保护、无标签自动定位审计 | 几何真值、clean target、自动接受正确率 |

所有数据路径相对于 `SCREENRESTORE_DATA_ROOT`。训练不写增强副本；checkpoint、ONNX 和
指标只写入 `SCREENRESTORE_RUN_ROOT`。

## 启动训练

```bash
source .venv/bin/activate
which python
export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
export SCREENRESTORE_RUN_ROOT="$HOME/screenrestore-runs"

# 小规模端到端 smoke；会依次验证几何、同尺寸恢复、x2 与 wild-x4 超分。
bash scripts/train_p1.sh smoke --with-private-identity

# 使用所有已下载数据的正式分阶段训练；也可省略 full。
bash scripts/train_p1.sh full --with-private-identity
```

`smoke` 的目标是检查 GPU/MPS、数据解析、checkpoint、ONNX 和指标链路；不能作为质量结论。
`full` 先完成 SmartDoc 几何训练，再分别完成 DIV2K Fidelity、x2 bicubic 超分与 wild-x4
超分，同一时间只运行一个 MPS 训练任务。私有图片必须由操作者显式传入
`--with-private-identity`，脚本默认不读取它。

## P2 geometry contract repair 与 mixed-domain retraining

P2 geometry 与恢复专项分离。`scripts/train_p2_geometry.sh` 只调用 QuadLocator 训练、ONNX
导出、几何 benchmark、overlay 和冻结 calibration，不会调用 Fidelity 或超分入口。

P2 checkpoint `format_version=2`，ONNX 固定七个输出：content/outer corner heatmap、content
mask、boundary、content presence、outer presence、class。训练入口的 `--init-checkpoint`
按名称与 shape 加载 P1 参数，新 outer presence head 保持初始化，并把 loaded/skipped/missing
名称与参数数写入终端和 `run.json`。

数据与标注准备顺序：

```bash
source .venv/bin/activate
which python
python scripts/prepare_p2_geometry_data.py \
  --dataset all --data-root "$SCREENRESTORE_DATA_ROOT" --met-count 1500
python -m training.quadlocator.generate_synthetic \
  --output-directory "$SCREENRESTORE_DATA_ROOT/geometry/synthetic" \
  --count 24000 --size 640 --negative-ratio 0.25 \
  --content-directory "$SCREENRESTORE_DATA_ROOT/textures/met-open-access/images" \
  --content-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR" \
  --background-directory "$SCREENRESTORE_DATA_ROOT/backgrounds/coco/val2017" \
  --background-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR"
python scripts/build_p2_geometry_manifests.py --data-root "$SCREENRESTORE_DATA_ROOT"
```

当前私人开发集由 `scripts/private_development_validation.py` 在预测冻结后评分；训练入口会
拒绝清单中带私人来源、分组或图片路径的记录。层级或目标选择不确定的公开样本会写成明确
拒绝监督。标注窗口用 `M` 标记多个完整且同等合理的目标；这类样本统一写成
`scene_type=gallery_multi_target`、`ambiguous=true` 和 `present=false`，不得附带任意一幅画的
四角。存在明确主画、其它目标仅局部入镜时仍按主画标注。overlay 报告会单列多目标样本数、
拒绝数和拒绝率。合成器还会先按稳定散列拆分 Met/DIV2K 内容纹理与 COCO/DIV2K 背景纹理，保证同一
公开作品或场景不会通过不同透视与退化跨 split 泄漏。

手动训练时为一次实验固定同一个 run name：

```bash
export SCREENRESTORE_RUN_NAME="p2-geometry-w1-$(date +%Y%m%d-%H%M%S)"
export P2_DEVICE=mps  # Apple Silicon 正式训练禁止静默回落 CPU
bash scripts/train_p2_geometry.sh preflight  # 只读检查全部输入、30 GiB 数据上限与 10 GiB 余量
bash scripts/train_p2_geometry.sh stage-a  # P1 warm start，3 epochs，3e-4
bash scripts/train_p2_geometry.sh stage-b  # mixed-domain，最多 18 epochs，early stopping
bash scripts/train_p2_geometry.sh stage-d  # 用 stage-b 权重做公开 validation calibration
```

`P2_WIDTH=1.5` 必须配合新的 `SCREENRESTORE_RUN_NAME` 从 Stage A 开始，不能覆盖 width=1.0
run。脚本检测到已有 `best.pt` 时默认拒绝覆盖。每阶段的 `history.json` 记录 NCE、IoU、mask、
boundary、outer presence precision/recall/FPR/Brier/ECE、class confusion、no-candidate、
layer-ambiguous、多目标歧义拒绝率和产品代理指标；best checkpoint 按多维
`selection_score` 选择，多目标误接受会受到独立惩罚。

最终 `docs/P2_GEOMETRY_RESULTS.md` 只能在各阶段训练、独立数据评测和 overlay 人工复核完成后
生成。private 独立 group 少于 100 时必须报告 `insufficient evidence`，不得改低正式 gate。

## 私人开发验证闭环

SmartDoc 的五种背景在不同文档模型中复用；按文档模型划出的原始 validation/test
无法证明跨拍摄场景泛化。当前公开 replay 构建统一把 SmartDoc 原片和纸面衍生图限制在
train，训练读取器同时按 `scene_group_id` 拦截旧的跨 split 清单。P11 的纸面验证属于
同场景诊断，不能用于独立目标域验收。场景隔离版可从已审计的 P11 清单复现：

```bash
source .venv/bin/activate
which python
python scripts/isolate_smartdoc_scenes.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --input "$SCREENRESTORE_DATA_ROOT/manifests/p11/p8-gallery-photo-docci-full-bleed-replay-20260918.geometry.jsonl" \
  --output "$SCREENRESTORE_DATA_ROOT/manifests/p12/scene-isolated-replay-20260918-v3.geometry.jsonl"
```

P12 清单保留 80,688 条，排除 4,174 条相关 validation/test；另外把 13,431 条旧
`present=false` 合成负样本的 `visible` 改为 false，逐条通过 geometry schema。
MIDV-500 与 MIDV-Holo 的不同证件组也复用了拍摄布景。P13 在 P12 基础上再排除
2,357 条相关 validation/test，保留 78,331 条；训练读取器会拒绝旧的跨场景清单：

```bash
python scripts/isolate_midv_scenes.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --input "$SCREENRESTORE_DATA_ROOT/manifests/p12/scene-isolated-replay-20260918-v3.geometry.jsonl" \
  --output "$SCREENRESTORE_DATA_ROOT/manifests/p13/scene-isolated-replay-20260918.geometry.jsonl"
python scripts/build_p2_geometry_manifests.py --data-root "$SCREENRESTORE_DATA_ROOT"
```

P13 replay 的画作目标域 validation 仅有 10 张 SynGallery 合成视角；P13 基础清单的
calibration 全部来自合成图。这些处理不增加独立纸面验证场景；后续仍需采集或取得
清楚授权的独立目标域照片，才能选择和检验自动接受策略。

`private_set` 的 13 个真实对象各有 6 个不同视角。它们必须按对象共享 `group_id`，同时为每张
照片单独标注 `content_quad` 和可选 `outer_quad`。旧的 `label_private_geometry.py` 会给同组的
缩略图/HD 同图对复用四角，不适用于这批不同视角；本集合统一使用
`scripts/private_development_validation.py`。

一次公开训练迭代的固定顺序如下：

1. 只用公开 train 训练；只用公开 validation 选择 checkpoint 和阈值。
2. 导出候选 ONNX 并完成 Torch/ONNX parity，随后冻结 checkpoint、ONNX、代码状态与数据索引摘要。
3. `freeze` 只读取 78 张手机照片并保存预测，此时禁止读取人工四角。
4. `label` 逐帧标注并人工批准；`score` 核对图片、索引、预测和标注摘要后计算总体、分域、分视角指标。
5. `restore` 为 13 个对象各输出一次固定正视帧的人工几何 Archive 恢复；e2e 自动恢复仅从策略已接受的
   视角中按冻结置信度选择，不用真值挑图，拒绝对象保持拒绝。

`restore` 的 `restoration-report.json` 必须列出实际恢复栈。训练得到的 Fidelity/Photometric `.pt`
checkpoint 只有在完成正式导出、数值一致性和产品运行时接入后才能列入 learned restoration models；
未接入时输出属于经典 Archive 基线，不能宣称为训练模型恢复结果。

复现入口：

```bash
source .venv/bin/activate
which python
export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
export PRIVATE_VALIDATION_IMAGES="$SCREENRESTORE_DATA_ROOT/private-set-extract/private_set"
export PRIVATE_VALIDATION_ANNOTATIONS="$SCREENRESTORE_DATA_ROOT/private-validation/annotations.jsonl"
export PRIVATE_VALIDATION_QUAD_MODEL="$SCREENRESTORE_RUN_ROOT/<run>/geometry/quadlocator-s.onnx"
export PRIVATE_VALIDATION_CALIBRATOR="$SCREENRESTORE_RUN_ROOT/<run>/audits/<公开验证审计>/correctness-calibrator.json"
export PRIVATE_VALIDATION_RUN="<run>-private-dev"

bash scripts/run_private_validation.sh prepare
bash scripts/run_private_validation.sh freeze  # 必须先于 label/score
bash scripts/run_private_validation.sh label   # C/O 标四角，R 批准，S 暂存，Q 退出；重启后跳过已批准帧
bash scripts/run_private_validation.sh score
bash scripts/run_private_validation.sh review  # 绿色人工 content、蓝色人工 outer、红色冻结模型候选
bash scripts/run_private_validation.sh restore
```

同一私人开发集的后续冻结模型应设置
`PRIVATE_VALIDATION_INDEX="$PWD/output/private-validation/b0-20260912/prepare/dataset-index.json"`，
复用首次 `prepare` 的不可变索引；各模型仍使用独立 `PRIVATE_VALIDATION_RUN`，避免覆盖历史预测。

CLI 自动几何必须同时加载冻结 ONNX 与其公开验证校准器：

```bash
screenrestore input.jpg --output restored.png --corners auto \
  --quad-model "$SCREENRESTORE_RUN_ROOT/<run>/geometry/quadlocator-s.onnx" \
  --correctness-calibrator "$SCREENRESTORE_RUN_ROOT/<run>/audits/<公开验证审计>/correctness-calibrator.json"
```

这批数据没有 clean reference。它能量化自动几何、错层、拒绝行为和跨视角稳定性，也能用于人工
检查反光、色偏、饱和与纹理保持；不能生成有意义的 PSNR、SSIM 或 Delta E 原图还原分数。后续应为
自有纸质对象补充平板扫描，为屏幕对象保存实际显示的源文件，作为完全离线的 paired reference。
由于该集合会在每轮公开训练后反复使用，它只属于开发验证；最终发布还需另留从未用于调参的盲测集。

历史 B0 生成时，几何 augmentation 曾存在图片与四角标签不同步的问题；当前数据加载器已经把 tensor
构造移动到同步增强之后。修复性训练使用 `scripts/train_p5_geometry_recovery.sh`，从 B0 低学习率微调，
并以 epoch 0 的公开 validation median/tail 为保护门。私人标注只在模型和 ONNX 冻结后用于开发评分。

## P5 正式宽模型长训节点

P5 的公开数据路线使用 `target-v5-replay-20260913.geometry.jsonl`：P2 公开 replay 提供真实文档几何，
target-v5 合成集补充画作、薄边明信片、屏幕 bezel/支架，以及“暗框画作、画架不是屏幕”的反事实。
训练、选模、接受门和阈值均不得读取私人开发集。

`width_multiplier=1.5` 的 Stage A 已从随机初始化完成 2 epoch 预演。完整公开 validation 共 6414 张；
冻结审计得到 content NCE median `0.00463`、P95 `0.20769`、Quad IoU median `0.98033`、
P05 `0.50306`。严格 hard gate 接受 1503 张，正确 1483 张，接受正确率 `98.67%`、
严格正确样本覆盖率 `45.05%`。几何中位数已经明显改善，但接受正确率仍低于 `99%` 发布门，
所以该 checkpoint 只是长训初始化点，不能替换当前正式候选，也不能先到私人集调阈值。

主干全任务长训固定使用公开 train/validation 全量、宽度 1.5、自然 source/group 均衡采样、
14 个 epoch 和余弦学习率。受控预演显示，额外启用类别均衡会明显损伤 content NCE P95，
因此类别均衡仅保留给几何冻结后的决策头消融，不进入主干长训。
自然采样的 8000 张/1 epoch 预演在完整公开审计上将 NCE P95 改善到 `0.15999`、IoU P05
改善到 `0.55310`、严格几何率改善到 `69.64%`，同时 hard-gate 召回升到 `67.34%`；接受正确率
却暂时降到 `95.83%`。该权重不晋升，也不作为长训起点。长训仍从 Stage A 的 `98.67%`
接受正确率 checkpoint 开始，逐个保留里程碑；训练结束后以完整公开冻结审计选择同时满足几何与
`>=99%` 接受正确率的 checkpoint，不能依据训练内 proxy 或私人集挑选。
Stage A 实测吞吐折算约需 11–13 小时；机器或系统负载变化会影响实际时间。中间 checkpoint 在
1/2/4/6/8/10/12/14 epoch 保存，`best_geometry.pt` 继续受 epoch-0 median/tail 保护门约束。
可选 collapse watchdog 只在连续三轮出现严重几何尾部退化时提前终止，避免错误训练持续覆盖计算资源。

使用抽样训练集时，`P5_REQUIRE_TRAIN_SOURCES` 和 `P5_REQUIRE_VALIDATION_SOURCES` 可分别用
逗号分隔的 `SOURCE=MIN_COUNT` 固定稀有来源最低数量；`P5_SAMPLES_PER_EPOCH` 控制每轮实际
抽取数。每次 warm start 默认要求至少 95% 的模型参数值成功加载，避免模型宽度与 checkpoint
不符时误把随机初始化训练当成微调。需要有意迁移不同架构时，再显式调整
`P5_MIN_WARM_START_PARAMETER_RATIO`。

```bash
cd /Users/caozichen/Image_manage
source .venv/bin/activate
which python

export SCREENRESTORE_DATA_ROOT=/Users/caozichen/screenrestore-data
export SCREENRESTORE_RUN_ROOT=/Users/caozichen/screenrestore-runs
export P5_RUN_NAME=p5-wide-w15-v5-long-20260914
export P5_INITIALIZATION=warm_start
export P5_BASELINE_CHECKPOINT=/Users/caozichen/screenrestore-runs/p5-wide-w15-v5-stage-a-20260913/geometry/best_geometry.pt
export P5_MANIFEST=/Users/caozichen/screenrestore-data/manifests/p5/target-v5-replay-20260913.geometry.jsonl
export P5_WIDTH_MULTIPLIER=1.5
export P5_IMAGE_SIZE=512
export P5_LOSS_PROFILE=full
export P5_TRAINABLE_SCOPE=all
export P5_CLASS_BALANCED_SAMPLING=0
export P5_CHECKPOINT_KIND=geometry
export P5_LEARNING_RATE=1e-4
export P5_EPOCHS=14
export P5_SCHEDULER_T_MAX=14
export P5_CHECKPOINT_EPOCHS=1,2,4,6,8,10,12,14
export P5_TRAIN_SAMPLES=0
export P5_VALIDATION_SAMPLES=0
export P5_BATCH_SIZE=8
export P5_WORKERS=0
export P5_DEVICE=mps
export P5_EARLY_STOPPING_PATIENCE=0
export P5_GEOMETRY_COLLAPSE_PATIENCE=3
export P5_GEOMETRY_COLLAPSE_NCE_P95_RATIO=1.35
export P5_GEOMETRY_COLLAPSE_IOU_P05_RATIO=0.70

bash scripts/train_p5_geometry_recovery.sh preflight
set -o pipefail
caffeinate -dimsu bash scripts/train_p5_geometry_recovery.sh train 2>&1 \
  | tee /Users/caozichen/screenrestore-runs/p5-wide-w15-v5-long-20260914.console.log
```

训练完成后先对 `best_geometry.pt` 执行完整公开冻结审计。仅当公开指标和发布接受门均确认后，
才导出并冻结私人开发集预测；私人评分仍只决定该公开候选是否通过开发验收，不反向改权重、阈值或
checkpoint。决策头短训可作为长训后的独立消融，但当前 8000 张/1 epoch 预演虽然改善 artwork/none
类别召回，却降低了公开综合选择分数，因此不进入上述主训练命令。

### 2026-09-15 长训冻结结论

`p5-wide-w15-v5-long-20260914` 已完成 14/14 epoch，公开 validation 的 epoch 14 同时成为
`best_geometry.pt` 与 `best_product.pt`。原 decoder-v2 的最终 NCE median/P95 为
`0.00428/0.04051`，IoU median/P05 为 `0.98193/0.86620`，严格几何率 `80.15%`。
现有 hard gate 接受正确率只有 `93.05%`，不能发布；公开 group 隔离校准器在独立 evaluation
达到 `100%` 接受正确率，覆盖率仅 `14.37%`。

实拍开发集揭示独立四角峰会跨多个矩形实例拼接。正式运行时使用
`quad-coherent-evidence-guarded-v1`：联合角点峰值、content mask 与 boundary 选择同一个
内容实例，并要求替代候选的证据乘积至少是合法独立峰的 2 倍。`quad-coherent-repair-v1`
仅保留在离线审计中作为历史对照；全量 `quad-coherent-evidence-v1` 也只作为上界对照，避免
soft mask 噪声过度改写原本正确的合法四角。

在 stage2 与 v6+v7 完整公开验证上，guarded 相对 repair 将 NCE P95 从 `0.09036` 改善到
`0.08845`，IoU P05 从 `0.74238` 改善到 `0.76155`，严格几何率从 `70.61%` 提升到
`70.72%`。公开独立 evaluation 在 `100%` 接受正确率下覆盖为 `1.58%`，尚未解决安全覆盖
不足。冻结私人集上总体 IoU 中位数保持 `0.79143`，screen 中位数提升到 `0.98808`，far
中位数提升到 `0.37232`；总体 IoU P05 从 `0.14770` 降到 `0.09964`，该退化继续作为下一轮
候选选择训练与解码校准的重点。

历史冻结修复版在 78 张私人开发图上的候选 NCE median 从 `0.14166` 降到 `0.08199`，IoU median
从 `0.57723` 提升到 `0.71114`，严格几何率从 `14.10%` 提升到 `19.74%`，错吸附 outer 的比例
从 `26.09%` 降到 `17.39%`。公开校准器拒绝全部 78 张，说明 fail-closed 生效，真实目标域覆盖
仍未建立。下一轮训练应补充与私人验证集隔离的真实画作、海报和屏幕训练组，重点训练类别、presence
与多实例歧义；继续增加当前合成 replay 的 epoch 不足以解决真实域偏移。

## 第二阶段：恢复专项数据与调参

去噪、去模糊、光度、反光、去摩尔纹和超分数据统一使用
`datasets/schemas/restoration.schema.json`。每条记录都有 `task`、`split`、`group_id`、
`capture_session`、`input_image`、`target_image`、数据来源和许可证；反光多帧还必须列出
`observed_frames`。路径必须相对于数据根，原始图片和训练结果均不提交仓库。

在任何专项训练前运行审计：

```bash
source .venv/bin/activate
which python
python scripts/audit_restoration_manifest.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --manifest "$SCREENRESTORE_DATA_ROOT/manifests/<专项>.restoration.jsonl"
```

本地受控采集集若位于 `private/`，必须由操作者明确确认后才可传入 `--allow-private`。审计只输出
记录数、任务/split 分布和尺寸计数，不输出图片内容或文件名。

当前 Fidelity checkpoint 的调参使用固定切片：clean identity、轻/重噪声、失焦、运动模糊、JPEG、
曝光、白平衡、光照梯度和组合相机退化。各候选必须同时保持 clean identity，并分别比较对应切片，
不能用混合总分覆盖某一专项退化：

```bash
python -m training.restoration.evaluate_slices \
  --checkpoint "$SCREENRESTORE_RUN_ROOT/p1-<时间戳>/restoration/best.pt" \
  --hr-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
  --output "$SCREENRESTORE_RUN_ROOT/p1-<时间戳>/restoration/evaluation-slices.json" \
  --samples 100 --device auto
```

反光、去摩尔纹、色彩参数预测和超分必须各自拥有带任务 GT 的独立训练与测试清单；它们不能借
Fidelity 切片评测宣称能力。

## 使用训练结果

设运行目录为 `$SCREENRESTORE_RUN_ROOT/p1-<时间戳>`。

几何模型使用同一产品定位服务：

```bash
screenrestore input.jpg \
  --corners auto \
  --quad-model "$SCREENRESTORE_RUN_ROOT/p1-<时间戳>/geometry/quadlocator-s.onnx" \
  --output output.png \
  --json-diagnostics
```

Fidelity 恢复 ONNX 是同尺寸、有限残差模型。安装 ONNX 可选依赖后，在运行目录新建本地清单
`fidelity-residual.json`：

```json
{
  "id": "p1-fidelity-residual",
  "name": "P1 Fidelity bounded residual",
  "type": "onnx",
  "role": "restoration",
  "task": "denoise",
  "model_path": "fidelity-residual.onnx",
  "required_files": ["fidelity-residual.onnx"],
  "supports_tiling": true,
  "tile_size": 256,
  "tile_overlap": 32,
  "tile_padding": 16,
  "license": "项目自训练权重；仅限本地实验",
  "timeout_seconds": 3600
}
```

将该清单路径填入 GUI/CLI 的“AI 恢复模型”节点。它属于 Archive 的观测恢复先验，输出仍需保留
来源报告；不得把它当作对饱和、遮挡或反光覆盖内容的真实重建。

## 无 GT 与 wild 审计

private 的无 GT 图像完成训练后只能用于分布审计：

```bash
source .venv/bin/activate
python -m pip install -e '.[inference-onnx]'
python scripts/audit_unlabeled_geometry.py \
  --image-directory "$SCREENRESTORE_DATA_ROOT/private" \
  --quad-model "$SCREENRESTORE_RUN_ROOT/p1-<时间戳>/geometry/quadlocator-s.onnx" \
  --output "$SCREENRESTORE_RUN_ROOT/p1-<时间戳>/geometry/private-unlabeled-audit.json"
```

报告只含接受数、类别计数与拒绝原因，不含图像名称或像素。DIV2K wild 配对记录保存在
`manifests/div2k.restoration.jsonl` 的 `wild_x4_images`；`full` 会把 x2 和 wild-x4 分别训练为
独立的保守超分模型，禁止混成同一个数据源或权重。

## P3 高置信几何与 faithful restoration

P3 使用 `scripts/train_p3.sh` 作为唯一分阶段入口。B0 固定为 P2 Stage B epoch 12，B1/B3 是
短 warm-start 消融，B2/B4 只做评估与校准代码闭环，B5 是唯一 FULL geometry。恢复顺序为最多
一个强专项 → FidelityNet-v2 → 参数化 PhotometricNet → 可选保守 SR；在线退化不会写 augmentation
cache。

可直接执行的完整阶段命令、真实清单路径、外部数据 BLOCKED 状态与条件 B6/SR 门见
[`P3_TRAINING_COMMANDS.md`](P3_TRAINING_COMMANDS.md)。正式训练要求 `P3_DEVICE=mps`，MPS 不可用时
硬失败；输出阶段目录已存在时拒绝覆盖。
