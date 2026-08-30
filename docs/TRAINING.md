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
python scripts/label_private_geometry.py \
  --data-root "$SCREENRESTORE_DATA_ROOT" \
  --image-directory "$SCREENRESTORE_DATA_ROOT/private" \
  --output "$SCREENRESTORE_DATA_ROOT/private/geometry.annotations.jsonl"
python -m training.quadlocator.generate_synthetic \
  --output-directory "$SCREENRESTORE_DATA_ROOT/geometry/synthetic" \
  --count 24000 --size 640 --negative-ratio 0.25 \
  --content-directory "$SCREENRESTORE_DATA_ROOT/textures/met-open-access/images" \
  --content-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR" \
  --background-directory "$SCREENRESTORE_DATA_ROOT/backgrounds/coco/val2017" \
  --background-directory "$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR"
python scripts/build_p2_geometry_manifests.py --data-root "$SCREENRESTORE_DATA_ROOT"
```

private 工具按内容指纹合并 thumbnail/HD，同一 group 固定 60%/20%/20% split；已有标注的
split 会保留。Stage C 清单只把 private-train 与 public/synthetic train replay 放进 train，
validation/test 中只保留对应 private group。层级或目标选择不确定的 private 标注会写成明确
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
bash scripts/train_p2_geometry.sh stage-c  # private train + replay，最多 8 epochs，7.5e-5
bash scripts/train_p2_geometry.sh stage-d  # 冻结模型，只做 validation calibration
```

`P2_WIDTH=1.5` 必须配合新的 `SCREENRESTORE_RUN_NAME` 从 Stage A 开始，不能覆盖 width=1.0
run。脚本检测到已有 `best.pt` 时默认拒绝覆盖。每阶段的 `history.json` 记录 NCE、IoU、mask、
boundary、outer presence precision/recall/FPR/Brier/ECE、class confusion、no-candidate、
layer-ambiguous、多目标歧义拒绝率和产品代理指标；best checkpoint 按多维
`selection_score` 选择，多目标误接受会受到独立惩罚。

最终 `docs/P2_GEOMETRY_RESULTS.md` 只能在各阶段训练、独立数据评测和 overlay 人工复核完成后
生成。private 独立 group 少于 100 时必须报告 `insufficient evidence`，不得改低正式 gate。

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
