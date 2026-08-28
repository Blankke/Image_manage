# P1 训练与模型使用

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

## 第二阶段：专项数据与调参

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
