# P3 正式训练命令

以下命令已按当前 `scripts/train_p3.sh`、各 Python CLI 的 `--help` 和本地真实清单路径核对。
本次代码任务没有执行这些长训练阶段。整组命令使用同一个唯一 run name；已有阶段目录会使脚本
硬失败，避免覆盖 checkpoint、ONNX 或 evaluation。

## 环境与固定路径

```bash
cd /Users/caozichen/Image_manage
source .venv/bin/activate
which python

export SCREENRESTORE_DATA_ROOT=/Users/caozichen/screenrestore-data
export SCREENRESTORE_RUN_ROOT=/Users/caozichen/screenrestore-runs
export SCREENRESTORE_RUN_NAME="p3-full-$(date +%Y%m%d-%H%M%S)"
export P3_DEVICE=mps
export P3_SEED=20260830
```

脚本实际读取：

- geometry：`$SCREENRESTORE_DATA_ROOT/manifests/p2/stage-b.geometry.jsonl`
- calibration：`$SCREENRESTORE_DATA_ROOT/manifests/p2/calibration.geometry.jsonl`
- SmartDoc test：`$SCREENRESTORE_DATA_ROOT/manifests/smartdoc.geometry.jsonl`
- Fidelity/专项 synthetic source：`$SCREENRESTORE_DATA_ROOT/superres/div2k/DIV2K_train_HR`
- 固定 B0：`/Users/caozichen/screenrestore-runs/p2-geometry-w1-20260829-110658/stage-b/best.pt`
- 固定 B0 ONNX：同目录 `quadlocator-s.onnx`

## Preflight、smoke 与 geometry

```bash
bash scripts/train_p3.sh preflight
bash scripts/train_p3.sh smoke

# B0：固定 P2 epoch 12，只评估。
bash scripts/train_p3.sh geometry-b0

# B1：boundary-only，2 epochs，短 warm-start 消融。
bash scripts/train_p3.sh geometry-b1

# B2：code-only；对 validation 冻结预测、提取 24 个特征并拟合 JSON logistic 校准器。
bash scripts/train_p3.sh geometry-b2

# B3：tail loss + hard sampling，3 epochs，短 warm-start 消融。
bash scripts/train_p3.sh geometry-b3

# B4：code-only；B0 + B2 校准器 + constrained refinement gate 的 test 评估。
bash scripts/train_p3.sh geometry-b4

# B5：唯一一次 FULL geometry，默认 18 epochs；结束后只用 validation 重拟合校准器。
bash scripts/train_p3.sh geometry-b5
```

B6 只有在 B5 的逐角误差、热图分辨率与 overlay 共同证明存在 quantization floor 后才进入。
当前 B6 stage 会明确返回 BLOCKED，因为本轮没有 B5 证据，也尚未引入 offset head 契约；不得通过
扩大 width 或无依据设置环境变量绕过这一门。

```bash
# 仅在后续 B5 分析明确批准 B6 后执行；当前预期 BLOCKED。
export P3_ENABLE_B6=1
bash scripts/train_p3.sh geometry-b6
unset P3_ENABLE_B6
```

## Experimental dewarp 与 faithful restoration

```bash
# D2 仅为 synthetic ABLATION，不阻塞 release。
bash scripts/train_p3.sh dewarp

# FidelityNet-v2：FULL，默认 30 epochs；photometric nuisance 在 input/target 中保持一致。
bash scripts/train_p3.sh fidelity

# 参数预测型 PhotometricNet：FULL，默认 24 epochs。
bash scripts/train_p3.sh photometric

# 若没有已审计真实 manifest，两个阶段自动完成 synthetic ABLATION，并写真实训练 BLOCKED JSON。
bash scripts/train_p3.sh demoire
bash scripts/train_p3.sh reflection

# 先评估 bicubic、P1 与可选 P3 checkpoint；本阶段不自动重训 SR。
bash scripts/train_p3.sh superres

# 小型 multi-label + severity Router：FULL，在线 degradation trace 监督。
bash scripts/train_p3.sh router
```

SR 只有评估证明 P1 明显失败后才单独设计 P3 重训。若已有受审计的候选 checkpoint，可在评估前
显式设置：

```bash
export P3_PIPELINE_SR_CHECKPOINT=/absolute/path/to/p3-superres/best.pt
bash scripts/train_p3.sh superres
unset P3_PIPELINE_SR_CHECKPOINT
```

## 最终评估与报告

```bash
bash scripts/train_p3.sh evaluate
bash scripts/train_p3.sh report
```

`evaluate` 优先读取 B5 ONNX 与 B5 validation 校准器；B5 不存在时读取 B0 与 B2 校准器。
机器报告为 `evaluation.json`、`slices.json`、`risk-coverage.json` 和 `release-gate.json`，FAIL
会原样保留。

## 外部真实专项数据

真实去摩尔纹数据当前缺失，期望目录为
`$SCREENRESTORE_DATA_ROOT/demoire/fhdmi`，或设置
`P3_DEMOIRE_MANIFEST=/absolute/path/to/audited.restoration.jsonl`。FHDMi 的许可、训练/商业限制与
最终保留量尚未审计；只允许从官方/作者渠道手工获取 thin/subset，预计保留量必须在 30 GiB
总上限内。缺失会阻塞真实 PSNR/SSIM/LPIPS、周期能量与 clean texture 结论，不阻塞 synthetic。

真实 reflection paired 数据当前缺失，期望目录为
`$SCREENRESTORE_DATA_ROOT/reflection/paired`，或设置
`P3_REFLECTION_MANIFEST=/absolute/path/to/audited.restoration.jsonl`。优先按
`docs/P3_DATA_CAPTURE_GUIDE.md` 本地采集约 1–2 GiB；SIR2/Real20/Nature 只有在来源、许可、
研究/商业限制和配准质量全部写入 P3 schema 后才可使用。缺失会阻塞真实单帧恢复与 unresolved
覆盖率结论，不阻塞 synthetic。

两类 manifest 均遵循 `datasets/schemas/restoration.schema.json`，并在训练前自动执行
`scripts/audit_restoration_manifest.py`。外部数据不会自动下载。
