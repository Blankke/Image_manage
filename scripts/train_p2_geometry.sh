#!/usr/bin/env bash
# ScreenRestore P2 geometry 分阶段手动训练入口；不包含 Fidelity 或 super-resolution。
#
# 使用范例：
#   export SCREENRESTORE_DATA_ROOT="$HOME/screenrestore-data"
#   export SCREENRESTORE_RUN_ROOT="$HOME/screenrestore-runs"
#   export SCREENRESTORE_RUN_NAME="p2-geometry-w1-$(date +%Y%m%d-%H%M%S)"
#   export P2_DEVICE=mps  # Apple Silicon 正式训练显式失败，禁止静默回落 CPU
#   bash scripts/train_p2_geometry.sh preflight
#   bash scripts/train_p2_geometry.sh stage-a
#   bash scripts/train_p2_geometry.sh stage-b
#   bash scripts/train_p2_geometry.sh stage-c
#   bash scripts/train_p2_geometry.sh stage-d
#
# 对 width_multiplier=1.5 做独立实验时，使用新的 RUN_NAME 并设置 P2_WIDTH=1.5，
# 从 stage-a 重新开始。每个 stage 训练完成后自动导出 ONNX、运行诊断 benchmark 并生成
# public validation 50 张及 private validation/test 全量 overlay。benchmark FAIL 不会伪装为成功。

set -euo pipefail

STAGE="${1:-}"
if [[ "$STAGE" != "preflight" && "$STAGE" != "stage-a" && "$STAGE" != "stage-b" && "$STAGE" != "stage-c" && "$STAGE" != "stage-d" && "$STAGE" != "all" ]]; then
  echo "用法：bash scripts/train_p2_geometry.sh [preflight|stage-a|stage-b|stage-c|stage-d|all]" >&2
  exit 2
fi

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SCREENRESTORE_DATA_ROOT:-$HOME/screenrestore-data}"
RUN_ROOT="${SCREENRESTORE_RUN_ROOT:-$HOME/screenrestore-runs}"
RUN_NAME="${SCREENRESTORE_RUN_NAME:-p2-geometry-manual}"
RUN_DIRECTORY="$RUN_ROOT/$RUN_NAME"
P1_CHECKPOINT="${P1_GEOMETRY_CHECKPOINT:-/Users/caozichen/screenrestore-runs/p1-full-20260828-133513/geometry/best.pt}"
WIDTH="${P2_WIDTH:-1.0}"
IMAGE_SIZE="${P2_IMAGE_SIZE:-512}"
BATCH_SIZE="${P2_BATCH_SIZE:-4}"
WORKERS="${P2_WORKERS:-0}"
DEVICE="${P2_DEVICE:-auto}"
SEED="${P2_SEED:-20260828}"

if [[ ! -x "$SCRIPT_ROOT/.venv/bin/python" ]]; then
  echo "未找到项目 Python 3.11 虚拟环境：$SCRIPT_ROOT/.venv/bin/python" >&2
  exit 1
fi
source "$SCRIPT_ROOT/.venv/bin/activate"
which python
python -c 'import sys; print(sys.executable)'
export XDG_STATE_HOME="${XDG_STATE_HOME:-/tmp/screenrestore-state}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "缺少必需输入：$1" >&2
    exit 1
  fi
}

ensure_fresh_stage() {
  if [[ -e "$1/best.pt" && "${P2_ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "stage 已存在，拒绝覆盖历史 checkpoint：$1（如确认重跑，显式设置 P2_ALLOW_OVERWRITE=1）" >&2
    exit 1
  fi
}

preflight_inputs() {
  require_path "$P1_CHECKPOINT"
  require_path "$DATA_ROOT/manifests/p2/stage-a.geometry.jsonl"
  require_path "$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl"
  require_path "$DATA_ROOT/manifests/p2/stage-c.geometry.jsonl"
  require_path "$DATA_ROOT/manifests/p2/calibration.geometry.jsonl"
  require_path "$DATA_ROOT/manifests/smartdoc.geometry.jsonl"
  require_path "$DATA_ROOT/private/geometry.annotations.jsonl"

  local data_kib free_kib
  data_kib="$(du -sk "$DATA_ROOT" | awk '{print $1}')"
  free_kib="$(df -Pk "$DATA_ROOT" | awk 'NR == 2 {print $4}')"
  if (( data_kib > 30 * 1024 * 1024 )); then
    echo "数据根超过 30 GiB 硬限制：${DATA_ROOT}（${data_kib} KiB）" >&2
    exit 1
  fi
  if (( free_kib < 10 * 1024 * 1024 )); then
    echo "可用空间不足 10 GiB：${DATA_ROOT}（${free_kib} KiB）" >&2
    exit 1
  fi

  python - "$DEVICE" <<'PY'
import sys

from training.quadlocator.train import _device

selected = _device(sys.argv[1])
print(f"preflight device={selected}")
PY

  # 只解析清单与检查相对图片路径，不创建模型、optimizer 或 checkpoint。
  python - "$DATA_ROOT" \
    "$DATA_ROOT/manifests/p2/stage-a.geometry.jsonl" \
    "$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl" \
    "$DATA_ROOT/manifests/p2/stage-c.geometry.jsonl" \
    "$DATA_ROOT/manifests/p2/calibration.geometry.jsonl" <<'PY'
import sys
from collections import Counter
from pathlib import Path

from training.quadlocator.dataset import _read_manifest

data_root = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    manifest = Path(value).resolve()
    records = _read_manifest(manifest)
    missing = [record["image"] for record in records if not (data_root / record["image"]).is_file()]
    if missing:
        raise SystemExit(f"{manifest.name} 有 {len(missing)} 条图片路径不存在")
    splits = Counter(str(record["split"]) for record in records)
    if manifest.name != "calibration.geometry.jsonl" and not {"train", "validation"} <= splits.keys():
        raise SystemExit(f"{manifest.name} 缺少 train 或 validation")
    if manifest.name == "calibration.geometry.jsonl" and set(splits) != {"validation"}:
        raise SystemExit("calibration 清单只能包含 validation")
    print(f"preflight {manifest.name}: samples={len(records)} splits={dict(sorted(splits.items()))}")
PY
  echo "preflight 通过：data=${data_kib} KiB，free=${free_kib} KiB，device=$DEVICE" >&2
}

render_stage_overlays() {
  local stage_directory="$1"
  local public_manifest="$2"
  local model="$stage_directory/quadlocator-s.onnx"
  python scripts/render_geometry_overlays.py \
    --manifest "$public_manifest" \
    --dataset-root "$DATA_ROOT" \
    --split validation \
    --max-images 50 \
    --quad-model "$model" \
    --output-directory "$stage_directory/overlays-public-validation"
  if [[ -f "$DATA_ROOT/private/geometry.annotations.jsonl" ]]; then
    for split in validation test; do
      python scripts/render_geometry_overlays.py \
        --manifest "$DATA_ROOT/private/geometry.annotations.jsonl" \
        --dataset-root "$DATA_ROOT" \
        --split "$split" \
        --max-images 0 \
        --quad-model "$model" \
        --output-directory "$stage_directory/overlays-private-$split"
    done
  fi
}

evaluate_smartdoc() {
  local stage_directory="$1"
  set +e
  python -m benchmarks.geometry_e2e.run \
    --data-directory "$DATA_ROOT/geometry/smartdoc/frames" \
    --manifest "$DATA_ROOT/manifests/smartdoc.geometry.jsonl" \
    --dataset-root "$DATA_ROOT" \
    --split test \
    --quad-model "$stage_directory/quadlocator-s.onnx" \
    --output "$stage_directory/smartdoc-test-e2e.json"
  local benchmark_status=$?
  set -e
  echo "SmartDoc gate exit=${benchmark_status}（报告始终保留，FAIL 不阻断后续人工检查）" >&2
}

run_stage_a() {
  require_path "$P1_CHECKPOINT"
  require_path "$DATA_ROOT/manifests/p2/stage-a.geometry.jsonl"
  local output="$RUN_DIRECTORY/stage-a"
  ensure_fresh_stage "$output"
  python -m training.quadlocator.train \
    --manifest "$DATA_ROOT/manifests/p2/stage-a.geometry.jsonl" \
    --dataset-root "$DATA_ROOT" \
    --output-directory "$output" \
    --init-checkpoint "$P1_CHECKPOINT" \
    --epochs 3 \
    --learning-rate 3e-4 \
    --image-size "$IMAGE_SIZE" \
    --width-multiplier "$WIDTH" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
  python -m training.quadlocator.export_onnx \
    --checkpoint "$output/best.pt" \
    --output "$output/quadlocator-s.onnx"
  evaluate_smartdoc "$output"
  render_stage_overlays "$output" "$DATA_ROOT/manifests/p2/stage-a.geometry.jsonl"
}

run_stage_b() {
  require_path "$RUN_DIRECTORY/stage-a/best.pt"
  require_path "$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl"
  local output="$RUN_DIRECTORY/stage-b"
  ensure_fresh_stage "$output"
  python -m training.quadlocator.train \
    --manifest "$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl" \
    --dataset-root "$DATA_ROOT" \
    --output-directory "$output" \
    --init-checkpoint "$RUN_DIRECTORY/stage-a/best.pt" \
    --epochs 18 \
    --early-stopping-patience 5 \
    --learning-rate 2e-4 \
    --image-size "$IMAGE_SIZE" \
    --width-multiplier "$WIDTH" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
  python -m training.quadlocator.export_onnx \
    --checkpoint "$output/best.pt" \
    --output "$output/quadlocator-s.onnx"
  evaluate_smartdoc "$output"
  render_stage_overlays "$output" "$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl"
}

run_stage_c() {
  require_path "$RUN_DIRECTORY/stage-b/best.pt"
  require_path "$DATA_ROOT/manifests/p2/stage-c.geometry.jsonl"
  local output="$RUN_DIRECTORY/stage-c"
  ensure_fresh_stage "$output"
  python -m training.quadlocator.train \
    --manifest "$DATA_ROOT/manifests/p2/stage-c.geometry.jsonl" \
    --dataset-root "$DATA_ROOT" \
    --output-directory "$output" \
    --init-checkpoint "$RUN_DIRECTORY/stage-b/best.pt" \
    --epochs 8 \
    --early-stopping-patience 3 \
    --learning-rate 7.5e-5 \
    --image-size "$IMAGE_SIZE" \
    --width-multiplier "$WIDTH" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED"
  python -m training.quadlocator.export_onnx \
    --checkpoint "$output/best.pt" \
    --output "$output/quadlocator-s.onnx"
  evaluate_smartdoc "$output"
  render_stage_overlays "$output" "$DATA_ROOT/manifests/p2/stage-b.geometry.jsonl"
}

run_stage_d() {
  require_path "$RUN_DIRECTORY/stage-c/best.pt"
  require_path "$DATA_ROOT/manifests/p2/calibration.geometry.jsonl"
  mkdir -p "$RUN_DIRECTORY/stage-d"
  python -m training.quadlocator.calibrate \
    --checkpoint "$RUN_DIRECTORY/stage-c/best.pt" \
    --manifest "$DATA_ROOT/manifests/p2/calibration.geometry.jsonl" \
    --dataset-root "$DATA_ROOT" \
    --output "$RUN_DIRECTORY/stage-d/calibration.json" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --minimum-precision 0.99
}

case "$STAGE" in
  preflight) preflight_inputs ;;
  stage-a) run_stage_a ;;
  stage-b) run_stage_b ;;
  stage-c) run_stage_c ;;
  stage-d) run_stage_d ;;
  all)
    preflight_inputs
    run_stage_a
    run_stage_b
    run_stage_c
    run_stage_d
    ;;
esac

echo "P2 geometry $STAGE 完成：$RUN_DIRECTORY" >&2
