#!/usr/bin/env bash
# ScreenRestore P1 手动训练入口。
#
# 使用范例：
#   bash scripts/train_p1.sh smoke --with-private-identity
#   bash scripts/train_p1.sh full --with-private-identity
#
# smoke 用少量样本检查数据、MPS、checkpoint 和 ONNX 导出；full 依次使用全部 SmartDoc、
# DIV2K HR、x2 bicubic 与 wild x4。private 无 GT 图片只有显式传入 --with-private-identity
# 时参与，且仅用于 Fidelity identity 保护。x2 与 wild x4 始终分别训练独立超分模型。产物写
# 入 SCREENRESTORE_RUN_ROOT。

set -euo pipefail

MODE="${1:-full}"
shift || true
USE_PRIVATE_IDENTITY=false
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SCREENRESTORE_DATA_ROOT:-$HOME/screenrestore-data}"
RUN_ROOT="${SCREENRESTORE_RUN_ROOT:-$HOME/screenrestore-runs}"
RUN_NAME="${SCREENRESTORE_RUN_NAME:-p1-$(date +%Y%m%d-%H%M%S)}"
RUN_DIRECTORY="$RUN_ROOT/$RUN_NAME"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "用法：bash scripts/train_p1.sh [smoke|full] [--with-private-identity]" >&2
  exit 2
fi
for option in "$@"; do
  case "$option" in
    --with-private-identity) USE_PRIVATE_IDENTITY=true ;;
    *) echo "未知选项：$option" >&2; exit 2 ;;
  esac
done
if [[ ! -x "$SCRIPT_ROOT/.venv/bin/python" ]]; then
  echo "未找到 Python 3.11 虚拟环境：$SCRIPT_ROOT/.venv/bin/python" >&2
  exit 1
fi
for required in \
  "$DATA_ROOT/manifests/smartdoc.geometry.jsonl" \
  "$DATA_ROOT/manifests/div2k.restoration.jsonl" \
  "$DATA_ROOT/geometry/smartdoc/frames" \
  "$DATA_ROOT/superres/div2k/DIV2K_train_HR" \
  "$DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
  "$DATA_ROOT/superres/div2k/DIV2K_train_LR_bicubic/X2" \
  "$DATA_ROOT/superres/div2k/DIV2K_valid_LR_bicubic/X2" \
  "$DATA_ROOT/superres/div2k/wild_x4/DIV2K_train_LR_wild" \
  "$DATA_ROOT/superres/div2k/wild_x4/DIV2K_valid_LR_wild"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少训练数据：$required" >&2
    exit 1
  fi
done

source "$SCRIPT_ROOT/.venv/bin/activate"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"
export SCREENRESTORE_DATA_ROOT="$DATA_ROOT"
export SCREENRESTORE_RUN_ROOT="$RUN_ROOT"
mkdir -p "$RUN_DIRECTORY/geometry" "$RUN_DIRECTORY/restoration" "$RUN_DIRECTORY/superres-x2" "$RUN_DIRECTORY/superres-wild-x4"

if [[ "$MODE" == "smoke" ]]; then
  GEOMETRY_EPOCHS=1
  GEOMETRY_TRAIN_SAMPLES=480
  GEOMETRY_VALIDATION_SAMPLES=96
  GEOMETRY_IMAGE_SIZE=320
  GEOMETRY_WIDTH=0.5
  GEOMETRY_BATCH=4
  RESTORATION_EPOCHS=1
  RESTORATION_TRAIN_SAMPLES=200
  RESTORATION_VALIDATION_SAMPLES=50
  RESTORATION_PATCH=128
  RESTORATION_CHANNELS=16
  RESTORATION_BLOCKS=4
  RESTORATION_BATCH=4
  SUPERRES_EPOCHS=1
  SUPERRES_TRAIN_SAMPLES=200
  SUPERRES_VALIDATION_SAMPLES=50
  SUPERRES_PATCH=128
  SUPERRES_CHANNELS=16
  SUPERRES_BLOCKS=4
  SUPERRES_BATCH=4
else
  GEOMETRY_EPOCHS=12
  GEOMETRY_TRAIN_SAMPLES=0
  GEOMETRY_VALIDATION_SAMPLES=0
  GEOMETRY_IMAGE_SIZE=512
  GEOMETRY_WIDTH=1.0
  GEOMETRY_BATCH=4
  RESTORATION_EPOCHS=20
  RESTORATION_TRAIN_SAMPLES=0
  RESTORATION_VALIDATION_SAMPLES=0
  RESTORATION_PATCH=192
  RESTORATION_CHANNELS=32
  RESTORATION_BLOCKS=6
  RESTORATION_BATCH=8
  SUPERRES_EPOCHS=16
  SUPERRES_TRAIN_SAMPLES=0
  SUPERRES_VALIDATION_SAMPLES=0
  SUPERRES_PATCH=192
  SUPERRES_CHANNELS=32
  SUPERRES_BLOCKS=6
  SUPERRES_BATCH=8
fi

if [[ "$USE_PRIVATE_IDENTITY" == "true" ]]; then
  if [[ ! -d "$DATA_ROOT/private" ]]; then
    echo "找不到显式授权的 private 目录：$DATA_ROOT/private" >&2
    exit 1
  fi
fi

echo "[$MODE] QuadLocator-S：SmartDoc 有监督几何训练"
python -m training.quadlocator.train \
  --manifest "$DATA_ROOT/manifests/smartdoc.geometry.jsonl" \
  --dataset-root "$DATA_ROOT" \
  --output-directory "$RUN_DIRECTORY/geometry" \
  --epochs "$GEOMETRY_EPOCHS" \
  --train-samples "$GEOMETRY_TRAIN_SAMPLES" \
  --validation-samples "$GEOMETRY_VALIDATION_SAMPLES" \
  --image-size "$GEOMETRY_IMAGE_SIZE" \
  --width-multiplier "$GEOMETRY_WIDTH" \
  --batch-size "$GEOMETRY_BATCH" \
  --device auto
python -m training.quadlocator.export_onnx \
  --checkpoint "$RUN_DIRECTORY/geometry/best.pt" \
  --output "$RUN_DIRECTORY/geometry/quadlocator-s.onnx"

echo "[$MODE] Fidelity：DIV2K HR 在线退化的同尺寸恢复训练"
RESTORATION_COMMAND=(python -m training.restoration.train \
  --train-hr-directory "$DATA_ROOT/superres/div2k/DIV2K_train_HR" \
  --validation-hr-directory "$DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
  --output-directory "$RUN_DIRECTORY/restoration" \
  --epochs "$RESTORATION_EPOCHS" \
  --train-samples "$RESTORATION_TRAIN_SAMPLES" \
  --validation-samples "$RESTORATION_VALIDATION_SAMPLES" \
  --patch-size "$RESTORATION_PATCH" \
  --channels "$RESTORATION_CHANNELS" \
  --blocks "$RESTORATION_BLOCKS" \
  --batch-size "$RESTORATION_BATCH" \
  --device auto)
if [[ "$USE_PRIVATE_IDENTITY" == "true" ]]; then
  RESTORATION_COMMAND+=(--private-identity-directory "$DATA_ROOT/private" --private-identity-weight 0.20)
fi
"${RESTORATION_COMMAND[@]}"
python -m training.restoration.evaluate \
  --checkpoint "$RUN_DIRECTORY/restoration/best.pt" \
  --hr-directory "$DATA_ROOT/superres/div2k/DIV2K_valid_HR" \
  --output "$RUN_DIRECTORY/restoration/evaluation.json" \
  --samples "$RESTORATION_VALIDATION_SAMPLES" \
  --batch-size "$RESTORATION_BATCH" \
  --device auto
python -m training.restoration.export_onnx \
  --checkpoint "$RUN_DIRECTORY/restoration/best.pt" \
  --output "$RUN_DIRECTORY/restoration/fidelity-residual.onnx"

for variant in x2 wild_x4; do
  if [[ "$variant" == "x2" ]]; then
    stage_directory="$RUN_DIRECTORY/superres-x2"
  else
    stage_directory="$RUN_DIRECTORY/superres-wild-x4"
  fi
  echo "[$MODE] Conservative SR：DIV2K $variant 独立监督训练"
  python -m training.superres.train \
    --manifest "$DATA_ROOT/manifests/div2k.restoration.jsonl" \
    --data-root "$DATA_ROOT" \
    --variant "$variant" \
    --output-directory "$stage_directory" \
    --epochs "$SUPERRES_EPOCHS" \
    --train-samples "$SUPERRES_TRAIN_SAMPLES" \
    --validation-samples "$SUPERRES_VALIDATION_SAMPLES" \
    --patch-size "$SUPERRES_PATCH" \
    --channels "$SUPERRES_CHANNELS" \
    --blocks "$SUPERRES_BLOCKS" \
    --batch-size "$SUPERRES_BATCH" \
    --device auto
  python -m training.superres.export_onnx \
    --checkpoint "$stage_directory/best.pt" \
    --output "$stage_directory/conservative-superres.onnx"
done

echo "训练完成：$RUN_DIRECTORY"
if [[ "$USE_PRIVATE_IDENTITY" == "true" ]]; then
  echo "可额外运行无 GT 几何审计：python scripts/audit_unlabeled_geometry.py --image-directory \"$DATA_ROOT/private\" --quad-model \"$RUN_DIRECTORY/geometry/quadlocator-s.onnx\" --output \"$RUN_DIRECTORY/geometry/private-unlabeled-audit.json\""
fi
