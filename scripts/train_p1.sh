#!/usr/bin/env bash
# ScreenRestore P1 手动训练入口。
#
# 使用范例：
#   bash scripts/train_p1.sh smoke
#   bash scripts/train_p1.sh smoke --with-private-identity
#   bash scripts/train_p1.sh baseline --with-private-identity --with-wild-audit
#
# smoke 用少量样本检查数据、MPS、checkpoint 和 ONNX 导出；baseline 使用全部 SmartDoc
# 与 DIV2K 数据。private 无 GT 图片只有显式传入 --with-private-identity 时才参与，且仅用
# 于 Fidelity 的 identity 保护，绝不作为几何标签或恢复真值。DIV2K wild 是真实 x4 退化，
# --with-wild-audit 会纳入成对数据审计；它不会被错误混入当前同尺寸 Fidelity 训练。产物
# 写入 SCREENRESTORE_RUN_ROOT。

set -euo pipefail

MODE="${1:-smoke}"
shift || true
USE_PRIVATE_IDENTITY=false
USE_WILD_AUDIT=false
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SCREENRESTORE_DATA_ROOT:-$HOME/screenrestore-data}"
RUN_ROOT="${SCREENRESTORE_RUN_ROOT:-$HOME/screenrestore-runs}"
RUN_NAME="${SCREENRESTORE_RUN_NAME:-p1-$(date +%Y%m%d-%H%M%S)}"
RUN_DIRECTORY="$RUN_ROOT/$RUN_NAME"

if [[ "$MODE" != "smoke" && "$MODE" != "baseline" ]]; then
  echo "用法：bash scripts/train_p1.sh [smoke|baseline] [--with-private-identity] [--with-wild-audit]" >&2
  exit 2
fi
for option in "$@"; do
  case "$option" in
    --with-private-identity) USE_PRIVATE_IDENTITY=true ;;
    --with-wild-audit) USE_WILD_AUDIT=true ;;
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
  "$DATA_ROOT/superres/div2k/DIV2K_valid_HR"; do
  if [[ ! -e "$required" ]]; then
    echo "缺少训练数据：$required" >&2
    exit 1
  fi
done

source "$SCRIPT_ROOT/.venv/bin/activate"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"
export SCREENRESTORE_DATA_ROOT="$DATA_ROOT"
export SCREENRESTORE_RUN_ROOT="$RUN_ROOT"
mkdir -p "$RUN_DIRECTORY/geometry" "$RUN_DIRECTORY/restoration"

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
fi

PRIVATE_ARGS=()
if [[ "$USE_PRIVATE_IDENTITY" == "true" ]]; then
  if [[ ! -d "$DATA_ROOT/private" ]]; then
    echo "找不到显式授权的 private 目录：$DATA_ROOT/private" >&2
    exit 1
  fi
  PRIVATE_ARGS=(--private-identity-directory "$DATA_ROOT/private" --private-identity-weight 0.20)
fi

if [[ "$USE_WILD_AUDIT" == "true" ]]; then
  for required in \
    "$DATA_ROOT/superres/div2k/wild_x4/DIV2K_train_LR_wild" \
    "$DATA_ROOT/superres/div2k/wild_x4/DIV2K_valid_LR_wild"; do
    if [[ ! -d "$required" ]]; then
      echo "缺少 DIV2K wild x4 数据：$required" >&2
      exit 1
    fi
  done
  echo "wild x4 已接入清单：用于真实退化域差审计；当前同尺寸 Fidelity 不直接训练 x4 输出。"
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

echo "[$MODE] Fidelity：DIV2K 有监督恢复训练"
python -m training.restoration.train \
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
  --device auto \
  "${PRIVATE_ARGS[@]}"
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

echo "训练完成：$RUN_DIRECTORY"
if [[ "$USE_PRIVATE_IDENTITY" == "true" ]]; then
  echo "可额外运行无 GT 几何审计：python scripts/audit_unlabeled_geometry.py --image-directory \"$DATA_ROOT/private\" --quad-model \"$RUN_DIRECTORY/geometry/quadlocator-s.onnx\" --output \"$RUN_DIRECTORY/geometry/private-unlabeled-audit.json\""
fi
