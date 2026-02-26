#!/bin/bash
# Korean Essay Rater 학습 실행 스크립트
# Usage: bash run_train.sh [OPTIONS]
#
# 예시:
#   bash run_train.sh                    # 기본 설정으로 학습
#   bash run_train.sh --dry_run          # 데이터/모델 로딩만 테스트
#   bash run_train.sh --no_wandb         # WandB 없이 학습
#   bash run_train.sh --epochs 4         # 에폭 수 변경

set -e

# ─── 기본 설정 ───────────────────────────────────
DEVICE_ID=0
BASE_MODEL="/home/khko/models/exaone"
DATA_DIR="./data_precomputed"
EPOCHS=8
BATCH_SIZE=4
GRAD_ACCUM=8
LR=2e-4
MAX_SEQ_LENGTH=1536

# ─── 프로젝트 디렉토리로 이동 ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " Korean Essay Rater - Training"
echo "============================================"
echo " Model     : $BASE_MODEL"
echo " Data      : $DATA_DIR"
echo " Device    : GPU $DEVICE_ID"
echo " Epochs    : $EPOCHS"
echo " Batch     : ${BATCH_SIZE} x ${GRAD_ACCUM} (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo " LR        : $LR"
echo " Max Seq   : $MAX_SEQ_LENGTH"
echo " Optimizer : paged_adamw_8bit"
echo " Packing   : ON"
echo " Compile   : OFF (QLoRA 비호환)"
echo "============================================"

# ─── 전처리 데이터 확인 ──────────────────────────
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "[INFO] 사전 추출 데이터가 없습니다. 전처리를 먼저 실행합니다..."
    python preprocess_data.py --input_dir ./data --output_dir ./data_precomputed
    echo "[INFO] 전처리 완료."
fi

# ─── 학습 실행 ───────────────────────────────────
python train.py \
    --device_id "$DEVICE_ID" \
    --base_model_name "$BASE_MODEL" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --gradient_checkpointing \
    --pack_sequences \
    "$@"

echo ""
echo "============================================"
echo " 학습 완료!"
echo "============================================"
