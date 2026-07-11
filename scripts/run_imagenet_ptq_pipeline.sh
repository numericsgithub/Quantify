#!/bin/bash
# scripts/run_imagenet_ptq_qat_pipeline.sh
#
# Runs the full PTQ -> QAT pipeline in order:
#   1. find_perfect_lsbs_imagenet_ptq.py --mode weights
#   2. find_perfect_lsbs_imagenet_ptq.py --mode bias --init-from-ckpt <1's checkpoint>
#   3. find_perfect_lsbs_imagenet_ptq.py --mode activations --init-from-ckpt <2's checkpoint>
#   4. train_imagenet_qat.py --find-lr --init-from-ptq <3's checkpoint>
#   5. train_imagenet_qat.py --init-from-ptq <3's checkpoint> --lr <LR found in 4>
#
# Bias quantization (FixedPointPerTensorBiasQuant, requires_input_scale=False)
# calibrates directly against the bias values, with no dependency on
# activation quantization having run -- so it slots in between weights and
# activations rather than after them.
#
# Each of the 3 PTQ LSB search steps (weights/bias/activations) is skipped by
# default if its checkpoint already exists on disk. Set FORCE_LSB_SEARCH=1 to
# re-run them anyway.
#
# Override any setting via environment variables, e.g.:
#   MODEL=resnet50 WEIGHT_BITS=8 ACT_BITS=8 DATA_DIR=/data/imagenet \
#       ./scripts/run_imagenet_ptq_qat_pipeline.sh
#   FORCE_LSB_SEARCH=1 ./scripts/run_imagenet_ptq_qat_pipeline.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Config (override via env vars) ──────────────────────────────────────────
MODEL="${MODEL:-resnet18}"
NUM_CLASSES="${NUM_CLASSES:-1000}"
PRETRAINED="${PRETRAINED:-1}"     # 1 = pass --pretrained for the weights PTQ run
FUSE_BN="${FUSE_BN:-1}"           # 1 = pass --fuse-bn (must be consistent across PTQ runs)
FORCE_LSB_SEARCH="${FORCE_LSB_SEARCH:-0}"  # 1 = re-run a PTQ search step even if its checkpoint already exists

WEIGHT_BITS="${WEIGHT_BITS:-8}"
ACT_BITS="${ACT_BITS:-8}"
BIAS_BITS="${BIAS_BITS:-8}"
SEARCH_RADIUS="${SEARCH_RADIUS:-7}"
EVAL_BATCHES="${EVAL_BATCHES:-}"  # empty = full validation set per LSB candidate

DATA_DIR="${DATA_DIR:-/home/th/tmp/datasets/imagenet}"  # ImageFolder root; set empty to use --hf-dataset instead
HF_DATASET="${HF_DATASET:-ILSVRC/imagenet-1k}"
BATCH_SIZE_PTQ="${BATCH_SIZE_PTQ:-512}"
BATCH_SIZE_QAT="${BATCH_SIZE_QAT:-1024}"

EPOCHS="${EPOCHS:-150}"
FLOAT_WARMUP_EPOCHS="${FLOAT_WARMUP_EPOCHS:-0}"  # model is already PTQ-calibrated
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-100}"
QAT_GAP="${QAT_GAP:-100}"
ANNEALING_STEPS="${ANNEALING_STEPS:-10}"
REDUCE_LR_PATIENCE="${REDUCE_LR_PATIENCE:-20}"
REDUCE_LR_FACTOR="${REDUCE_LR_FACTOR:-0.5}"
REDUCE_LR_METRIC="${REDUCE_LR_METRIC:-val_loss}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"  # 0 = pass --no-mixed-precision

OUTPUT_DIR_PTQ="${OUTPUT_DIR_PTQ:-output/ptq_lsb_search}"
OUTPUT_DIR_QAT="${OUTPUT_DIR_QAT:-output/imagenet_qat}"
EXP_WEIGHTS="${EXP_WEIGHTS:-${MODEL}_weights_${WEIGHT_BITS}b}"
EXP_BIAS="${EXP_BIAS:-${MODEL}_bias_${BIAS_BITS}b}"
EXP_ACTS="${EXP_ACTS:-${MODEL}_activations_${ACT_BITS}b}"
EXP_QAT="${EXP_QAT:-${MODEL}_qat}"

PYTHON="${PYTHON:-/home/th/miniconda3/envs/brevitas-qat/bin/python}"

# ── Derived flags ────────────────────────────────────────────────────────────
PRETRAINED_FLAG=()
[[ "$PRETRAINED" == "1" ]] && PRETRAINED_FLAG=(--pretrained)
FUSE_BN_FLAG=()
[[ "$FUSE_BN" == "1" ]] && FUSE_BN_FLAG=(--fuse-bn)

DATA_FLAGS=()
if [[ -n "$DATA_DIR" ]]; then
    DATA_FLAGS=(--data-dir "$DATA_DIR")
else
    DATA_FLAGS=(--hf-dataset "$HF_DATASET")
fi

EVAL_BATCHES_FLAG=()
[[ -n "$EVAL_BATCHES" ]] && EVAL_BATCHES_FLAG=(--eval-batches "$EVAL_BATCHES")

MIXED_PRECISION_FLAG=(--no-mixed-precision)
[[ "$MIXED_PRECISION" == "1" ]] && MIXED_PRECISION_FLAG=()

# Returns success (skip) when $1's checkpoint already exists and a re-run
# wasn't forced via FORCE_LSB_SEARCH=1.
_ptq_ckpt_already_done() {
    [[ -f "$1" && "$FORCE_LSB_SEARCH" != "1" ]]
}

WEIGHTS_CKPT="${OUTPUT_DIR_PTQ}/${EXP_WEIGHTS}/ptq_calibrated_model.pt"
BIAS_CKPT="${OUTPUT_DIR_PTQ}/${EXP_BIAS}/ptq_calibrated_model.pt"
ACTS_CKPT="${OUTPUT_DIR_PTQ}/${EXP_ACTS}/ptq_calibrated_model.pt"

echo "================================================================"
echo "  PTQ -> QAT pipeline for ${MODEL}"
echo "  weight bits: ${WEIGHT_BITS}b   bias bits: ${BIAS_BITS}b   activation bits: ${ACT_BITS}b"
echo "================================================================"

# ── Step 1: PTQ weight LSB search ───────────────────────────────────────────
echo
if _ptq_ckpt_already_done "$WEIGHTS_CKPT"; then
    echo "[1/5] PTQ weight LSB search... SKIPPED (checkpoint exists: $WEIGHTS_CKPT)"
    echo "      Set FORCE_LSB_SEARCH=1 to re-run anyway."
else
    echo "[1/5] PTQ weight LSB search..."
    "$PYTHON" -m examples.find_perfect_lsbs_imagenet_ptq \
        --model "$MODEL" \
        --mode weights \
        --bit-width "$WEIGHT_BITS" \
        --num-classes "$NUM_CLASSES" \
        --search-radius "$SEARCH_RADIUS" \
        --batch-size "$BATCH_SIZE_PTQ" \
        --output-dir "$OUTPUT_DIR_PTQ" \
        --experiment-name "$EXP_WEIGHTS" \
        "${PRETRAINED_FLAG[@]}" "${FUSE_BN_FLAG[@]}" "${DATA_FLAGS[@]}" "${EVAL_BATCHES_FLAG[@]}"
fi

if [[ ! -f "$WEIGHTS_CKPT" ]]; then
    echo "[ERROR] Expected checkpoint not found: $WEIGHTS_CKPT" >&2
    exit 1
fi

# ── Step 2: PTQ bias LSB search, continuing from the weight checkpoint ──────
# --pretrained is intentionally omitted: the weight checkpoint already carries
# the pretrained-derived (and now weight-quantized) weights. Bias quantization
# calibrates directly against the bias values (requires_input_scale=False),
# so it doesn't need activations to be quantized first.
echo
if _ptq_ckpt_already_done "$BIAS_CKPT"; then
    echo "[2/5] PTQ bias LSB search... SKIPPED (checkpoint exists: $BIAS_CKPT)"
    echo "      Set FORCE_LSB_SEARCH=1 to re-run anyway."
else
    echo "[2/5] PTQ bias LSB search (continuing from weight-quantized model)..."
    "$PYTHON" -m examples.find_perfect_lsbs_imagenet_ptq \
        --model "$MODEL" \
        --mode bias \
        --bit-width "$BIAS_BITS" \
        --num-classes "$NUM_CLASSES" \
        --search-radius "$SEARCH_RADIUS" \
        --batch-size "$BATCH_SIZE_PTQ" \
        --output-dir "$OUTPUT_DIR_PTQ" \
        --experiment-name "$EXP_BIAS" \
        --init-from-ckpt "$WEIGHTS_CKPT" \
        "${FUSE_BN_FLAG[@]}" "${DATA_FLAGS[@]}" "${EVAL_BATCHES_FLAG[@]}"
fi

if [[ ! -f "$BIAS_CKPT" ]]; then
    echo "[ERROR] Expected checkpoint not found: $BIAS_CKPT" >&2
    exit 1
fi

# ── Step 3: PTQ activation LSB search, continuing from weight+bias checkpoint
echo
if _ptq_ckpt_already_done "$ACTS_CKPT"; then
    echo "[3/5] PTQ activation LSB search... SKIPPED (checkpoint exists: $ACTS_CKPT)"
    echo "      Set FORCE_LSB_SEARCH=1 to re-run anyway."
else
    echo "[3/5] PTQ activation LSB search (continuing from weight+bias-quantized model)..."
    "$PYTHON" -m examples.find_perfect_lsbs_imagenet_ptq \
        --model "$MODEL" \
        --mode activations \
        --bit-width "$ACT_BITS" \
        --num-classes "$NUM_CLASSES" \
        --search-radius "$SEARCH_RADIUS" \
        --batch-size "$BATCH_SIZE_PTQ" \
        --output-dir "$OUTPUT_DIR_PTQ" \
        --experiment-name "$EXP_ACTS" \
        --init-from-ckpt "$BIAS_CKPT" \
        "${FUSE_BN_FLAG[@]}" "${DATA_FLAGS[@]}" "${EVAL_BATCHES_FLAG[@]}"
fi

if [[ ! -f "$ACTS_CKPT" ]]; then
    echo "[ERROR] Expected checkpoint not found: $ACTS_CKPT" >&2
    exit 1
fi
