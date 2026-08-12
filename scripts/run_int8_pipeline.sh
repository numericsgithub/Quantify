#!/usr/bin/env bash
#
# run_int8_pipeline.sh — full W8/A8/B8 QAT pipeline for ONE model.
#
# Reproduces the MobileNetV2 procedure documented in
# docs/llm/skills/int8_qat_training_pipeline.md:
#
#   Phase 0  PTQ checkpoint (weights+bias, single forward)     ~minutes
#   Phase 1  weights+bias QAT chain (activations float)        10x40 epochs
#   Phase 2  progressive activation introduction (annealed)    130 epochs
#   Phase 3  post-activation QAT chain (full quant, polish)    10x40 epochs
#
# Each phase hard-fails if the previous produced no usable checkpoint, so a
# broken stage stops the pipeline instead of silently training on nothing.
#
# Required env:
#   MODEL   e.g. mobilenetv1 | resnet18 | resnet50
#   BATCH   1024 (mobilenets) | 512 (resnets)
#   GAP     Phase-2 qat_gap, tuned per model+batch so the activation ramp ends
#           ~2/3 through the 130-epoch run (see the table in the doc):
#             mobilenetv1 b1024 -> 2080   resnet18 b512 -> 4500
#             resnet50    b512  -> 1790   mobilenetv2 b1024 -> 800
#
# Optional env: DATA_DIR, PYTHON, OUT_BASE, WLRS, ALRS, ACT_EPOCHS, DRY_RUN,
#               START_PHASE (0..3, resume the pipeline at a later phase).

set -uo pipefail

MODEL="${MODEL:?set MODEL (mobilenetv1|resnet18|resnet50|mobilenetv2)}"
BATCH="${BATCH:?set BATCH (1024 for mobilenets, 512 for resnets)}"
GAP="${GAP:?set GAP (Phase-2 qat_gap; see doc table)}"

DATA_DIR="${DATA_DIR:-/home/th/tmp/datasets/imagenet}"
PYTHON="${PYTHON:-/home/th/miniconda3/envs/brevitas-qat/bin/python}"
OUT_BASE="${OUT_BASE:-output/pipeline_${MODEL}}"
ACT_EPOCHS="${ACT_EPOCHS:-130}"
# Per-model weight-LSB k. Empty = built-in default (12). MobileNetV1 needs ~20
# (its depthwise layers collapse under per-tensor k=12); see pitfall #15.
WEIGHT_SIGMA_K="${WEIGHT_SIGMA_K:-}"
WLRS="${WLRS:-1e-7 1e-6 1e-5 1e-6 1e-7 1e-8 1e-7 1e-6 1e-7 1e-8}"
ALRS="${ALRS:-2e-7 2e-6 1e-5 2e-6 2e-7 2e-8 2e-7 2e-6 2e-7 2e-8}"
START_PHASE="${START_PHASE:-0}"
DRY_RUN="${DRY_RUN:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_BASE"
PLOG="$OUT_BASE/pipeline.log"
log() { echo "[pipe:$MODEL $(date '+%F %T')] $*" | tee -a "$PLOG"; }

PTQ_CKPT="output/ptq/${MODEL}_W8_Anone_B8_singlepass.pt"
WCHAIN_DIR="$OUT_BASE/wchain"
ACT_DIR="$OUT_BASE/act_intro"
ACHAIN_DIR="$OUT_BASE/actchain"

# Best best.pt across a chain's run dirs (monotonic chain => last run's best is
# global best, but scan to be safe). Prints a path or empty string.
best_in_chain() {
    "$PYTHON" - "$1" <<'PY'
import glob, sys, torch
root = sys.argv[1]; best = None
for p in sorted(glob.glob(root + "/**/best.pt", recursive=True)):
    try:
        va = (torch.load(p, map_location="cpu", weights_only=False).get("metrics") or {}).get("val_acc")
        if va is not None and (best is None or va > best[0]):
            best = (va, p)
    except Exception:
        pass
print(best[1] if best else "")
PY
}
acc_of() { "$PYTHON" - "$1" <<'PY'
import sys, torch
try:
    m = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("metrics") or {}
    print(f"{m.get('val_acc')}")
except Exception as e:
    print("?")
PY
}

run() { if [[ "$DRY_RUN" == "1" ]]; then log "DRY: $*"; else "$@"; fi; }

log "=========================================================================="
log "FULL INT8 PIPELINE  model=$MODEL  batch=$BATCH  gap=$GAP  start_phase=$START_PHASE"
log "  out=$OUT_BASE"
log "=========================================================================="

# ---- Phase 0: PTQ checkpoint ---------------------------------------------
if (( START_PHASE <= 0 )); then
    log "PHASE 0: PTQ checkpoint -> $PTQ_CKPT  (weight_sigma_k=${WEIGHT_SIGMA_K:-default})"
    P0=( "$PYTHON" -u -m examples.create_ptq_checkpoint
         --model "$MODEL" --data-dir "$DATA_DIR" --batch-size 128 )
    [[ -n "$WEIGHT_SIGMA_K" ]] && P0+=( --weight-sigma-k "$WEIGHT_SIGMA_K" )
    run "${P0[@]}" 2>&1 | tee -a "$PLOG"
    if [[ "$DRY_RUN" != "1" && ! -f "$PTQ_CKPT" ]]; then
        log "FATAL: Phase 0 produced no $PTQ_CKPT"; exit 1
    fi
fi

# ---- Phase 1: weights+bias chain -----------------------------------------
if (( START_PHASE <= 1 )); then
    log "PHASE 1: weights+bias chain -> $WCHAIN_DIR"
    run env MODEL="$MODEL" BATCH="$BATCH" PTQ_CKPT="$PTQ_CKPT" OUT_ROOT="$WCHAIN_DIR" \
        LRS="$WLRS" DATA_DIR="$DATA_DIR" DRY_RUN="$DRY_RUN" WSIGMA="$WEIGHT_SIGMA_K" \
        bash "$HERE/run_mnv2_qat_chain.sh" 2>&1 | tee -a "$PLOG"
fi
if [[ "$DRY_RUN" != "1" ]]; then
    P1_BEST="$(best_in_chain "$WCHAIN_DIR")"
    [[ -z "$P1_BEST" ]] && { log "FATAL: no best.pt in $WCHAIN_DIR"; exit 1; }
    log "Phase 1 best = $P1_BEST  (val_acc=$(acc_of "$P1_BEST"))"
else
    P1_BEST="$WCHAIN_DIR/run10/checkpoints/best.pt"
fi

# ---- Phase 2: progressive activation introduction ------------------------
if (( START_PHASE <= 2 )); then
    log "PHASE 2: activation introduction (gap=$GAP, anneal=100, ${ACT_EPOCHS} ep) -> $ACT_DIR"
    P2=( "$PYTHON" -u -m examples.train_imagenet_qat
         --data-dir "$DATA_DIR" --model "$MODEL"
         --init-from-ptq "$P1_BEST"
         --batch-size "$BATCH"
         --qat-gap "$GAP" --annealing-steps 100
         --no-seed-best --require-full-quant-for-best
         --lr 2e-8 --weight-decay 0 --epochs "$ACT_EPOCHS"
         --float-warmup-epochs 0 --no-mixed-precision
         --output-dir "$ACT_DIR" )
    [[ -n "$WEIGHT_SIGMA_K" ]] && P2+=( --weight-sigma-k "$WEIGHT_SIGMA_K" )
    run "${P2[@]}" 2>&1 | tee -a "$PLOG"
fi
if [[ "$DRY_RUN" != "1" ]]; then
    P2_BEST="$ACT_DIR/checkpoints/best.pt"
    [[ ! -f "$P2_BEST" ]] && {
        log "FATAL: Phase 2 wrote no $P2_BEST (no epoch reached full quantization —"
        log "       ramp too slow for ${ACT_EPOCHS} epochs? check GAP vs steps/epoch)"; exit 1; }
    log "Phase 2 best = $P2_BEST  (val_acc=$(acc_of "$P2_BEST"))"
else
    P2_BEST="$ACT_DIR/checkpoints/best.pt"
fi

# ---- Phase 3: post-activation chain --------------------------------------
if (( START_PHASE <= 3 )); then
    log "PHASE 3: post-activation chain (ACT_QUANT=1) -> $ACHAIN_DIR"
    run env ACT_QUANT=1 MODEL="$MODEL" BATCH="$BATCH" PTQ_CKPT="$P2_BEST" OUT_ROOT="$ACHAIN_DIR" \
        LRS="$ALRS" DATA_DIR="$DATA_DIR" DRY_RUN="$DRY_RUN" WSIGMA="$WEIGHT_SIGMA_K" \
        bash "$HERE/run_mnv2_qat_chain.sh" 2>&1 | tee -a "$PLOG"
fi
if [[ "$DRY_RUN" != "1" ]]; then
    P3_BEST="$(best_in_chain "$ACHAIN_DIR")"
    [[ -z "$P3_BEST" ]] && { log "FATAL: no best.pt in $ACHAIN_DIR"; exit 1; }
    log "=========================================================================="
    log "DONE $MODEL. Final W8/A8/B8 model: $P3_BEST  (val_acc=$(acc_of "$P3_BEST"))"
    log "=========================================================================="
fi
