#!/usr/bin/env bash
#
# run_mnv2_act_then_chain.sh — two phases, hands-off:
#
#   PHASE 1  Activation-introduction run (single process, 130 epochs).
#            Starts from the best weights+bias-only checkpoint (72.30%) and
#            introduces the 35 activation quantizers GRADUALLY, input-side first:
#            quantizer with execution-order id K waits K*gap forward passes, then
#            calibrates itself and anneals 0->1 over 100 steps. Weights & biases
#            are preserved (active from step 0). With gap=800 the last activation
#            comes on around epoch 88, so the model is fully quantized for the
#            final ~42 epochs.
#
#            best.pt is guarded two ways for this phase:
#              --no-seed-best              : do NOT seed best.pt at the partial
#                                            72.30% start (that score is
#                                            unbeatable once activations quantize).
#              --require-full-quant-for-best : only epochs with EVERY quantizer
#                                            fully quantized can win best.pt, so a
#                                            high-scoring partial epoch mid-ramp
#                                            cannot be picked.
#            Accuracy is EXPECTED to dip below 72.30% — that is fine.
#
#   PHASE 2  QAT chain (10 runs x 40 epochs) WITH activation quantization,
#            seeded from Phase 1's fully-quantized best.pt. Same learning rates
#            as the previous chain. Everything is already calibrated, so each run
#            is fully quantized from step 0 (ACT_QUANT=1 -> gap 0 / anneal 1).
#
# Phase 2 only starts if Phase 1 produced a fully-quantized best.pt. If Phase 1
# is interrupted before the model ever reaches full quantization (~epoch 88),
# there is no best.pt and the script stops rather than chaining from nothing.
#
# Usage:
#   scripts/run_mnv2_act_then_chain.sh              # run both phases
#   SKIP_PHASE1=1 scripts/run_mnv2_act_then_chain.sh  # Phase 1 already done, chain only
#   DRY_RUN=1 scripts/run_mnv2_act_then_chain.sh    # print commands only

set -uo pipefail

# ---- config ---------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/home/th/tmp/datasets/imagenet}"
SEED_CKPT="${SEED_CKPT:-output/ptq/mnv2_chain2_best_7230.pt}"   # 72.30% weights+bias
PYTHON="${PYTHON:-/home/th/miniconda3/envs/brevitas-qat/bin/python}"
MODEL="${MODEL:-mobilenetv2}"

# Phase 1 (activation introduction)
ACT_INTRO_DIR="${ACT_INTRO_DIR:-output/mnv2_act_intro}"
ACT_EPOCHS="${ACT_EPOCHS:-130}"
ACT_GAP="${ACT_GAP:-800}"
ACT_ANNEAL="${ACT_ANNEAL:-100}"
ACT_LR="${ACT_LR:-2e-8}"
ACT_WD="${ACT_WD:-0.0}"

# Phase 2 (chain) — frozen seed for the chain + same LRs as the previous chain
CHAIN_OUT="${CHAIN_OUT:-output/mnv2_chain3}"
CHAIN_SEED="${CHAIN_SEED:-output/ptq/mnv2_act_intro_best.pt}"
CHAIN_LRS="${CHAIN_LRS:-2e-7 2e-6 1e-5 2e-6 2e-7 2e-8 2e-7 2e-6 2e-7 2e-8}"

SKIP_PHASE1="${SKIP_PHASE1:-0}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
# Reduce allocator fragmentation over the long run (PyTorch-recommended). The
# diagnostics OOM itself is fixed at the source (utils/quantizer_diagnostics.py
# now bounds its reductions); this is just headroom insurance at batch 1024.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() { echo "[act-then-chain $(date '+%F %T')] $*"; }

# ---- preflight ------------------------------------------------------------
if [[ ! -d "$DATA_DIR" ]]; then
    log "FATAL: data dir not found: $DATA_DIR"; exit 1
fi
if [[ "$SKIP_PHASE1" != "1" && ! -f "$SEED_CKPT" ]]; then
    log "FATAL: Phase-1 seed checkpoint not found: $SEED_CKPT"; exit 1
fi

ACT_BEST="$ACT_INTRO_DIR/checkpoints/best.pt"

# ==========================================================================
# PHASE 1 — activation introduction
# ==========================================================================
if [[ "$SKIP_PHASE1" == "1" ]]; then
    log "SKIP_PHASE1=1 — skipping activation-introduction run."
else
    log "=================================================================="
    log "PHASE 1: activation introduction"
    log "  seed      : $SEED_CKPT"
    log "  epochs    : $ACT_EPOCHS   gap=$ACT_GAP   anneal=$ACT_ANNEAL"
    log "  lr        : $ACT_LR   weight_decay=$ACT_WD"
    log "  output    : $ACT_INTRO_DIR"
    log "=================================================================="

    P1=( "$PYTHON" -u -m examples.train_imagenet_qat
         --data-dir "$DATA_DIR"
         --model "$MODEL"
         --init-from-ptq "$SEED_CKPT"
         --qat-gap "$ACT_GAP"
         --annealing-steps "$ACT_ANNEAL"
         --no-seed-best
         --require-full-quant-for-best
         --lr "$ACT_LR"
         --weight-decay "$ACT_WD"
         --epochs "$ACT_EPOCHS"
         --float-warmup-epochs 0
         --no-mixed-precision
         --output-dir "$ACT_INTRO_DIR" )

    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN: ${P1[*]}"
    else
        mkdir -p "$ACT_INTRO_DIR"
        "${P1[@]}" 2>&1 | tee "$ACT_INTRO_DIR/run.log"
        STATUS=${PIPESTATUS[0]}
        if (( STATUS != 0 )); then
            log "FATAL: Phase 1 exited with status $STATUS. Not starting the chain."
            exit "$STATUS"
        fi
        if [[ ! -f "$ACT_BEST" ]]; then
            log "FATAL: Phase 1 finished but wrote no $ACT_BEST."
            log "       (No epoch reached full quantization — need >~epoch 88 at gap=$ACT_GAP.)"
            exit 1
        fi
        log "PHASE 1 done -> $ACT_BEST"
    fi
fi

# Freeze Phase-1 best as the chain seed (stable path, immune to later edits).
if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -f "$ACT_BEST" ]]; then
        log "FATAL: chain seed source $ACT_BEST missing — cannot start Phase 2."
        exit 1
    fi
    cp "$ACT_BEST" "$CHAIN_SEED"
    log "Froze chain seed: $CHAIN_SEED"
    "$PYTHON" - "$CHAIN_SEED" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = c.get("metrics") or {}
print(f"[act-then-chain] chain seed val_acc={m.get('val_acc')}  quant_pct={m.get('quant_pct')}")
PY
fi

# ==========================================================================
# PHASE 2 — QAT chain with activation quantization
# ==========================================================================
log "=================================================================="
log "PHASE 2: QAT chain (act quant) from $CHAIN_SEED"
log "  LRs    : $CHAIN_LRS"
log "  output : $CHAIN_OUT"
log "=================================================================="

ACT_QUANT=1 \
LRS="$CHAIN_LRS" \
OUT_ROOT="$CHAIN_OUT" \
PTQ_CKPT="$CHAIN_SEED" \
DATA_DIR="$DATA_DIR" \
DRY_RUN="$DRY_RUN" \
    bash "$(dirname "$0")/run_mnv2_qat_chain.sh"
STATUS=$?

if (( STATUS != 0 )); then
    log "Phase 2 chain exited with status $STATUS."
    exit "$STATUS"
fi

log "=================================================================="
log "ALL DONE. Activation-quantized model chain complete under $CHAIN_OUT."
log "=================================================================="
