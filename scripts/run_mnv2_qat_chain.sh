#!/usr/bin/env bash
#
# run_mnv2_qat_chain.sh — 10 chained QAT runs x 40 epochs = 400 epochs total.
#
# Each run is a SEPARATE PROCESS. That is the point: a fresh interpreter per run
# means a leak, a wedged DALI iterator or a CUDA fragmentation problem in run N
# cannot bleed into run N+1, and any single run can be re-launched on its own.
#
# Chaining: run N+1 starts from run N's best.pt (best VALIDATION ACCURACY, not
# last epoch). The trainer evaluates the incoming checkpoint first and seeds
# best.pt with it (trainer_v2.py:338 -> CheckpointManager.seed_best), setting the
# best-threshold to that score. So a run that only ever gets worse leaves the
# seeded checkpoint in place and the chain cannot regress: run N+1 then picks up
# run N's starting model rather than a worse ending one.
#
# LR schedule: each run has its own base LR (see LRS), held flat for 30 epochs
# then multiplied by 0.8 for the last 10. --step-lr-milestones takes FRACTIONS
# of the run, so 30/40 = 0.75.
#
# Weights + biases only (--no-act-quant); the PTQ checkpoint has no activation
# quantizers.
#
# Usage:
#   scripts/run_mnv2_qat_chain.sh                    # run the whole chain
#   START_AT=4 scripts/run_mnv2_qat_chain.sh         # resume from run 4
#   DRY_RUN=1 scripts/run_mnv2_qat_chain.sh          # print the commands only

set -uo pipefail

# ---- config ---------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/home/th/tmp/datasets/imagenet}"
PTQ_CKPT="${PTQ_CKPT:-output/ptq/mobilenetv2_W8_Anone_B8_singlepass.pt}"
OUT_ROOT="${OUT_ROOT:-output/mnv2_chain}"
MODEL="${MODEL:-mobilenetv2}"
BATCH="${BATCH:-1024}"          # 512 for ResNets (bigger activation tensors)
EPOCHS="${EPOCHS:-40}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-8}"
LR_DROP_FRAC="${LR_DROP_FRAC:-0.75}"     # 30/40
LR_DROP_GAMMA="${LR_DROP_GAMMA:-0.8}"
PYTHON="${PYTHON:-/home/th/miniconda3/envs/brevitas-qat/bin/python}"
START_AT="${START_AT:-1}"
DRY_RUN="${DRY_RUN:-0}"
# ACT_QUANT=1 runs WITH activation quantization (drops --no-act-quant). The seed
# checkpoint must already have every quantizer — weights, biases AND activations
# — calibrated; --init-from-ptq preserves them, so the model is fully quantized
# from step 0 (no re-annealing). Default 0 = weights+bias only.
ACT_QUANT="${ACT_QUANT:-0}"
# Per-model weight-LSB k override (empty = use the built-in default of 12).
WSIGMA="${WSIGMA:-}"

# Per-run base learning rates. Override with the LRS env var (space-separated),
# e.g.  LRS="2e-7 2e-6 1e-5 2e-6 ..."  — the run count follows the list length.
if [[ -n "${LRS:-}" ]]; then
    read -ra LRS <<< "$LRS"
else
    LRS=(1e-7 1e-6 1e-5 1e-6 1e-7 1e-8 1e-7 1e-6 1e-7 1e-8)
fi

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

# ---- preflight ------------------------------------------------------------
# Fail loudly here rather than 40 epochs in.
if [[ ! -f "$PTQ_CKPT" ]]; then
    echo "FATAL: PTQ checkpoint not found: $PTQ_CKPT" >&2
    echo "       Create it with: python -m examples.create_ptq_checkpoint --data-dir $DATA_DIR" >&2
    exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "FATAL: data dir not found: $DATA_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
CHAIN_LOG="$OUT_ROOT/chain.log"

log() { echo "[chain $(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

# Resolve the checkpoint run $1 must start from: the PTQ checkpoint for run 1,
# otherwise the previous run's best.pt.
resolve_input() {
    local run=$1
    if (( run == 1 )); then
        echo "$PTQ_CKPT"
    else
        local prev
        prev=$(printf '%s/run%02d_lr%s/checkpoints/best.pt' \
               "$OUT_ROOT" "$((run - 1))" "${LRS[$((run - 2))]}")
        echo "$prev"
    fi
}

log "=========================================================================="
log "MobileNetV2 QAT chain: ${#LRS[@]} runs x ${EPOCHS} epochs = $(( ${#LRS[@]} * EPOCHS )) epochs"
log "  LRs        : ${LRS[*]}"
log "  LR decay   : x${LR_DROP_GAMMA} at ${LR_DROP_FRAC} of each run (epoch 30/40)"
log "  seed PTQ   : $PTQ_CKPT"
log "  output     : $OUT_ROOT"
log "  start at   : run $START_AT"
log "=========================================================================="

for (( i = START_AT; i <= ${#LRS[@]}; i++ )); do
    LR="${LRS[$((i - 1))]}"
    RUN_DIR=$(printf '%s/run%02d_lr%s' "$OUT_ROOT" "$i" "$LR")
    IN_CKPT=$(resolve_input "$i")

    if [[ ! -f "$IN_CKPT" && "$DRY_RUN" != "1" ]]; then
        log "FATAL: run $i needs $IN_CKPT but it does not exist."
        log "       Run $((i - 1)) did not produce a best.pt — stopping the chain"
        log "       rather than silently restarting from the PTQ checkpoint."
        exit 1
    fi

    log "--------------------------------------------------------------------"
    log "RUN $i/${#LRS[@]}   lr=$LR   epochs=$EPOCHS"
    log "  from : $IN_CKPT"
    log "  to   : $RUN_DIR"

    CMD=( "$PYTHON" -u -m examples.train_imagenet_qat
          --data-dir "$DATA_DIR"
          --model "$MODEL"
          --init-from-ptq "$IN_CKPT"
          --batch-size "$BATCH"
          --lr "$LR"
          --weight-decay "$WEIGHT_DECAY"
          --epochs "$EPOCHS"
          --float-warmup-epochs 0
          --step-lr
          --step-lr-milestones "$LR_DROP_FRAC"
          --step-lr-gamma "$LR_DROP_GAMMA"
          --no-mixed-precision
          --output-dir "$RUN_DIR" )

    if [[ "$ACT_QUANT" == "1" ]]; then
        # Fully quantized already (weights+bias+act preserved from the seed);
        # gap=0 / annealing=1 keep it that way — no gradual re-introduction.
        CMD+=( --qat-gap 0 --annealing-steps 1 )
    else
        CMD+=( --no-act-quant )
    fi
    [[ -n "$WSIGMA" ]] && CMD+=( --weight-sigma-k "$WSIGMA" )

    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN: ${CMD[*]}"
        continue
    fi

    mkdir -p "$RUN_DIR"
    "${CMD[@]}" 2>&1 | tee "$RUN_DIR/run.log"
    STATUS=${PIPESTATUS[0]}

    if (( STATUS != 0 )); then
        log "FATAL: run $i exited with status $STATUS. Stopping the chain."
        log "       Fix it, then resume with: START_AT=$i $0"
        exit "$STATUS"
    fi

    OUT_BEST="$RUN_DIR/checkpoints/best.pt"
    if [[ ! -f "$OUT_BEST" ]]; then
        log "FATAL: run $i finished but wrote no $OUT_BEST. Stopping."
        exit 1
    fi

    # Report whether this run actually improved on what it was handed. Because
    # best.pt is seeded from the incoming checkpoint, "no improvement" means
    # best.pt IS the input — the chain holds rather than regresses.
    "$PYTHON" - "$IN_CKPT" "$OUT_BEST" "$i" <<'PY' 2>/dev/null | tee -a "$CHAIN_LOG"
import sys, torch
inp, out, run = sys.argv[1], sys.argv[2], sys.argv[3]
def acc(p):
    try:
        m = torch.load(p, map_location="cpu", weights_only=False).get("metrics") or {}
        return m.get("val_acc")
    except Exception:
        return None
a, b = acc(inp), acc(out)
if a is None or b is None:
    print(f"[chain] run {run}: (could not read metrics for comparison)")
else:
    d = b - a
    verdict = "IMPROVED" if d > 0 else "no improvement (chain held)"
    print(f"[chain] run {run}: in={a:.4f} -> best={b:.4f}  ({d:+.4f})  {verdict}")
PY

    log "RUN $i done -> $OUT_BEST"
done

log "=========================================================================="
log "Chain complete. Final model:"
FINAL=$(printf '%s/run%02d_lr%s/checkpoints/best.pt' \
        "$OUT_ROOT" "${#LRS[@]}" "${LRS[$(( ${#LRS[@]} - 1 ))]}")
log "  $FINAL"
log "Per-run summary:"
grep -a "^\[chain\] run" "$CHAIN_LOG" | tail -n "${#LRS[@]}" | tee -a "$CHAIN_LOG"
log "=========================================================================="
