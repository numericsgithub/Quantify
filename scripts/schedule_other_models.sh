#!/usr/bin/env bash
#
# schedule_other_models.sh — run the full W8/A8/B8 pipeline for the three
# remaining models, one after another, on a single GPU.
#
# MobileNetV2 already went through the pipeline. This does MobileNetV1, ResNet18
# and ResNet50 in sequence (never concurrently — one GPU). Per-model batch size
# and Phase-2 activation-ramp gap are set below from the measured activation
# schedules (see docs/llm/skills/int8_qat_training_pipeline.md):
#
#     model         batch   gap     (gap tuned so the activation ramp ends
#     mobilenetv1   1024    1376     ~ep88 of the 130-epoch intro run, leaving
#     resnet18       512    3191     ~42 fully-quantized recovery epochs)
#     resnet50       512    1251
#
# The MobileNetV2 run is (probably) still using the GPU. By default this script
# WAITS until the GPU has been essentially idle for a sustained window before it
# starts, so it will not fight the running job or trip over the brief gaps
# between chain runs. Launch it now and forget it:
#
#     nohup bash scripts/schedule_other_models.sh > output/other_models.out 2>&1 &
#
# Env:
#   SKIP_GPU_WAIT=1   start immediately (e.g. after you stopped MobileNetV2)
#   ONLY="resnet18"   run just one model (space-separated list to subset)
#   FREE_MIB=4000     "idle" threshold in MiB (default 4000)
#   DRY_RUN=1         print the pipeline commands only

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"
FREE_MIB="${FREE_MIB:-4000}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-resnet18 resnet50 mobilenetv1}"

# model:batch:gap:weight_sigma_k
#   gap  — measured WITH bias quantizers threaded (they shift activation exec ids)
#   wsk  — per-model weight-LSB k; empty = default 12. ResNets quantize fine at
#          12; MobileNetV1's depthwise layers collapse at 12 and need ~20
#          (pitfall #15). ResNets run first (validated); MNv1 last.
PLAN=(
  "resnet18:512:3191:"
  "resnet50:512:1251:"
  "mobilenetv1:1024:1376:20"
)

log() { echo "[schedule $(date '+%F %T')] $*"; }

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

wait_for_free_gpu() {
    # Require the GPU to read idle for 3 consecutive checks 90s apart, so the
    # few-second gaps between a running chain's runs cannot be mistaken for "done".
    local need=3 ok=0
    log "Waiting for the GPU to be idle (<${FREE_MIB} MiB) for ${need}x90s before starting…"
    while true; do
        local used; used="$(gpu_used_mib)"
        if [[ -z "$used" ]]; then log "nvidia-smi unavailable; assuming free"; return 0; fi
        if (( used < FREE_MIB )); then
            ok=$((ok + 1))
            log "  GPU idle ($used MiB) — $ok/$need"
            (( ok >= need )) && { log "GPU free. Starting."; return 0; }
        else
            (( ok > 0 )) && log "  GPU busy again ($used MiB) — resetting"
            ok=0
        fi
        sleep 90
    done
}

if [[ "$SKIP_GPU_WAIT" != "1" && "$DRY_RUN" != "1" ]]; then
    wait_for_free_gpu
fi

log "=========================================================================="
log "Scheduling pipelines for: $ONLY"
log "=========================================================================="

for entry in "${PLAN[@]}"; do
    IFS=":" read -r model batch gap wsk <<< "$entry"
    [[ " $ONLY " == *" $model "* ]] || { log "skip $model (not in ONLY)"; continue; }

    log "--------------------------------------------------------------------"
    log ">>> START pipeline: $model  (batch=$batch gap=$gap weight_sigma_k=${wsk:-default})"
    log "--------------------------------------------------------------------"

    MODEL="$model" BATCH="$batch" GAP="$gap" WEIGHT_SIGMA_K="$wsk" DRY_RUN="$DRY_RUN" \
        bash "$HERE/run_int8_pipeline.sh"
    status=$?

    if (( status != 0 )); then
        log "!!! pipeline for $model FAILED (status $status). Stopping the schedule."
        log "    Fix it, then resume the rest with: ONLY=\"<remaining models>\" $0"
        exit "$status"
    fi
    log "<<< DONE pipeline: $model"
done

log "=========================================================================="
log "ALL SCHEDULED PIPELINES COMPLETE."
log "=========================================================================="
