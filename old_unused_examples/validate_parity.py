"""
validate_parity.py
------------------
Runs Ultralytics' official validator on both the official yolov8n and our
custom reimplementation, then compares mAP scores side by side.

Using the same validator for both models ensures any difference is purely
the model — NMS thresholds, metric computation, and data loading are identical.

Requirements
------------
    pip install ultralytics

    The COCO val2017 dataset must be available. Point --data at your
    existing coco_train.yaml (from train_yolo_coco.py) or a standard
    coco.yaml. The script will download COCO val images automatically
    via Ultralytics if the path doesn't exist yet.

Run
---
    # Using your existing COCO yaml from the training script:
    python validate_parity.py --data ./runs/yolo_coco_train/data/coco_yolo_fmt/coco_train.yaml

    # Or let Ultralytics use its own coco.yaml (downloads if needed):
    python validate_parity.py --data coco.yaml

    # Faster smoke-test on coco8 (8 images, no download needed):
    python validate_parity.py --data coco8.yaml --smoke-test

    # GPU:
    python validate_parity.py --data coco.yaml --device cuda
"""

import argparse
import sys
import torch
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER

from models.yolov8n_model import YOLOv8n
from examples.yolov8n_adapter import YOLOv8nDetectionModel, _remap_official_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_official_yolo(pt_path: str) -> YOLO:
    """Load the official yolov8n as a standard Ultralytics YOLO object."""
    return YOLO(pt_path)


def load_custom_yolo(pt_path: str) -> YOLO:
    """
    Build a YOLO object backed by our custom nn.Module with pretrained weights.
    The YOLO wrapper provides the val() method and handles all postprocessing.
    """
    # Start from the official checkpoint so YOLO sets up task/names/stride
    yolo = YOLO(pt_path)

    # Swap the inner DetectionModel for our custom one
    custom_det_model = YOLOv8nDetectionModel(nc=yolo.model.nc, verbose=False)

    # Load weights from the checkpoint, cast fp16→fp32
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    official_model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    official_sd = official_model.state_dict()
    remapped = {k: v.float() for k, v in _remap_official_weights(official_sd).items()}

    missing, unexpected = custom_det_model._yolo.load_state_dict(remapped, strict=True)
    assert not missing and not unexpected, \
        f"Weight loading failed: {len(missing)} missing, {len(unexpected)} unexpected"

    # Replace the model inside the YOLO wrapper
    # We need to make _yolo's forward pass reachable via the YOLO wrapper's
    # predict pipeline. The simplest way: replace yolo.model with a shim that
    # routes to our _yolo but keeps all Ultralytics DetectionModel attributes.
    _patch_detection_model(yolo.model, custom_det_model._yolo)

    return yolo


def _patch_detection_model(official_det_model, our_yolo_nn: YOLOv8n):
    """
    Patch official DetectionModel in-place to run our backbone+neck.

    Two things happen here:
    1. our_yolo_nn.detect cv2/cv3 weights are copied into the official Detect
       head, so it produces identical outputs to ours.
    2. _predict_once is overridden to run our backbone+neck, then feed the
       FPN tensors to the official Detect head (DFL decode + NMS prep intact).
    3. our_yolo_nn is registered as a proper submodule so .to(device) / .half()
       on the DetectionModel propagates automatically.
    """
    import types

    # Step 1: copy our head weights into the official Detect head
    _patch_detect_head(official_det_model.model[-1], our_yolo_nn.detect)

    # Step 2: register as submodule for automatic device/dtype propagation
    official_det_model.our_yolo_nn = our_yolo_nn

    # Step 3: override _predict_once
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        nn = self.our_yolo_nn

        x = nn.b0(x)
        x = nn.b1(x)
        x = nn.b2(x)
        x = nn.b3(x)
        p3 = nn.b4(x)
        x = nn.b5(p3)
        p4 = nn.b6(x)
        x = nn.b7(p4)
        x = nn.b8(x)
        p5 = nn.b9(x)

        x = nn.up1(p5)
        x = torch.cat([x, p4], dim=1)
        p4_up = nn.n12(x)
        x = nn.up2(p4_up)
        x = torch.cat([x, p3], dim=1)
        p3_out = nn.n15(x)

        x = nn.n16(p3_out)
        x = torch.cat([x, p4_up], dim=1)
        p4_out = nn.n18(x)
        x = nn.n19(p4_out)
        x = torch.cat([x, p5], dim=1)
        p5_out = nn.n21(x)

        return self.model[-1]([p3_out, p4_out, p5_out])

    official_det_model._predict_once = types.MethodType(_predict_once, official_det_model)


def _patch_detect_head(official_detect, our_detect_head):
    """
    Make the official Detect head re-use our DetectHead's cv2/cv3 weights,
    so the DFL decode + anchor logic in the official head runs on our outputs.

    Rather than re-implementing DFL decode ourselves, we copy our trained
    cv2/cv3 weights into the official head and let it do the rest.
    This keeps NMS, stride anchors, and box decoding 100% identical to official.
    """
    # Copy our cv2/cv3 weights into the official detect head
    with torch.no_grad():
        for i in range(len(our_detect_head.cv2)):
            # Our head and official head have the same cv2/cv3 structure
            our_sd = our_detect_head.cv2[i].state_dict()
            official_detect.cv2[i].load_state_dict(our_sd)

            our_sd = our_detect_head.cv3[i].state_dict()
            official_detect.cv3[i].load_state_dict(our_sd)

    # Now patch official detect's forward to use our backbone+neck outputs.
    # The official _predict_once feeds raw image features into detect.forward(),
    # which runs cv2/cv3 then DFL. Since we've copied our cv2/cv3 weights above,
    # the official head will now produce identical outputs to ours — we just
    # need to feed it the right backbone features.
    #
    # Strategy: override official DetectionModel._predict_once to run our
    # backbone+neck directly, then pass to the official detect head.
    pass  # weights already copied above — official head will use them


def run_validation(yolo: YOLO, data: str, device: str, batch: int,
                   imgsz: int, label: str) -> dict:
    """Run val() and return the metrics dict."""
    print(f"\n{'─' * 55}")
    print(f"  Validating: {label}")
    print(f"{'─' * 55}")

    metrics = yolo.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
    )
    return metrics


def print_comparison(official_metrics, custom_metrics):
    """Print a side-by-side mAP comparison table."""

    def get(m, key):
        # Ultralytics metrics object — try attribute access then dict
        try:
            return float(getattr(m, key))
        except AttributeError:
            try:
                return float(m.results_dict.get(key, float('nan')))
            except Exception:
                return float('nan')

    metrics_to_compare = [
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50-95"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ]

    print(f"\n{'=' * 60}")
    print(f"  {'Metric':<22} {'Official':>12} {'Custom':>12} {'Δ':>10}")
    print(f"{'─' * 60}")

    all_match = True
    for key, label in metrics_to_compare:
        off_val = get(official_metrics, key)
        our_val = get(custom_metrics, key)
        delta = our_val - off_val
        match = abs(delta) < 0.005  # within 0.5 mAP points
        flag = "✅" if match else "⚠️ "
        print(f"  {flag} {label:<20} {off_val:>12.4f} {our_val:>12.4f} {delta:>+10.4f}")
        if not match:
            all_match = False

    print(f"{'=' * 60}")
    return all_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Validate official vs custom YOLOv8n on COCO")
    p.add_argument("--pt", default="yolov8n.pt", help="Path to yolov8n.pt checkpoint")
    p.add_argument("--data", default="coco8.yaml", help="Dataset YAML (coco.yaml or your coco_train.yaml)")
    p.add_argument("--device", default="cpu", help="torch device")
    p.add_argument("--batch", default=16, type=int, help="Validation batch size")
    p.add_argument("--imgsz", default=640, type=int, help="Input image size")
    p.add_argument("--smoke-test", action="store_true", help="Use coco8.yaml for a quick 8-image test")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.data = "coco8.yaml"
        print("Smoke-test mode: using coco8.yaml (8 images)")

    print("=" * 60)
    print("  YOLOv8n Validation Parity")
    print("=" * 60)
    print(f"  Checkpoint : {args.pt}")
    print(f"  Data       : {args.data}")
    print(f"  Device     : {args.device}")
    print(f"  Batch      : {args.batch}")

    # ── Load both models ──────────────────────────────────────────────────
    print(f"\nLoading official model...")
    official_yolo = load_official_yolo(args.pt)

    print(f"Loading custom model...")
    custom_yolo = load_custom_yolo(args.pt)

    # ── Validate both ─────────────────────────────────────────────────────
    official_metrics = run_validation(
        official_yolo, args.data, args.device, args.batch, args.imgsz,
        label="Official yolov8n"
    )
    custom_metrics = run_validation(
        custom_yolo, args.data, args.device, args.batch, args.imgsz,
        label="Custom YOLOv8n (our reimplementation)"
    )

    # ── Compare ───────────────────────────────────────────────────────────
    all_match = print_comparison(official_metrics, custom_metrics)

    if all_match:
        print("\n  🎉 Validation parity confirmed — models produce identical mAP.")
        print("     Phase 1 complete. Ready to proceed to QAT scaffolding.")
    else:
        print("\n  ⚠️  mAP difference exceeds 0.5 points — investigate before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()