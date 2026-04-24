"""
yolov8n_pan_only.py
-------------------
YOLOv8n variant with the FPN top-down path removed.

Standard YOLOv8n neck:
    Backbone → P3, P4, P5
    Top-down:   P5 → upsample → cat(P4) → C2f → P4_up
                P4_up → upsample → cat(P3) → C2f → P3_out
    Bottom-up:  P3_out → conv → cat(P4_up) → C2f → P4_out
                P4_out → conv → cat(P5)    → C2f → P5_out
    Detect:     [P3_out, P4_out, P5_out]

This variant (PAN-only):
    Backbone → P3, P4, P5        (unchanged)
    Top-down:   removed entirely  (no upsamplers, no n12, no n15)
    Bottom-up:  P3 → conv → cat(P4) → C2f → P4_out
                P4_out → conv → cat(P5) → C2f → P5_out
    Detect:     [P3, P4_out, P5_out]

Effects:
  - Removes 2 Upsample layers, 2 C2f blocks (n12, n15)
  - Saves ~half the neck parameters
  - P3 detection scale sees no multi-scale context from deeper features
  - Better for latency-critical or edge deployments
  - No pretrained weight compatibility (neck is structurally different)
"""

import math
import torch
import torch.nn as nn

# Re-use all building blocks from the original model
from models.yolov8n import (
    Conv, C2f, SPPF, DetectHead,
    make_divisible, scale_channels, scale_depth,
)


class YOLOv8nPANOnly(nn.Module):
    """
    YOLOv8n with FPN top-down path removed — pure bottom-up PAN neck.

    Identical backbone to YOLOv8n. Neck is leaner:
      n16: Conv(C3, C3, s=2)               — downsample P3
      n18: C2f(C3+C4 → C4)                — fuse with P4
      n19: Conv(C4, C4, s=2)               — downsample
      n21: C2f(C4+C5 → C5)                — fuse with P5

    Head detects from [P3, P4_out, P5_out] at strides [8, 16, 32].
    """

    def __init__(self, nc: int = 80):
        super().__init__()
        C1, C2, C3, C4, C5 = 16, 32, 64, 128, 256

        # ---- Backbone (identical to YOLOv8n) ----
        self.b0  = Conv(3,  C1, k=3, s=2)
        self.b1  = Conv(C1, C2, k=3, s=2)
        self.b2  = C2f(C2, C2, n=scale_depth(3), shortcut=True)
        self.b3  = Conv(C2, C3, k=3, s=2)
        self.b4  = C2f(C3, C3, n=scale_depth(6), shortcut=True)   # → P3
        self.b5  = Conv(C3, C4, k=3, s=2)
        self.b6  = C2f(C4, C4, n=scale_depth(6), shortcut=True)   # → P4
        self.b7  = Conv(C4, C5, k=3, s=2)
        self.b8  = C2f(C5, C5, n=scale_depth(3), shortcut=True)
        self.b9  = SPPF(C5, C5, k=5)                              # → P5

        # ---- Neck: bottom-up only ----
        # P3 → downsample → concat P4 → C2f
        self.n16 = Conv(C3, C3, k=3, s=2)           # C3 → C3
        self.n18 = C2f(C3 + C4, C4, n=scale_depth(3), shortcut=False)  # P4_out

        # P4_out → downsample → concat P5 → C2f
        self.n19 = Conv(C4, C4, k=3, s=2)           # C4 → C4
        self.n21 = C2f(C4 + C5, C5, n=scale_depth(3), shortcut=False)  # P5_out

        # ---- Detection head ----
        # Detects from [P3 (raw), P4_out, P5_out]
        self.detect = DetectHead(nc=nc, ch=(C3, C4, C5))

        self.register_buffer('stride', torch.tensor([8., 16., 32.]))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            return self.loss(x)
        return self._forward_features(x)

    def _forward_features(self, x: torch.Tensor):
        # --- Backbone ---
        x  = self.b0(x)
        x  = self.b1(x)
        # x  = self.b2(x)
        x  = self.b3(x)
        p3 = self.b4(x)            # P3/8
        x  = self.b5(p3)
        p4 = self.b6(x)            # P4/16
        x  = self.b7(p4)
        x  = self.b8(x)
        p5 = self.b9(x)            # P5/32

        # --- Neck: bottom-up only ---
        x      = self.n16(p3)                       # downsample P3
        x      = torch.cat([x, p4], dim=1)          # cat with P4
        p4_out = self.n18(x)                        # P4_out

        x      = self.n19(p4_out)                   # downsample P4_out
        x      = torch.cat([x, p5], dim=1)          # cat with P5
        p5_out = self.n21(x)                        # P5_out

        # --- Head: P3 passed directly (no top-down enrichment) ---
        return self.detect([p3, p4_out, p5_out])

    # ------------------------------------------------------------------
    # Loss / criterion
    # ------------------------------------------------------------------

    def loss(self, batch, preds=None):
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        if preds is None or isinstance(preds, torch.Tensor):
            was_training = self.training
            self.train()
            preds = self._forward_features(batch['img'])
            if not was_training:
                self.eval()
        return self.criterion(preds, batch)

    def init_criterion(self):
        from ultralytics.utils.loss import v8DetectionLoss
        return v8DetectionLoss(self)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    orig  = __import__('yolov8n_model').YOLOv8n(nc=80)
    model = YOLOv8nPANOnly(nc=80)

    dummy = torch.zeros(1, 3, 640, 640)
    model.detect.stride = model.stride

    model.train()
    out_train = model._forward_features(dummy)
    print("Training output (dict):")
    print(f"  boxes:  {tuple(out_train['boxes'].shape)}")
    print(f"  scores: {tuple(out_train['scores'].shape)}")
    print(f"  feats:  {[tuple(f.shape) for f in out_train['feats']]}")

    model.eval()
    with torch.no_grad():
        out_eval = model._forward_features(dummy)
    print(f"\nInference output: {tuple(out_eval.shape)}  (expected (1, 84, 8400))")

    n_orig  = sum(p.numel() for p in orig.parameters())
    n_model = sum(p.numel() for p in model.parameters())
    removed = n_orig - n_model
    print(f"\nParameters:  original={n_orig:,}  pan_only={n_model:,}  removed={removed:,} ({removed/n_orig*100:.1f}%)")