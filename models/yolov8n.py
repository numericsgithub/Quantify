"""
yolov8n_model.py
----------------
Clean PyTorch reimplementation of YOLOv8n as a plain nn.Module.

Architecture mirrors the official yolov8.yaml with n-scale factors applied:
    depth_multiple  = 0.33   (repeats = max(round(n * 0.33), 1))
    width_multiple  = 0.25   (channels = make_divisible(c * 0.25, 8))
    max_channels    = 1024

Backbone (10 modules, indices 0-9):
    0   Conv(3,   16,  k=3, s=2)        P1/2
    1   Conv(16,  32,  k=3, s=2)        P2/4
    2   C2f(32,  32,  n=1, shortcut=True)
    3   Conv(32,  64,  k=3, s=2)        P3/8   → save as P3
    4   C2f(64,  64,  n=2, shortcut=True)
    5   Conv(64,  128, k=3, s=2)        P4/16  → save as P4
    6   C2f(128, 128, n=2, shortcut=True)
    7   Conv(128, 256, k=3, s=2)        P5/32
    8   C2f(256, 256, n=1, shortcut=True)
    9   SPPF(256, 256, k=5)             → save as P5

Neck + Head (modules 10-21):
    10  Upsample(scale=2)
    11  Concat([10, P4])                → 128+256 = 384 ... wait, see note below
    12  C2f(384→128, n=1, shortcut=False)   → P4_up
    13  Upsample(scale=2)
    14  Concat([13, P3])                → 128+64 = 192
    15  C2f(192→64,  n=1, shortcut=False)   → P3_out  (small)
    16  Conv(64,  64,  k=3, s=2)
    17  Concat([16, P4_up])             → 64+128 = 192
    18  C2f(192→128, n=1, shortcut=False)   → P4_out  (medium)
    19  Conv(128, 128, k=3, s=2)
    20  Concat([19, P5])                → 128+256 = 384
    21  C2f(384→256, n=1, shortcut=False)   → P5_out  (large)

Detection head (module 22):
    Detect([P3_out, P4_out, P5_out], nc=80)

Width-scaled channel sizes (width=0.25, divisor=8):
    base 64  → 16
    base 128 → 32
    base 256 → 64
    base 512 → 128
    base 1024→ 256

Depth-scaled repeats (depth=0.33):
    n=3 → max(round(3*0.33),1) = 1
    n=6 → max(round(6*0.33),1) = 2

Note: concat channel arithmetic uses the *scaled* channel counts above.
"""

import math
import torch
import torch.nn as nn
import brevitas.nn as qnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_divisible(x: float, divisor: int = 8) -> int:
    """Round x up to the nearest multiple of divisor."""
    return max(divisor, int(math.ceil(x / divisor) * divisor))


def scale_channels(c: int, width: float = 0.25, divisor: int = 8) -> int:
    return make_divisible(c * width, divisor)


def scale_depth(n: int, depth: float = 0.33) -> int:
    return int(max(round(n * depth), 1))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Standard Conv + BN + SiLU. Supports Brevitas quantization."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None,
                 weight_quant=None, act_quant=None):
        super().__init__()
        if p is None:
            p = k // 2  # 'same' padding for odd kernels
        self.conv = qnn.QuantConv2d(in_ch, out_ch, k, s, p, bias=False, weight_quant=weight_quant) #,  , output_quant=act_quant
        # self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False) # The unquantized version
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)
        # TODO: Add SiLU quantizer later

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """
    YOLOv8 Bottleneck.
    Unlike YOLOv5, both convolutions use 3x3 kernels.
    shortcut is only applied when in_ch == out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, e: float = 0.5,
                 weight_quant=None, act_quant=None):
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1 = Conv(in_ch,  hidden, k=3, s=1, weight_quant=weight_quant, act_quant=act_quant)
        self.cv2 = Conv(hidden, out_ch, k=3, s=1, weight_quant=weight_quant, act_quant=act_quant)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    Cross-Stage Partial with 2 convolutions and n Bottlenecks.
    Gradient flow is improved by splitting and re-concatenating feature maps.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = False, e: float = 0.5,
                 weight_quant=None, act_quant=None):
        super().__init__()
        self.hidden = int(out_ch * e)  # hidden channels per branch
        self.cv1 = Conv(in_ch, 2 * self.hidden, k=1, weight_quant=weight_quant, act_quant=act_quant)
        self.cv2 = Conv((2 + n) * self.hidden, out_ch, k=1, weight_quant=weight_quant, act_quant=act_quant)
        # Named 'm' to match official Ultralytics C2f checkpoint keys exactly
        self.m = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, shortcut=shortcut, e=1.0, weight_quant=weight_quant, act_quant=act_quant)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the output of cv1 into two halves along the channel dim
        y = list(self.cv1(x).chunk(2, dim=1))
        # Sequentially apply bottlenecks, appending each output
        for bottleneck in self.m:
            y.append(bottleneck(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling – Fast.
    Equivalent to SPP(k=(5,9,13)) but uses sequential 5x5 max-pools.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5, weight_quant=None, act_quant=None):
        super().__init__()
        hidden = in_ch // 2
        self.cv1  = Conv(in_ch,  hidden, k=1, weight_quant=weight_quant, act_quant=act_quant)
        self.cv2  = Conv(hidden * 4, out_ch, k=1, weight_quant=weight_quant, act_quant=act_quant)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class DFL(nn.Module):
    """
    Distribution Focal Loss regression head.
    Converts a distribution over `reg_max` bins to a single offset value.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        # Fixed weights: [0, 1, 2, ..., reg_max-1]
        self.conv.weight.data[:] = torch.arange(reg_max, dtype=torch.float).reshape(1, reg_max, 1, 1)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, 4*reg_max, anchors
        return self.conv(
            x.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)


class DetectHead(nn.Module):
    """
    Anchor-free decoupled detection head (YOLOv8 style).

    For each of the three FPN levels:
      - A 2-conv regression branch → 4 * reg_max outputs (DFL box encoding)
      - A 2-conv classification branch → nc outputs

    Outputs during inference:
        (batch, 4 + nc, total_anchors)   where total_anchors = H1*W1 + H2*W2 + H3*W3

    During training the raw branch outputs are returned as a list so that the
    Ultralytics loss function can consume them directly (it expects that format).
    """

    reg_max = 16  # DFL bins — must match official checkpoint

    def __init__(self, nc: int = 80, ch: tuple = (64, 128, 256), stride: torch.Tensor = None,
                 weight_quant=None, act_quant=None):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)      # number of detection layers
        self.no = nc + self.reg_max * 4  # outputs per anchor

        if stride is None:
            stride = torch.tensor([8., 16., 32.])
        self.register_buffer('stride', stride)

        # Two-conv branches per scale.
        # Official yolov8n uses fixed hidden dims:
        #   cv2 (box branch):  hidden = max(c, 4 * reg_max)  → max(c, 64)
        #   cv3 (cls branch):  hidden = max(c, nc)           → max(c, 80)
        # For the nano scale (c = 64, 128, 256) this gives:
        #   cv2: 64, 64, 64   (since max(64,64)=64, max(128,64)=128 → official uses 64 always)
        #   cv3: 80, 80, 80   (since max(64,80)=80)
        # The official checkpoint actually hard-codes 64 for cv2 and 80 for cv3
        # across all scales for yolov8n, so we mirror that exactly.
        c2_hidden = max(ch[0], 4 * self.reg_max)   # 64
        c3_hidden = max(ch[0], nc)                  # 80

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c, c2_hidden, k=3, weight_quant=weight_quant, act_quant=act_quant),
                Conv(c2_hidden, c2_hidden, k=3, weight_quant=weight_quant, act_quant=act_quant),
                nn.Conv2d(c2_hidden, 4 * self.reg_max, 1)
            )
            for c in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(c, c3_hidden, k=3, weight_quant=weight_quant, act_quant=act_quant),
                Conv(c3_hidden, c3_hidden, k=3, weight_quant=weight_quant, act_quant=act_quant),
                nn.Conv2d(c3_hidden, nc, 1)
            )
            for c in ch
        )
        self.dfl = DFL(self.reg_max)

    def forward(self, features: list):
        """
        features: list of tensors [P3_out, P4_out, P5_out]

        Training mode  → returns dict {"boxes", "scores", "feats"} for v8DetectionLoss.
        Inference mode → returns decoded (B, 4+nc, total_anchors) tensor for NMS/AutoBackend.
        """
        bs = features[0].shape[0]
        boxes  = torch.cat(
            [self.cv2[i](feat).view(bs, 4 * self.reg_max, -1) for i, feat in enumerate(features)],
            dim=-1
        )
        scores = torch.cat(
            [self.cv3[i](feat).view(bs, self.nc, -1) for i, feat in enumerate(features)],
            dim=-1
        )
        if self.training:
            return {"boxes": boxes, "scores": scores, "feats": features}
        return self._inference(boxes, scores, features)

    def _inference(self, boxes: torch.Tensor, scores: torch.Tensor, features: list) -> torch.Tensor:
        """
        Decode raw DFL box distribution + class scores into (B, 4+nc, total_anchors).
        Mirrors Detect._get_decode_boxes + _inference exactly.
        """
        from ultralytics.utils.tal import make_anchors, dist2bbox

        shape = features[0].shape
        if not hasattr(self, '_anchors') or self._inf_shape != shape:
            # make_anchors returns (A,2), (A,1) — transpose to (2,A), (1,A)
            self._anchors, self._strides = (
                a.transpose(0, 1) for a in make_anchors(features, self.stride, 0.5)
            )
            self._inf_shape = shape

        # DFL decode: (B, 4*reg_max, A) → (B, 4, A) ltrb distances
        dfl_out = self.dfl(boxes)                                      # (B, 4, A)
        # dist2bbox: ltrb + anchors → xywh scaled by strides
        dbox = dist2bbox(dfl_out, self._anchors.unsqueeze(0), xywh=True, dim=1) * self._strides
        return torch.cat([dbox, scores.sigmoid()], 1)                  # (B, 4+nc, A)


# ---------------------------------------------------------------------------
# Full YOLOv8n Model
# ---------------------------------------------------------------------------

class YOLOv8n(nn.Module):
    """
    Clean YOLOv8n reimplementation as a plain nn.Module.

    Channel sizes after width scaling (width=0.25):
        P3: 64   P4: 128   P5: 256

    This class intentionally has no dependency on Ultralytics internals.
    It is designed to be:
      1. Weight-compatible with the official yolov8n.pt checkpoint (after
         key remapping in the adapter layer).
      2. QAT-ready: all conv+bn+act sequences are in Conv submodules for
         easy layer fusion.
    """

    def __init__(self, nc: int = 80, weight_quant=None, act_quant=None, weight_bit_width: int = 8, act_bit_width: int = 8):
        super().__init__()
        # ---- Scaled channel counts ----
        # base → scaled (width=0.25)
        #   64 → 16,  128 → 32,  256 → 64,  512 → 128,  1024 → 256
        C1, C2, C3, C4, C5 = 16, 32, 64, 128, 256

        # ---- Backbone ----
        self.b0  = Conv(3,  C1, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                         # 0  P1/2
        self.b1  = Conv(C1, C2, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                         # 1  P2/4
        self.b2  = C2f(C2, C2, n=scale_depth(3), shortcut=True, weight_quant=weight_quant, act_quant=act_quant)   # 2
        self.b3  = Conv(C2, C3, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                         # 3  P3/8
        self.b4  = C2f(C3, C3, n=scale_depth(6), shortcut=True, weight_quant=weight_quant, act_quant=act_quant)   # 4
        self.b5  = Conv(C3, C4, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                         # 5  P4/16
        self.b6  = C2f(C4, C4, n=scale_depth(6), shortcut=True, weight_quant=weight_quant, act_quant=act_quant)   # 6
        self.b7  = Conv(C4, C5, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                         # 7  P5/32
        self.b8  = C2f(C5, C5, n=scale_depth(3), shortcut=True, weight_quant=weight_quant, act_quant=act_quant)   # 8
        self.b9  = SPPF(C5, C5, k=5, weight_quant=weight_quant, act_quant=act_quant)                              # 9

        # ---- Neck (PAN-FPN) ----
        # Top-down path
        self.up1  = nn.Upsample(scale_factor=2, mode='nearest')    # 10
        # concat with b6 (P4): C5 + C4 = 256+128 = 384
        self.n12  = C2f(C5 + C4, C4, n=scale_depth(3), shortcut=False, weight_quant=weight_quant, act_quant=act_quant)  # 12

        self.up2  = nn.Upsample(scale_factor=2, mode='nearest')    # 13
        # concat with b4 (P3): C4 + C3 = 128+64 = 192
        self.n15  = C2f(C4 + C3, C3, n=scale_depth(3), shortcut=False, weight_quant=weight_quant, act_quant=act_quant)  # 15  P3_out

        # Bottom-up path
        self.n16  = Conv(C3, C3, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                        # 16
        # concat with n12 (P4_up): C3 + C4 = 64+128 = 192
        self.n18  = C2f(C3 + C4, C4, n=scale_depth(3), shortcut=False, weight_quant=weight_quant, act_quant=act_quant)  # 18  P4_out

        self.n19  = Conv(C4, C4, k=3, s=2, weight_quant=weight_quant, act_quant=act_quant)                        # 19
        # concat with b9 (P5): C4 + C5 = 128+256 = 384
        self.n21  = C2f(C4 + C5, C5, n=scale_depth(3), shortcut=False, weight_quant=weight_quant, act_quant=act_quant)  # 21  P5_out

        # ---- Detection head ----
        self.detect = DetectHead(nc=nc, ch=(C3, C4, C5), stride=self.stride, weight_quant=weight_quant, act_quant=act_quant)

        # Strides registered as a buffer so they survive .to(device) and are
        # always accessible via model.stride — required by the Ultralytics
        # trainer (build_dataset, _setup_train) and validator.
        self.register_buffer('stride', torch.tensor([8., 16., 32.]))

    def forward(self, x, **kwargs):
        """
        Dispatch based on input type:
          - dict  → training: compute and return loss
          - tensor → inference or feature extraction (kwargs like augment/embed
                     passed by the Ultralytics validator are accepted but ignored;
                     augmented inference is not implemented for our custom model)
        """
        if isinstance(x, dict):
            return self.loss(x)
        return self._forward_features(x)

    def _forward_features(self, x: torch.Tensor):
        """Run backbone + neck + head and return raw [box, cls] per scale."""
        # --- Backbone ---
        x  = self.b0(x)
        x  = self.b1(x)
        x  = self.b2(x)
        x  = self.b3(x)
        p3 = self.b4(x)
        x  = self.b5(p3)
        p4 = self.b6(x)
        x  = self.b7(p4)
        x  = self.b8(x)
        p5 = self.b9(x)

        # --- Neck: top-down ---
        x      = self.up1(p5)
        x      = torch.cat([x, p4], dim=1)
        p4_up  = self.n12(x)

        x      = self.up2(p4_up)
        x      = torch.cat([x, p3], dim=1)
        p3_out = self.n15(x)

        # --- Neck: bottom-up ---
        x      = self.n16(p3_out)
        x      = torch.cat([x, p4_up], dim=1)
        p4_out = self.n18(x)

        x      = self.n19(p4_out)
        x      = torch.cat([x, p5], dim=1)
        p5_out = self.n21(x)

        # --- Head ---
        return self.detect([p3_out, p4_out, p5_out])

    def loss(self, batch, preds=None):
        """Compute v8DetectionLoss. Called by forward() when batch is a dict,
        and also by the validator as model.loss(batch, preds) where preds is
        the already-decoded inference tensor — in that case we ignore preds
        and recompute from the image in training mode."""
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        # If preds is a plain tensor (inference output from the validator) rather
        # than the training dict, discard it and recompute in training mode.
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
# Quick sanity check (shapes only, no weights)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = YOLOv8n(nc=80)
    dummy = torch.zeros(1, 3, 640, 640)

    model.train()
    out_train = model._forward_features(dummy)
    print("Training output (dict):")
    print(f"  boxes:  {tuple(out_train['boxes'].shape)}")
    print(f"  scores: {tuple(out_train['scores'].shape)}")
    print(f"  feats:  {[tuple(f.shape) for f in out_train['feats']]}")

    model.eval()
    model.detect.stride = torch.tensor([8., 16., 32.])
    with torch.no_grad():
        out_eval = model._forward_features(dummy)
    print(f"\nInference output (decoded tensor): {tuple(out_eval.shape)}")
    print(f"  Expected: (1, 84, 8400)")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
    print("Expected (official yolov8n): ~3,157,200")
