"""
yolov8n_adapter.py
------------------
Thin adapter that makes YOLOv8n (our clean nn.Module) usable with the
Ultralytics training infrastructure (model.train(), model.val(), etc.).

Strategy
--------
Ultralytics' YOLO class wraps a DetectionModel internally. DetectionModel
is itself an nn.Module with specific attributes and a forward() signature
that the Ultralytics loss (v8DetectionLoss) and validator expect.

Rather than rewriting all of that, we subclass DetectionModel and replace
its internal .model (the sequential layer list) with our clean module, while
keeping all the Ultralytics plumbing (loss, metrics, NMS, etc.) intact.

Key contract points DetectionModel must satisfy for Ultralytics training:
    - self.model        : the actual nn.Module (we replace this)
    - self.nc           : number of classes
    - self.names        : class names dict
    - self.stride       : anchor strides tensor (e.g. [8, 16, 32])
    - self.args         : training args (populated by Ultralytics)
    - forward(x)        : returns raw head outputs during training,
                          post-processed results during inference
    - loss(batch, preds): calls v8DetectionLoss

The cleanest seam is: override forward() to call our YOLOv8n, then
format the output so Ultralytics' loss function is happy.

Usage
-----
    from yolov8n_adapter import YOLOv8nDetectionModel
    from ultralytics import YOLO

    # Wrap our custom model in a YOLO object
    yolo = YOLO.__new__(YOLO)
    yolo.model = YOLOv8nDetectionModel(nc=80)
    yolo.model.load_pretrained_weights("yolov8n.pt")
    results = yolo.train(data="coco.yaml", ...)

Or more simply via the helper at the bottom of this file.
"""

import torch
import torch.nn as nn
from pathlib import Path

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

# from models.yolov8n_model import YOLOv8n
from models.yolov8PanOnly import YOLOv8nPANOnly



# ---------------------------------------------------------------------------
# Key remapping: official Ultralytics → our naming
# ---------------------------------------------------------------------------

def _remap_official_weights(official_sd: dict) -> dict:
    """
    Map official Ultralytics DetectionModel state_dict keys to our naming.

    Official key structure (from DetectionModel.state_dict()):
        model.0.conv.weight             Conv layer 0
        model.2.cv1.conv.weight         C2f layer 2, cv1
        model.2.m.0.cv1.conv.weight     C2f layer 2, bottleneck 0 (named 'm')
        model.22.cv2.0.0.conv.weight    Detect head, cv2 branch scale 0, layer 0

    Our naming:
        b0.conv.weight
        b2.cv1.conv.weight
        b2.m.0.cv1.conv.weight
        detect.cv2.0.0.conv.weight
    """
    layer_map = {
        # Backbone
        "model.0": "b0",
        "model.1": "b1",
        "model.2": "b2",
        "model.3": "b3",
        "model.4": "b4",
        "model.5": "b5",
        "model.6": "b6",
        "model.7": "b7",
        "model.8": "b8",
        "model.9": "b9",
        # Neck (layers 10,13=Upsample; 11,14,17,20=Concat — no weights)
        "model.12": "n12",
        "model.15": "n15",
        "model.16": "n16",
        "model.18": "n18",
        "model.19": "n19",
        "model.21": "n21",
        # Head
        "model.22": "detect",
    }

    remapped = {}
    skipped = []

    for k, v in official_sd.items():
        matched = False
        for official_prefix, our_prefix in layer_map.items():
            if k.startswith(official_prefix + ".") or k == official_prefix:
                new_key = our_prefix + k[len(official_prefix):]
                remapped[new_key] = v
                matched = True
                break
        if not matched:
            skipped.append(k)

    if skipped:
        print(f"[_remap_official_weights] Skipped {len(skipped)} keys "
              f"(Upsample/Concat have no weights — expected): {skipped[:6]}")

    return remapped


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_weights_into_model(model: YOLOv8n, pt_path: str) -> YOLOv8n:
    """
    Load an official yolov8n.pt checkpoint into our YOLOv8n.

    Handles fp16 checkpoints (common in Ultralytics releases) by casting
    all weights to fp32.

    Args:
        model:   A YOLOv8n instance (caller's responsibility to match nc).
        pt_path: Path to yolov8n.pt. Downloaded automatically if missing.

    Returns:
        The same model with weights loaded in-place.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        pt_path = Path(attempt_download_asset(str(pt_path)))

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    official_obj = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    official_sd = official_obj.state_dict() if hasattr(official_obj, "state_dict") else official_obj

    remapped = {k: v.float() for k, v in _remap_official_weights(official_sd).items()}
    missing, unexpected = model.load_state_dict(remapped, strict=True)

    n_loaded = len(remapped) - len(missing)
    print(f"[load_weights] {n_loaded}/{len(remapped)} tensors loaded from {pt_path.name}")
    if missing:
        print(f"  Missing ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)}): {unexpected[:5]}")

    return model


# ---------------------------------------------------------------------------
# Patching helpers for validate_parity.py
# ---------------------------------------------------------------------------

def _patch_detect_head(official_detect, our_detect_head):
    """Copy our DetectHead cv2/cv3 weights into the official Detect head."""
    with torch.no_grad():
        for i in range(len(our_detect_head.cv2)):
            official_detect.cv2[i].load_state_dict(our_detect_head.cv2[i].state_dict())
            official_detect.cv3[i].load_state_dict(our_detect_head.cv3[i].state_dict())


def _patch_detection_model(official_det_model, our_yolo_nn: YOLOv8n):
    """
    Patch an official DetectionModel in-place to run our backbone+neck,
    feeding the resulting FPN tensors to the official Detect head.

    Registering our_yolo_nn as a proper submodule ensures .to(device) /
    .half() calls on DetectionModel propagate automatically.
    """
    import types

    _patch_detect_head(official_det_model.model[-1], our_yolo_nn.detect)
    official_det_model.our_yolo_nn = our_yolo_nn  # registered submodule

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        nn = self.our_yolo_nn
        x = nn.b0(x);
        x = nn.b1(x);
        x = nn.b2(x);
        x = nn.b3(x)
        p3 = nn.b4(x)
        x = nn.b5(p3);
        p4 = nn.b6(x)
        x = nn.b7(p4);
        x = nn.b8(x);
        p5 = nn.b9(x)
        x = torch.cat([nn.up1(p5), p4], 1);
        p4_up = nn.n12(x)
        x = torch.cat([nn.up2(p4_up), p3], 1);
        p3_out = nn.n15(x)
        x = torch.cat([nn.n16(p3_out), p4_up], 1);
        p4_out = nn.n18(x)
        x = torch.cat([nn.n19(p4_out), p5], 1);
        p5_out = nn.n21(x)
        return self.model[-1]([p3_out, p4_out, p5_out])

    official_det_model._predict_once = types.MethodType(_predict_once, official_det_model)


# ---------------------------------------------------------------------------
# Convenience: YOLO wrapper for validation
# ---------------------------------------------------------------------------

def build_yolo_with_custom_model(pt_path: str = "yolov8n.pt", nc: int = 80):
    """
    Returns an Ultralytics YOLO object whose inner DetectionModel runs our
    YOLOv8n backbone+neck with the official Detect head for NMS/postprocessing.

    Used by validate_parity.py to run end-to-end mAP evaluation.
    """
    from ultralytics import YOLO

    yolo = YOLO(pt_path)
    our_yolo = YOLOv8nPANOnly(nc=nc)
    # stride is a registered buffer — sync to detect head for the loss/validator
    our_yolo.detect.stride = our_yolo.stride
    load_weights_into_model(our_yolo, pt_path)
    _patch_detection_model(yolo.model, our_yolo)
    return yolo


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import types
    from ultralytics.nn.tasks import DetectionModel

    print("── Weight remapping ──")
    official = DetectionModel("yolov8n.yaml", nc=80, verbose=False)
    our = YOLOv8nPANOnly(nc=80)
    our.detect.stride = our.stride  # sync buffer to head
    remapped = {k: v.float() for k, v in _remap_official_weights(official.state_dict()).items()}
    missing, unexpected = our.load_state_dict(remapped, strict=True)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    assert not missing and not unexpected

    print("\n── Training forward (model(batch_dict)) ──")
    # Set the attributes the trainer normally provides
    our.model = [our.detect]
    our.end2end = False
    our.nc = 80
    our.detect.nc = 80
    our.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, tal_topk=10, tal_topk2=None)
    our.train()
    batch = {
        'img': torch.zeros(1, 3, 640, 640),
        'cls': torch.zeros(2, 1),
        'bboxes': torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.3, 0.1, 0.1]]),
        'batch_idx': torch.tensor([0., 0.]),
    }
    loss, items = our(batch)
    print(f"  loss={loss.tolist()}")

    print("\n── Inference forward (model(tensor)) ──")
    our.eval()
    with torch.no_grad():
        out = our(torch.zeros(1, 3, 640, 640))
    assert out.shape == (1, 84, 8400), f"Got {out.shape}"
    print(f"  output shape: {tuple(out.shape)}  ✅")

    print("\n✅ All adapter checks passed")
