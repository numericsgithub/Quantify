import math
import torch
import torch.nn.functional as F


def _round(x):  # lrintf default is round-half-to-even, which Python round() also does
    return int(round(x))


def _generate_random_crop(orig_w, orig_h, min_rel, max_rel, ar):
    # Direct port of TF GenerateRandomCrop: uniform aspect ratio, then uniform HEIGHT
    # in the valid range (not uniform area, which is what torchvision does).
    min_area = min_rel * orig_w * orig_h
    max_area = max_rel * orig_w * orig_h

    height = _round(math.sqrt(min_area / ar))
    max_height = _round(math.sqrt(max_area / ar))

    if _round(max_height * ar) > orig_w:
        eps = 1e-7
        max_height = int((orig_w + 0.5 - eps) / ar)
    if max_height > orig_h:
        max_height = orig_h
    if height >= max_height:
        height = max_height
    if height < max_height:
        height += int(torch.randint(0, max_height - height + 1, (1,)).item())

    width = _round(height * ar)
    if width <= 0 or height <= 0 or width > orig_w or height > orig_h:
        return None

    x = int(torch.randint(0, orig_w - width + 1, (1,)).item())
    y = int(torch.randint(0, orig_h - height + 1, (1,)).item())
    return x, y, width, height


class SampleDistortedBoundingBoxCrop:
    """Port of tf.image.sample_distorted_bounding_box for a full-image bbox."""
    def __init__(self, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                 area_range=(0.05, 1.0), max_attempts=100):
        self.moc = min_object_covered
        self.ar_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts

    def __call__(self, img):  # img: float tensor [C, H, W]
        _, H, W = img.shape
        img_area = float(W * H)
        for _ in range(self.max_attempts):
            ar = float(torch.empty(1).uniform_(*self.ar_range).item())
            r = _generate_random_crop(W, H, self.area_range[0], self.area_range[1], ar)
            if r is None:
                continue
            x, y, w, h = r
            # full-image bbox: coverage == crop_area / image_area
            if (w * h) / img_area >= self.moc:
                return img[:, y:y + h, x:x + w]
        return img  # use_image_if_no_bounding_boxes=True


class ResizeTF:
    """tf.image.resize_images BILINEAR. Closest PyTorch can get (see caveats)."""
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, img):
        return F.interpolate(
            img.unsqueeze(0), size=self.size, mode="bilinear",
            align_corners=False, antialias=False
        ).squeeze(0)


class RandomHorizontalFlipTF:
    def __call__(self, img):
        if torch.rand(1).item() < 0.5:
            return torch.flip(img, dims=[2])
        return img


class RandomBrightnessTF:
    """tf.image.random_brightness: ADDITIVE delta, no clamping."""
    def __init__(self, max_delta=32.0 / 255.0):
        self.max_delta = max_delta

    def __call__(self, img):
        delta = float(torch.empty(1).uniform_(-self.max_delta, self.max_delta).item())
        return img + delta  # deliberately not clamped, matching TF


class RandomSaturationTF:
    """tf.image.random_saturation: scale S in HSV space (verified equivalent)."""
    def __init__(self, lower=0.5, upper=1.5):
        self.lower, self.upper = lower, upper

    def __call__(self, img):  # img: [C, H, W] in [0, 1]
        k = float(torch.empty(1).uniform_(self.lower, self.upper).item())
        vmax = img.max(0, keepdim=True).values
        vmin = img.min(0, keepdim=True).values
        chroma = (vmax - vmin).clamp_min(1e-12)            # = S * V
        ratio = torch.minimum(torch.full_like(chroma, k), vmax / chroma)
        return vmax - ratio * (vmax - img)                 # exact HSV S-scaling


# Order matters: brightness then saturation, fixed (TF applies them in sequence here).
preprocess = torch.nn.Sequential  # placeholder; compose manually since these aren't nn.Modules

def preprocess_image_for_training(img):  # img: float tensor [C, H, W] in [0, 1]
    img = SampleDistortedBoundingBoxCrop()(img)
    img = ResizeTF((224, 224))(img)
    img = RandomHorizontalFlipTF()(img)
    img = RandomBrightnessTF(32.0 / 255.0)(img)
    img = RandomSaturationTF(0.5, 1.5)(img)
    return img