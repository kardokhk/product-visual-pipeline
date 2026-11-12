"""Automatic quality checks for generated images and videos.

Pixel-level metrics (pure numpy / scikit-image, unit tested):
  palette_delta_e    mean CIEDE2000 distance from the image's dominant colours
                     to the nearest brand palette colour (lower = on-brand)
  edge_ssim          SSIM between Canny edge maps of two images, a measure of
                     preserved geometry for before-and-after pairs (1 = same)
  flicker            mean absolute frame-to-frame change in a video, 0 to 255

Embedding metrics (pretrained encoders, loaded lazily):
  clip_score         100 x cosine(CLIP image, CLIP caption), prompt adherence
  dino_similarity    cosine of DINOv2 CLS embeddings, identity preservation
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color, feature, metrics, transform
from skimage.util import img_as_float


# ----------------------------------------------------------------- helpers
def hex_to_rgb01(hex_code: str) -> np.ndarray:
    h = hex_code.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex colour {hex_code!r}")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def to_rgb01(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(float) / 255.0
    return img_as_float(arr)


def _resize_to(a: np.ndarray, shape) -> np.ndarray:
    if a.shape[:2] == tuple(shape[:2]):
        return a
    return transform.resize(a, shape[:2], anti_aliasing=True)


# ----------------------------------------------------------------- colour
def dominant_colours(img, k: int = 5, seed: int = 0, sample: int = 20000, iters: int = 25):
    """k-means in CIELAB. Returns (centres_lab [k,3], weights [k]) sorted by weight."""
    lab = color.rgb2lab(to_rgb01(img)).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if lab.shape[0] > sample:
        lab = lab[rng.choice(lab.shape[0], sample, replace=False)]
    k = min(k, len(np.unique(lab.round(3), axis=0)))
    centres = lab[rng.choice(lab.shape[0], k, replace=False)]
    for _ in range(iters):
        d = ((lab[:, None, :] - centres[None]) ** 2).sum(-1)
        lbl = d.argmin(1)
        new = np.array([lab[lbl == j].mean(0) if np.any(lbl == j) else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    weights = np.bincount(lbl, minlength=k) / lbl.size
    order = np.argsort(-weights)
    return centres[order], weights[order]


def palette_delta_e(img, palette_hex, k: int = 5, seed: int = 0) -> float:
    """Weighted mean CIEDE2000 from each dominant colour to its nearest brand colour."""
    centres, weights = dominant_colours(img, k=k, seed=seed)
    pal = color.rgb2lab(np.stack([hex_to_rgb01(h) for h in palette_hex])[None])[0]
    de = np.array([[color.deltaE_ciede2000(c, p) for p in pal] for c in centres]).min(1)
    return float((de * weights).sum())


# ----------------------------------------------------------------- structure
def edge_ssim(a, b, sigma: float = 2.0, blur: float = 2.0) -> float:
    """SSIM between softened Canny edge maps of a and b (b resized to a)."""
    ga = color.rgb2gray(to_rgb01(a))
    gb = _resize_to(color.rgb2gray(to_rgb01(b)), ga.shape)
    from scipy.ndimage import gaussian_filter
    ea = gaussian_filter(feature.canny(ga, sigma=sigma).astype(float), blur)
    eb = gaussian_filter(feature.canny(gb, sigma=sigma).astype(float), blur)
    rng = max(ea.max(), eb.max(), 1e-8)
    return float(metrics.structural_similarity(ea, eb, data_range=rng))


def flicker(frames) -> float:
    """Mean absolute difference between consecutive frames, on a 0 to 255 scale."""
    f = np.asarray(frames, dtype=float)
    if f.ndim != 4 or f.shape[0] < 2:
        raise ValueError("expected frames array [T,H,W,C] with T >= 2")
    if f.max() <= 1.0:
        f = f * 255.0
    return float(np.abs(np.diff(f, axis=0)).mean())


# ----------------------------------------------------------------- embeddings
class Encoders:
    """CLIP ViT-L/14 and DINOv2-base, loaded once and reused."""

    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    def detect(self, img, query: str):
        """Highest-scoring OWLv2 box (left, top, right, bottom) for a text query, or None."""
        t = self.torch
        if not hasattr(self, "owl"):
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device).eval()
            self.owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        im = self._pil(img)
        with t.no_grad():
            inp = self.owl_proc(text=[[query]], images=im, return_tensors="pt").to(self.device)
            out = self.owl(**inp)
        # OWLv2 pads the image to a square, so boxes are scaled by the longer side
        side = max(im.size)
        res = self.owl_proc.image_processor.post_process_object_detection(
            out, threshold=0.0, target_sizes=t.tensor([[side, side]], device=self.device))[0]
        if len(res["scores"]) == 0:
            return None
        b = res["boxes"][int(res["scores"].argmax())].tolist()
        l, tp, r, bt = max(b[0], 0), max(b[1], 0), min(b[2], im.width), min(b[3], im.height)
        return (round(l), round(tp), round(r), round(bt))

    def detections(self, img, query: str):
        """All OWLv2 boxes and scores for a query, after non-maximum suppression (IoU 0.3)."""
        t = self.torch
        from torchvision.ops import nms
        self.detect(img, query)  # ensures the detector is loaded
        im = self._pil(img)
        with t.no_grad():
            inp = self.owl_proc(text=[[query]], images=im, return_tensors="pt").to(self.device)
            out = self.owl(**inp)
        side = max(im.size)
        res = self.owl_proc.image_processor.post_process_object_detection(
            out, threshold=0.05, target_sizes=t.tensor([[side, side]], device=self.device))[0]
        keep = nms(res["boxes"], res["scores"], 0.3)
        return res["boxes"][keep].cpu().numpy(), res["scores"][keep].cpu().numpy()

    def count(self, img, query: str, rel: float = 0.5) -> int:
        """Objects matching the query whose score is at least `rel` x the top score."""
        _, scores = self.detections(img, query)
        if len(scores) == 0:
            return 0
        return int((scores >= rel * scores.max()).sum())

    def _pil(self, img):
        return img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)).convert("RGB")

    def clip_score(self, img, caption: str) -> float:
        t = self.torch
        with t.no_grad():
            inp = self.clip_proc(text=[caption], images=self._pil(img), return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = self.clip(**inp)
            i = t.nn.functional.normalize(out.image_embeds, dim=-1)
            s = t.nn.functional.normalize(out.text_embeds, dim=-1)
        return float(100.0 * (i * s).sum())

    def dino_embed(self, imgs):
        t = self.torch
        with t.no_grad():
            inp = self.dino_proc(images=[self._pil(x) for x in imgs], return_tensors="pt").to(self.device)
            cls = self.dino(**inp).last_hidden_state[:, 0]
        return t.nn.functional.normalize(cls, dim=-1).cpu().numpy()

    def dino_similarity(self, a, b) -> float:
        e = self.dino_embed([a, b])
        return float((e[0] * e[1]).sum())

    def video_identity(self, frames, reference, every: int = 8):
        """DINOv2 similarity of sampled frames to the reference still: (mean, min)."""
        sel = list(frames[::every]) + [frames[-1]]
        e = self.dino_embed([reference] + sel)
        sims = e[1:] @ e[0]
        return float(sims.mean()), float(sims.min())


# ----------------------------------------------------------------- product region
def foreground_bbox(img, thresh: float = 12.0, pad: float = 0.06):
    """Bounding box (left, top, right, bottom) of the product on a plain backdrop.

    The backdrop colour is the median of the border pixels; pixels further than
    `thresh` CIE76 units from it are foreground. The largest connected
    component is kept and its box padded by `pad` of the image size.
    """
    from scipy import ndimage

    lab = color.rgb2lab(to_rgb01(img))
    h, w = lab.shape[:2]
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    mask = np.sqrt(((lab - bg) ** 2).sum(-1)) > thresh
    mask = ndimage.binary_opening(mask, iterations=2)
    lbl, n = ndimage.label(mask)
    if n == 0:
        return (0, 0, w, h)
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == (int(np.argmax(sizes)) + 1))
    py, px = int(pad * h), int(pad * w)
    return (max(xs.min() - px, 0), max(ys.min() - py, 0), min(xs.max() + 1 + px, w), min(ys.max() + 1 + py, h))


def crop_rel(img: Image.Image, box, ref_size) -> Image.Image:
    """Crop `img` with a box defined on an image of size ref_size (w, h)."""
    sx, sy = img.width / ref_size[0], img.height / ref_size[1]
    l, t, r, b = box
    return img.crop((round(l * sx), round(t * sy), round(r * sx), round(b * sy)))


def core_colour_lab(img, box=None, frac: float = 0.5) -> np.ndarray:
    """Median CIELAB colour of the central `frac` (by width and height) of a box."""
    rgb = to_rgb01(img)
    h, w = rgb.shape[:2]
    l, t, r, b = box if box is not None else (0, 0, w, h)
    cw, ch = (r - l) * frac, (b - t) * frac
    cx, cy = (l + r) / 2, (t + b) / 2
    x0, x1 = int(round(cx - cw / 2)), int(round(cx + cw / 2))
    y0, y1 = int(round(cy - ch / 2)), int(round(cy + ch / 2))
    core = rgb[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
    return np.median(color.rgb2lab(core).reshape(-1, 3), axis=0)


def product_colour_de(src, box_src, out, box_out, frac: float = 0.5) -> float:
    """CIEDE2000 between the median core colours of the product in two images."""
    return float(color.deltaE_ciede2000(core_colour_lab(src, box_src, frac), core_colour_lab(out, box_out, frac)))
