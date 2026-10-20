"""
Trainable bubble instance segmentation that handles overlapping bubbles.

Model: torchvision Mask R-CNN (ResNet50-FPN v2, COCO-pretrained). It predicts one mask per
object, so masks may overlap (nested / crossing bubbles). Masks are free-form (non-spherical
bubbles). The FPN with custom anchor sizes, plus upsampling and scale-jitter augmentation,
handles the large size range.

Annotations are napari *Shapes*: one circle or ellipse per bubble, overlaps allowed. A circle is
an ellipse with equal axes. They are stored as JSON next to the image they belong to:

    {
      "image": "frame.tif",                 # relative to the JSON file
      "bubbles": [{"type": "ellipse", "data": [[row, col] x 4]}, ...],   # corners of the bounding box
      "rois": [[r0, c0, r1, c1], ...]       # fully annotated regions; [] = whole image
    }

The model's predicted masks are saved the same way, as circles and ellipses (masks_to_shapes).
Polygons in older files are read as the ellipse with the same area, centre and orientation (as_ellipse).

All annotation coordinates are in NATIVE image pixels (napari convention: pixel centres at
integer coordinates). The model works at ``model_scale`` x native resolution.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
import pandas as pd
from skimage import draw, measure, transform

import bubble_seg as bs


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class RCNNConfig:
    # --- data ---
    annotation_dir: str = "annotations"
    model_dir: str = "models"
    model_scale: float = 3.0               # native -> model pixels (small bubbles need upsampling)
    channels: tuple = ("norm", "clahe", "ridge")  # 3 input channels (from bubble_seg.preprocess)
    val_fraction: float = 0.2               # fraction of annotated ROIs held out for validation

    # --- model ---
    pretrained: bool = True                 # COCO weights (downloaded once by torchvision)
    anchor_sizes: tuple = ((12,), (24,), (48,), (96,), (192,))   # model px, one tuple per FPN level
    anchor_ratios: tuple = (0.5, 1.0, 2.0)
    trainable_backbone_layers: int = 5
    box_nms_thresh: float = 0.7             # high: overlapping bubbles have overlapping boxes
    detections_per_img: int = 1000           # max bubbles per frame (above score_thresh); dense frames have > 400
    rpn_post_nms_top_n_test: int = 2000
    rpn_pre_nms_top_n_test: int = 4000

    # --- training ---
    device: str = "cuda"
    crop_size: int = 384                    # model px
    batch_size: int = 4
    iterations: int = 3000
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 200
    amp: bool = True                        # mixed precision on GPU
    num_workers: int = 4
    val_every: int = 250
    seed: int = 0
    # augmentation
    scale_jitter: tuple = (0.6, 1.6)
    rotate90: bool = True
    flip: bool = True
    intensity_jitter: float = 0.15          # brightness / contrast / gamma amplitude
    noise_std: float = 0.03
    min_instance_area: float = 12.0         # model px^2; smaller (cropped) instances are dropped

    # --- inference ---
    score_thresh: float = 0.5
    mask_thresh: float = 0.5
    mask_nms_iou: float = 0.7               # duplicate removal on masks (overlapping bubbles are ~<0.6)
    um_per_px: float = 3.2

    # --- shape of the saved outlines (masks_to_shapes) ---
    circle_min_axis_ratio: float = 0.93     # kind="auto": fitted ellipses rounder than this (minor/major) are saved as circles

    # --- bubbles cut by the image edge ---
    edge_min_fit_arc_deg: float = 45.0      # visible arc below which the fit is a circle through the arc only (a sliver
                                            # at the edge: its size is uncertain and it gets fit_reliable=False)
    edge_min_arc_deg: float = 100.0         # visible arc needed to even try an ellipse; shorter arcs -> circle
    edge_ellipse_gain: float = 1.6          # use the ellipse only if its RMS residual is this many times smaller
                                            # than the circle's (otherwise the circle is more reliable)
    edge_max_axis_ratio: float = 2.5        # reject edge-ellipse fits more elongated than this (-> circle)
    edge_reliable_arc_deg: float = 140.0    # edge fits from a shorter visible arc are flagged fit_reliable=False
    drop_unreliable_fits: bool = False      # exclude edge bubbles with fit_reliable=False from the analysis
    drop_edge_bubbles: bool = False         # internal boundary option 1: exclude every bubble touching the edge
    inner_margin: float = 0.0               # internal boundary option 2 (native px): exclude bubbles whose
                                            # (fitted) centre lies closer than this to the edge (or outside)

    # --- preprocessing passed to bubble_seg (lengths in native px) ---
    prep: dict = field(default_factory=lambda: dict(
        bg_sigma=24.0, denoise="gaussian", denoise_sigma=1.0, use_clahe=True,
        clahe_kernel=51.2, clahe_clip=0.01, ridge_sigmas=(0.8, 1.2, 1.6), exclude_boxes=[]))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        # settings that no longer exist (e.g. the old polygon tolerances) are ignored, new ones get their default
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        for k in ("channels", "anchor_ratios", "scale_jitter"):
            if k in d:
                d[k] = tuple(d[k])
        if "anchor_sizes" in d:
            d["anchor_sizes"] = tuple(tuple(a) for a in d["anchor_sizes"])
        return cls(**d)


# ----------------------------------------------------------------------------
# Image preparation
# ----------------------------------------------------------------------------
def prepare_image(img: np.ndarray, cfg: RCNNConfig) -> np.ndarray:
    """Native float image -> (3, H*s, W*s) float32 model input in [0, 1]."""
    pcfg = bs.Config(work_scale=cfg.model_scale, **cfg.prep)
    valid = bs.build_valid_mask(img.shape, pcfg)
    pre = bs.preprocess(img, valid, pcfg)
    chans = []
    for c in cfg.channels:
        if c == "ridge":
            chans.append(np.clip(pre["ridge_z"] / 10.0, 0, 1))
        else:
            chans.append(pre[c])
    return np.stack(chans).astype(np.float32)


def native_to_model(coords, s):
    """Native (row, col) -> model coords (pixel-centre convention, like skimage.resize)."""
    return np.asarray(coords, float) * s + (s - 1) / 2.0


def model_to_native(coords, s):
    return (np.asarray(coords, float) - (s - 1) / 2.0) / s


# ----------------------------------------------------------------------------
# Annotation I/O
# ----------------------------------------------------------------------------
def ellipse_shape(xc, yc, a, b, th, fit_kind: Optional[str] = None) -> dict:
    """napari ellipse (4 corners of its bounding box, (row, col)) from centre (x, y), semi-axes a, b and the angle
    of the a axis (rad, from the x axis). a == b gives a circle."""
    ca, sa = np.cos(th), np.sin(th)
    ux, uy = np.array([ca, sa]) * a, np.array([-sa, ca]) * b
    corners_xy = [(xc - ux[0] - uy[0], yc - ux[1] - uy[1]), (xc + ux[0] - uy[0], yc + ux[1] - uy[1]),
                  (xc + ux[0] + uy[0], yc + ux[1] + uy[1]), (xc - ux[0] + uy[0], yc - ux[1] + uy[1])]
    sh = dict(type="ellipse", data=[[float(y), float(x)] for x, y in corners_xy])
    if fit_kind:
        sh["fit_kind"] = fit_kind
    return sh


def ellipse_params(shape: dict):
    """napari ellipse -> centre x, y, semi-axes a >= b, angle of the major axis (rad, from the x axis).
    napari draws c + cos(t) e1 + sin(t) e2 from the corners of its bounding box; after a group resize these
    corners form a parallelogram, so the axes are the singular values of [e1 e2], not the side lengths."""
    d = np.asarray(shape["data"], float)[:, -2:]
    c = d.mean(axis=0)
    e1, e2 = (d[1] - d[0]) / 2.0, (d[3] - d[0]) / 2.0          # (row, col) half sides of the bounding box
    U, S, _ = np.linalg.svd(np.c_[e1, e2])
    return float(c[1]), float(c[0]), float(S[0]), float(S[1]), float(np.arctan2(U[0, 0], U[1, 0]))


def moment_ellipse(xy) -> Optional[tuple]:
    """Ellipse with the area, centroid and orientation (second moments) of a closed polygon given as (x, y)
    vertices: (x, y, a, b, theta) or None if the polygon is degenerate."""
    x, y = np.asarray(xy, float).T
    if len(x) < 3:
        return None
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    if abs(A) < 1e-6:
        return None
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    sxx = ((x ** 2 + x * x1 + x1 ** 2) * cr).sum() / (12 * A) - cx ** 2
    syy = ((y ** 2 + y * y1 + y1 ** 2) * cr).sum() / (12 * A) - cy ** 2
    sxy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / (24 * A) - cx * cy
    ev, evec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
    if not np.isfinite(ev).all() or ev[0] <= 0:
        return None
    a, b = 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0])                  # filled ellipse: variance = semi-axis^2 / 4
    k = np.sqrt(abs(A) / (np.pi * a * b))                          # keep the polygon's area
    return float(cx), float(cy), float(a * k), float(b * k), float(np.arctan2(evec[1, 1], evec[0, 1]))


def as_ellipse(shape: dict) -> Optional[dict]:
    """The shape as a napari ellipse. Ellipses (and circles) are returned unchanged; polygons from older files
    become the ellipse with the same area, centre and orientation (fit_kind 'from_polygon'); None if that
    is not possible (degenerate polygon, or a shape type that is not a bubble outline)."""
    t = shape.get("type")
    if t == "ellipse":
        return shape
    if t in ("polygon", "path"):
        p = moment_ellipse(np.asarray(shape["data"], float)[:, -2:][:, ::-1])
        if p is None:
            return None
        return {**{k: v for k, v in shape.items() if k not in ("type", "data", "fit_kind")},
                **ellipse_shape(*p, fit_kind="from_polygon")}
    return None


def as_ellipses(shapes: list, warn: str = "") -> list:
    """as_ellipse for a list, keeping the list positions (bubble_id = position in the list). Shapes that cannot
    be converted are kept as they are and skipped by the measurements."""
    out = [as_ellipse(s) or s for s in shapes]
    n = sum(s.get("type") in ("polygon", "path") for s in shapes)
    if n and warn:
        print(f"{warn}: {n} polygon(s) from an older version read as ellipses")
    return out


def shape_outline(shape: dict, n_ellipse: int = 128) -> np.ndarray:
    """Return the outline (N, 2) in (row, col) of a circle / ellipse annotation shape (older polygons: see as_ellipse)."""
    sh = as_ellipse(shape)
    if sh is None:
        raise ValueError(f"cannot use shape of type {shape.get('type')!r} as a bubble (circles and ellipses only)")
    # napari stores ellipses as the 4 corners of their (possibly rotated) bounding box
    d = np.asarray(sh["data"], float)[:, -2:]
    c = d.mean(axis=0)
    e1, e2 = (d[1] - d[0]) / 2.0, (d[3] - d[0]) / 2.0
    th = np.linspace(0, 2 * np.pi, n_ellipse, endpoint=False)
    return c + np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2


def rasterize(outline_rc: np.ndarray, shape_hw, s: float) -> np.ndarray:
    """Native-coordinate outline -> boolean mask at model resolution."""
    p = native_to_model(outline_rc, s)
    m = np.zeros(shape_hw, bool)
    rr, cc = draw.polygon(p[:, 0], p[:, 1], shape_hw)
    m[rr, cc] = True
    return m


def load_annotation(json_path: str) -> dict:
    with open(json_path) as f:
        ann = json.load(f)
    ann["json_path"] = json_path
    ann["image_path"] = os.path.join(os.path.dirname(json_path), ann["image"])
    ann.setdefault("rois", [])
    bub = as_ellipses(ann.get("bubbles", []), warn=os.path.basename(json_path))
    ann["bubbles"] = [b for b in bub if b.get("type") == "ellipse"]
    return ann


def save_annotation(json_path: str, image_name: str, bubbles: list, rois: list, **extra):
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(image=image_name, bubbles=bubbles, rois=rois, **extra), f)
    os.replace(tmp, json_path)


def content_key(bubbles: list, rois: list) -> str:
    """Fingerprint of an annotation's shapes and ROIs (rounded to 1/1000 px): tells real edits from re-saves."""
    import hashlib

    def rnd(v):
        return [round(float(x), 3) for x in v]
    payload = ([[b.get("type"), [rnd(p) for p in b.get("data", [])]] for b in bubbles], [rnd(r) for r in rois])
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:16]


def list_annotations(cfg: RCNNConfig) -> list:
    """Annotated frames used for training. Model drafts staged for correction (bubble_training.ipynb section 8)
    are used only once you ticked Reviewed in napari (the whole frame is corrected) or drew 'fully annotated'
    rectangles (only those regions are corrected), so uncorrected predictions never become training data."""
    out, unchecked = [], []
    for p in sorted(glob.glob(os.path.join(cfg.annotation_dir, "*.json"))):
        a = load_annotation(p)
        if a.get("draft") and not (a.get("reviewed") or a["rois"]):
            unchecked.append(os.path.basename(p))
        elif len(a["bubbles"]):
            out.append(a)
    if unchecked:
        print(f"{len(unchecked)} drafted frame(s) left out until you tick Reviewed (or draw 'fully annotated' "
              f"rectangles): {', '.join(unchecked[:5])}" + (" ..." if len(unchecked) > 5 else ""))
    return out


def stage_frames_for_annotation(frame_paths, cfg: RCNNConfig) -> list:
    """Copy frames (as float32 TIFF) into the annotation folder and create empty JSONs."""
    import tifffile
    os.makedirs(cfg.annotation_dir, exist_ok=True)
    out = []
    for p in frame_paths:
        name = os.path.splitext(os.path.basename(p))[0] + ".tif"
        dst = os.path.join(cfg.annotation_dir, name)
        if not os.path.exists(dst):
            tifffile.imwrite(dst, bs.load_image(p))
        js = os.path.join(cfg.annotation_dir, os.path.splitext(name)[0] + ".json")
        if not os.path.exists(js):
            save_annotation(js, name, [], [], frame_path=os.path.abspath(p))
        out.append(js)
    return out


# ----------------------------------------------------------------------------
# Training samples
# ----------------------------------------------------------------------------
class Sample:
    """One annotated region (ROI) at model resolution: image (3,h,w) + instance masks (n,h,w)."""

    def __init__(self, image, masks, name):
        self.image = image
        self.masks = masks
        self.name = name


def annotation_image(ann: dict, verbose: bool = True) -> str:
    """Path of an annotation's frame: the TIFF next to the JSON, else the original frame (frame_path, or
    raw/jpg_<run>/<name>.jpg derived from the name), e.g. on a machine where the TIFFs (not in git) are missing."""
    p = ann["image_path"]
    if os.path.exists(p):
        return p
    name = os.path.splitext(os.path.basename(p))[0]
    for c in [ann.get("frame_path") or ""] + sorted(glob.glob(os.path.join("raw", "jpg_" + name.split("_tid")[0],
                                                                          name + ".*"))):
        if c and os.path.exists(c):
            if verbose:
                print(f"{os.path.basename(p)} not found, using {c}")
            return c
    raise FileNotFoundError(f"frame image not found: {p} (nor its original in raw/)")


def build_samples(annotations: list, cfg: RCNNConfig) -> list:
    """Prepare every annotated ROI as a Sample (whole image if no ROI was drawn)."""
    s = cfg.model_scale
    samples = []
    for ann in annotations:
        img = bs.load_image(annotation_image(ann))
        x = prepare_image(img, cfg)
        H, W = x.shape[1:]
        outlines = [shape_outline(b) for b in ann["bubbles"]]
        rois = ann["rois"] or [[-0.5, -0.5, img.shape[0] - 0.5, img.shape[1] - 0.5]]   # the whole frame
        for k, (r0, c0, r1, c1) in enumerate(rois):
            (mr0, mc0), (mr1, mc1) = native_to_model([[r0, c0], [r1, c1]], s)
            mr0, mc0 = max(0, int(np.floor(mr0))), max(0, int(np.floor(mc0)))
            mr1, mc1 = min(H, int(np.ceil(mr1)) + 1), min(W, int(np.ceil(mc1)) + 1)
            masks = []
            for o in outlines:
                # cheap bbox test before rasterising
                oo = native_to_model(o, s)
                if oo[:, 0].max() < mr0 or oo[:, 0].min() > mr1 or oo[:, 1].max() < mc0 or oo[:, 1].min() > mc1:
                    continue
                m = rasterize(o, (H, W), s)[mr0:mr1, mc0:mc1]
                if m.sum() >= cfg.min_instance_area:
                    masks.append(m)
            masks = np.array(masks, bool).reshape(-1, mr1 - mr0, mc1 - mc0)
            samples.append(Sample(x[:, mr0:mr1, mc0:mc1].copy(), masks,
                                  f"{os.path.basename(ann['json_path'])}#roi{k}"))
    return samples


def _unit_hash(name: str, seed: int) -> float:
    import hashlib
    return int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_samples(samples: list, cfg: RCNNConfig):
    """Hold out about val_fraction of the annotated frames for validation (all regions of a frame together; with
    fewer than 3 frames, single regions). Whether a frame is held out depends (almost) only on its name, so adding
    frames does not move frames from training to validation, and a model is not evaluated on frames it was trained on.
    Needs >= 3 regions; with fewer, everything is used for training (no validation)."""
    if len(samples) < 3 or cfg.val_fraction <= 0:
        print(f"{len(samples)} annotated region(s): no validation split (draw >= 3 'fully annotated' rectangles to get one)")
        return list(samples), []
    frame_of = lambda smp: smp.name.split("#")[0]
    by_frame = len({frame_of(smp) for smp in samples}) >= 3
    key = frame_of if by_frame else (lambda smp: smp.name)
    keys = sorted({key(smp) for smp in samples})
    h = {k: _unit_hash(k, cfg.seed) for k in keys}
    val_keys = {k for k in keys if h[k] < cfg.val_fraction} or {min(keys, key=h.get)}
    if len(val_keys) == len(keys):
        val_keys.discard(max(val_keys, key=h.get))
    val = [smp for smp in samples if key(smp) in val_keys]
    train = [smp for smp in samples if key(smp) not in val_keys]
    print(f"validation: {len(val_keys)} of {len(keys)} {'frames' if by_frame else 'regions'} ({len(val)} regions)")
    return train, val


def _resize_chw(x, f, order):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    size = (max(1, int(round(x.shape[1] * f))), max(1, int(round(x.shape[2] * f))))
    mode = "bilinear" if order == 1 else "nearest"
    kw = dict(align_corners=False) if mode == "bilinear" else {}
    return F.interpolate(t, size=size, mode=mode, **kw)[0].numpy()


def augment(sample: Sample, cfg: RCNNConfig, rng: np.random.Generator):
    """Random scale + crop + flips/rot90 + intensity changes. Returns (image, masks)."""
    img, masks = sample.image, sample.masks
    cs = cfg.crop_size
    f = float(np.exp(rng.uniform(np.log(cfg.scale_jitter[0]), np.log(cfg.scale_jitter[1]))))
    # crop window in the ORIGINAL sample so we only resize what we need
    win = int(math.ceil(cs / f))
    h, w = img.shape[1:]
    r0 = int(rng.integers(0, max(1, h - win + 1)))
    c0 = int(rng.integers(0, max(1, w - win + 1)))
    img = img[:, r0:r0 + win, c0:c0 + win]
    masks = masks[:, r0:r0 + win, c0:c0 + win]
    if len(masks):                                   # only instances inside the window (saves memory / time)
        masks = masks[masks.reshape(len(masks), -1).any(1)]
    img = _resize_chw(img, f, 1)
    if len(masks):
        masks = _resize_chw(masks.astype(np.float32), f, 1) > 0.5
    else:
        masks = np.zeros((0,) + img.shape[1:], bool)
    if cfg.flip and rng.random() < 0.5:
        img, masks = img[:, :, ::-1], masks[:, :, ::-1]
    if cfg.flip and rng.random() < 0.5:
        img, masks = img[:, ::-1, :], masks[:, ::-1, :]
    if cfg.rotate90:
        k = int(rng.integers(0, 4))
        img, masks = np.rot90(img, k, axes=(1, 2)), np.rot90(masks, k, axes=(1, 2))
    a = cfg.intensity_jitter
    if a > 0:
        gain = 1 + rng.uniform(-a, a, (img.shape[0], 1, 1))
        bias = rng.uniform(-a, a, (img.shape[0], 1, 1)) * 0.5
        gamma = np.exp(rng.uniform(-a, a))
        img = np.clip(img * gain + bias, 0, 1) ** gamma
    if cfg.noise_std > 0:
        img = img + rng.normal(0, cfg.noise_std * rng.random(), img.shape)
    img = np.clip(img, 0, 1).astype(np.float32)
    keep = masks.reshape(len(masks), -1).sum(1) >= cfg.min_instance_area if len(masks) else np.zeros(0, bool)
    return np.ascontiguousarray(img), np.ascontiguousarray(masks[keep])


def to_target(masks: np.ndarray):
    import torch
    n = len(masks)
    boxes = np.zeros((n, 4), np.float32)
    for i, m in enumerate(masks):
        rr, cc = np.nonzero(m)
        boxes[i] = (cc.min(), rr.min(), cc.max() + 1, rr.max() + 1)
    return dict(boxes=torch.from_numpy(boxes), labels=torch.ones(n, dtype=torch.int64),
                masks=torch.from_numpy(masks.astype(np.uint8)))


class TrainDataset:
    """Map-style dataset that draws random augmented crops (index is ignored)."""

    def __init__(self, samples, cfg, length):
        self.samples, self.cfg, self.length = samples, cfg, length
        # sample ROIs proportionally to their area so big ROIs are not under-used
        a = np.array([s.image.shape[1] * s.image.shape[2] for s in samples], float)
        self.p = a / a.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import torch
        rng = np.random.default_rng((self.cfg.seed, idx))
        s = self.samples[rng.choice(len(self.samples), p=self.p)]
        img, masks = augment(s, self.cfg, rng)
        return torch.from_numpy(img), to_target(masks)


def _collate(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _identity_resize(image, target=None):
    return image, target


def build_model(cfg: RCNNConfig):
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    torch.manual_seed(cfg.seed)                      # reproducible initialisation of the new heads
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if cfg.pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None,
        trainable_backbone_layers=cfg.trainable_backbone_layers if cfg.pretrained else None,
        box_nms_thresh=cfg.box_nms_thresh, box_detections_per_img=cfg.detections_per_img,
        box_score_thresh=0.05,
        rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test, rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test)
    # two classes: background + bubble
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    # anchors matched to bubble sizes (same number per location -> pretrained RPN head is reused)
    assert len(cfg.anchor_sizes) == 5 and all(len(a) == 1 for a in cfg.anchor_sizes), \
        "anchor_sizes: 5 levels with one size each (keeps the pretrained RPN head)"
    model.rpn.anchor_generator = AnchorGenerator(cfg.anchor_sizes, (tuple(cfg.anchor_ratios),) * 5)
    # we control the resolution ourselves (model_scale); disable torchvision's internal resize
    model.transform.resize = _identity_resize
    # images are greyscale-derived [0,1] channels: normalise with neutral statistics
    model.transform.image_mean = [0.5, 0.5, 0.5]
    model.transform.image_std = [0.25, 0.25, 0.25]
    return model


def get_device(cfg: RCNNConfig):
    import torch
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available -> using CPU (slow)")
        return torch.device("cpu")
    device = torch.device(cfg.device)
    if device.type == "cuda":
        # a PyTorch build only runs on the GPU generations it was compiled for (sm_XY), or newer ones via PTX
        major, minor = torch.cuda.get_device_capability(device)
        cap, archs = major * 10 + minor, torch.cuda.get_arch_list()
        sm = {int(a[3:]) for a in archs if a.startswith("sm_")}
        ptx = {int(a[8:]) for a in archs if a.startswith("compute_")}
        if cap not in sm and not any(p <= cap for p in ptx):
            print(f"This PyTorch build ({torch.__version__}, built for {' '.join(archs)}) has no kernels for "
                  f"{torch.cuda.get_device_name(device)} (sm_{cap}) -> using CPU (slow).\n"
                  "Fix: start the JupyterHub session on an A100/H100 node, or install a PyTorch build that "
                  "supports this GPU (see README).")
            return torch.device("cpu")
    return device


def save_model(model, cfg: RCNNConfig, name="bubble_rcnn"):
    import torch
    os.makedirs(cfg.model_dir, exist_ok=True)
    path = os.path.join(cfg.model_dir, name + ".pt")
    torch.save(model.state_dict(), path)
    cfg.to_json(os.path.join(cfg.model_dir, name + "_config.json"))
    return path


def config_path(model_path: str) -> str:
    """models/bubble_rcnn_best.pt -> models/bubble_rcnn_best_config.json"""
    return os.path.splitext(model_path)[0] + "_config.json"


def load_model(path, cfg: Optional[RCNNConfig] = None, device=None):
    import torch
    if cfg is None:
        cfg = RCNNConfig.from_json(config_path(path))
    c = RCNNConfig(**{**asdict(cfg), "pretrained": False})
    model = build_model(c)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    device = device or get_device(cfg)
    return model.to(device).eval(), cfg


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(model, train_samples, val_samples, cfg: RCNNConfig, log_every=25, ckpt_name="bubble_rcnn"):
    import torch
    device = get_device(cfg)
    model.to(device)
    ds = TrainDataset(train_samples, cfg, cfg.iterations * cfg.batch_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers, collate_fn=_collate,
                                     persistent_workers=cfg.num_workers > 0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    def lr_at(it):
        if it < cfg.warmup_iters:
            return cfg.lr * (0.001 + 0.999 * it / cfg.warmup_iters)
        p = (it - cfg.warmup_iters) / max(1, cfg.iterations - cfg.warmup_iters)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * p))

    if not train_samples:
        raise ValueError("no training samples")
    if not val_samples:
        print(f"no validation set: {ckpt_name}_best.pt will be the same as {ckpt_name}_last.pt (not selected by F1)")
    torch.manual_seed(cfg.seed)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_f1 = [], -1.0
    t0 = time.time()
    model.train()
    for it, (imgs, targets) in enumerate(dl):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            losses = model(imgs, targets)
            loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt)
        scaler.update()
        rec = dict(iter=it, lr=lr_at(it), loss=loss.item(), **{k: v.item() for k, v in losses.items()})
        if (it + 1) % cfg.val_every == 0 or it + 1 == cfg.iterations:
            if val_samples:
                m = evaluate(model, val_samples, cfg)
                rec.update(val_precision=m["precision"], val_recall=m["recall"], val_f1=m["f1"])
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    save_model(model, cfg, ckpt_name + "_best")
            else:            # no validation: _best = _last, so an older _best.pt is never used by mistake
                save_model(model, cfg, ckpt_name + "_best")
            save_model(model, cfg, ckpt_name + "_last")
            model.train()
        history.append(rec)
        if it % log_every == 0 or "val_f1" in rec:
            msg = f"it {it:5d}  loss {rec['loss']:.3f}  lr {rec['lr']:.4f}  {time.time() - t0:6.0f}s"
            if "val_f1" in rec:
                msg += f"  | val P {rec['val_precision']:.2f} R {rec['val_recall']:.2f} F1 {rec['val_f1']:.2f}"
            print(msg)
    return pd.DataFrame(history)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def _bboxes(masks: np.ndarray) -> np.ndarray:
    """(n, 4) bounding boxes [r0, r1, c0, c1) of boolean masks (empty masks: an empty box)."""
    out = np.zeros((len(masks), 4), int)
    if len(masks):
        rows, cols = masks.any(axis=2), masks.any(axis=1)
        for i in range(len(masks)):
            r, c = np.flatnonzero(rows[i]), np.flatnonzero(cols[i])
            if len(r):
                out[i] = (r[0], r[-1] + 1, c[0], c[-1] + 1)
    return out


def _pair_iou(a, b, ba, bb, area_a, area_b) -> float:
    """IoU of two masks, computed only where their bounding boxes overlap."""
    r0, r1, c0, c1 = max(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), min(ba[3], bb[3])
    if r0 >= r1 or c0 >= c1:
        return 0.0
    inter = np.count_nonzero(a[r0:r1, c0:c1] & b[r0:r1, c0:c1])
    return inter / max(area_a + area_b - inter, 1)


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on masks; returns kept indices (sorted by score)."""
    order = np.argsort(-scores)
    boxes = _bboxes(masks)
    areas = masks.reshape(len(masks), -1).sum(1)
    keep = []
    for i in order:
        if all(_pair_iou(masks[i], masks[j], boxes[i], boxes[j], areas[i], areas[j]) <= iou_thr for j in keep):
            keep.append(i)
    return np.array(keep, int)


def predict(model, x: np.ndarray, cfg: RCNNConfig, score_thresh=None):
    """x: (3,H,W) model input. Returns dict(masks (n,H,W) bool, scores (n,))."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    st = cfg.score_thresh if score_thresh is None else score_thresh
    # keep only boxes above the threshold before the masks are made (memory), up to detections_per_img bubbles
    model.roi_heads.score_thresh = float(st)
    model.roi_heads.detections_per_img = int(cfg.detections_per_img)
    with torch.no_grad():
        out = model([torch.from_numpy(x).to(device)])[0]
    keep = out["scores"] >= st
    scores = out["scores"][keep].float().cpu().numpy()
    masks = (out["masks"][keep, 0] >= cfg.mask_thresh).cpu().numpy().reshape(-1, *x.shape[1:])
    nonempty = masks.reshape(len(masks), -1).any(1) if len(masks) else np.zeros(0, bool)
    masks, scores = masks[nonempty], scores[nonempty]
    if len(masks):
        k = mask_nms(masks, scores, cfg.mask_nms_iou)
        masks, scores = masks[k], scores[k]
    return dict(masks=masks, scores=scores)


def _fit_circle(xy):
    """Geometric least-squares circle fit (algebraic start). Returns xc, yc, r."""
    from scipy.optimize import least_squares
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    (a, b, c), *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    x0 = np.array([a / 2, b / 2, np.sqrt(max(c + a * a / 4 + b * b / 4, 1e-9))])
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], x0)
    return float(res.x[0]), float(res.x[1]), float(abs(res.x[2]))


def _fit_ellipse(xy) -> Optional[tuple]:
    """Direct least-squares ellipse fit to points (x, y) (Fitzgibbon et al., in the numerically stable form of
    Halir & Flusser), real arithmetic only. Returns (x, y, a, b, theta) - centre, semi-axes, angle of the a axis
    from the x axis - or None. (skimage's EllipseModel is not used: with some numpy / scikit-image versions it
    raises or fails on every input.)"""
    xy = np.asarray(xy, float)
    if len(xy) < 6:
        return None
    o = xy.mean(axis=0)
    sc = float(np.sqrt(((xy - o) ** 2).sum(axis=1).mean())) or 1.0
    x, y = ((xy - o) / sc).T                                   # centred and scaled for conditioning
    D1 = np.c_[x * x, x * y, y * y]
    D2 = np.c_[x, y, np.ones_like(x)]
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        M = np.array([M[2] / 2, -M[1], M[0] / 2])
        _, v = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    if not (cond > 0).any():
        return None
    A, B, C = v[:, int(np.argmax(cond > 0))]
    D, E, F = T @ np.array([A, B, C])
    den = B * B - 4 * A * C                                   # < 0 for an ellipse
    if den >= 0:
        return None
    x0, y0 = (2 * C * D - B * E) / den, (2 * A * E - B * D) / den
    num = 2 * (A * E * E + C * D * D - B * D * E + den * F)
    root = np.sqrt((A - C) ** 2 + B * B)
    with np.errstate(invalid="ignore"):
        a = -np.sqrt(num * (A + C + root)) / den
        b = -np.sqrt(num * (A + C - root)) / den
    th = 0.5 * np.arctan2(-B, C - A)
    if not (np.isfinite([x0, y0, a, b, th]).all() and min(a, b) > 0):
        return None
    if a < b:                                                 # make a the major axis
        a, b, th = b, a, th + np.pi / 2
    return float(x0 * sc + o[0]), float(y0 * sc + o[1]), float(a * sc), float(b * sc), float(th)


def _ellipse_distance(xy, p) -> np.ndarray:
    """Distance of each point (x, y) to the outline of the ellipse p = (x, y, a, b, theta)."""
    from scipy.spatial import cKDTree
    n = int(np.clip(8 * (p[2] + p[3]), 360, 4000))
    px, py = _ellipse_points(*p, n=n)
    return cKDTree(np.c_[px, py]).query(np.asarray(xy, float))[0]


def _ellipse_points(xc, yc, a, b, th, n=360):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    x = xc + a * np.cos(t) * ct - b * np.sin(t) * st
    y = yc + a * np.cos(t) * st + b * np.sin(t) * ct
    return x, y


def outline_fit(mask: np.ndarray, cfg: RCNNConfig, offset=(0, 0), full_hw=None, edge_only=False) -> Optional[dict]:
    """Fit an ellipse/circle to one predicted mask, in NATIVE coordinates.

    Masks of bubbles cut by the image edge have a straight side along the border. Those border
    points are dropped, and only the visible arc is fitted, so the centre may lie OUTSIDE the
    image. A circle is used unless the arc is long enough (edge_min_arc_deg) AND an ellipse fits it
    clearly better (edge_ellipse_gain): on short noisy arcs ellipse fits are unstable.
    Returns dict(x, y, a, b, theta, kind, truncated, arc_deg, visible_frac, circle) or None; `circle` is the
    circle fitted to the (visible) outline as (x, y, r). Slivers at the edge (visible arc < edge_min_fit_arc_deg)
    return kind='visible' (and their arc circle).
    `mask` may be a crop of the full-frame mask (with >= 1 px background margin except at the image
    border); `offset` is its top-left corner and `full_hw` the full model-resolution frame size.
    edge_only=True skips the fit for bubbles that do not touch the image edge (returns truncated=False).
    """
    s = cfg.model_scale
    H, W = full_hw if full_hw is not None else mask.shape
    cs = measure.find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
    if not cs:
        return None
    c = max(cs, key=len) - 1 + np.asarray(offset, float)   # model (row, col); border runs at -0.5 / H-0.5
    on_edge = (c[:, 0] <= 0) | (c[:, 0] >= H - 1) | (c[:, 1] <= 0) | (c[:, 1] >= W - 1)
    truncated = on_edge.sum() >= 3
    if edge_only and not truncated:
        return dict(truncated=False)
    arc = c[~on_edge] if truncated else c
    if len(arc) < 6:
        return None
    xy = model_to_native(arc, s)[:, ::-1]         # native (x, y)
    xc, yc, r = _fit_circle(xy)
    ang = np.arctan2(xy[:, 1] - yc, xy[:, 0] - xc)
    arc_deg = 10.0 * len(np.unique(np.floor((ang + np.pi) / np.deg2rad(10)).astype(int)))
    if truncated and arc_deg < cfg.edge_min_fit_arc_deg:      # sliver: the fitted circle is uncertain
        return dict(truncated=True, kind="visible", arc_deg=float(arc_deg), visible_frac=np.nan, circle=(xc, yc, r))
    fit = dict(x=xc, y=yc, a=r, b=r, theta=0.0, kind="circle", circle=(xc, yc, r))
    if not truncated or arc_deg >= cfg.edge_min_arc_deg:
        ep = _fit_ellipse(xy)
        if ep is not None:
            ex, ey, a, b, th = ep
            ok = True
            if truncated:          # partial arc: sanity checks + ellipse must clearly beat the circle
                rms_c = np.sqrt(np.mean((np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - r) ** 2))
                rms_e = np.sqrt(np.mean(_ellipse_distance(xy, ep) ** 2))
                ok = (max(a, b) / min(a, b) <= cfg.edge_max_axis_ratio and max(a, b) < 3 * r
                      and np.hypot(ex - xc, ey - yc) < r and rms_c > cfg.edge_ellipse_gain * max(rms_e, 1e-6))
            if ok:
                fit = dict(x=ex, y=ey, a=a, b=b, theta=th, kind="ellipse", circle=(xc, yc, r))
    # fraction of the fitted outline inside the image
    Hn, Wn = H / s, W / s
    px, py = _ellipse_points(fit["x"], fit["y"], fit["a"], fit["b"], fit["theta"])
    vis = (px >= -0.5) & (px <= Wn - 0.5) & (py >= -0.5) & (py <= Hn - 0.5)
    fit.update(truncated=bool(truncated), arc_deg=float(min(arc_deg, 360.0)), visible_frac=float(vis.mean()))
    return fit


def _orientation_from_row_axis(theta_major):
    """Angle (deg, [-90, 90)) between the row axis and a major axis at angle theta (rad, from x)."""
    o = np.degrees(np.arctan2(np.cos(theta_major), np.sin(theta_major)))
    return (o + 90) % 180 - 90


def measure_mask(m: np.ndarray, cfg: RCNNConfig) -> Optional[dict]:
    """Measure one instance mask (model resolution) -> row dict in native px (see measure_instances)."""
    s = cfg.model_scale
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    if not len(rows):
        return None
    # work on a crop around the bubble (1 px margin) - much faster than the full frame
    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, m.shape[0])
    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, m.shape[1])
    crop = m[r0:r1, c0:c1]
    rp = measure.regionprops(crop.astype(np.uint8))[0]
    cy, cx = model_to_native(np.asarray(rp.centroid) + (r0, c0), s)
    area_vis = rp.area / s ** 2
    fit = outline_fit(crop, cfg, offset=(r0, c0), full_hw=m.shape, edge_only=True)
    row = dict(x=cx, y=cy, area_px2=area_vis,
               r_eq=np.sqrt(area_vis / np.pi), major_axis=rp.axis_major_length / s,
               minor_axis=rp.axis_minor_length / s, orientation_deg=np.degrees(rp.orientation),
               eccentricity=rp.eccentricity, solidity=rp.solidity, area_visible_px2=area_vis,
               edge_truncated=False, fit_kind="mask", arc_deg=360.0, outline_visible_frac=1.0,
               fit_reliable=True)
    if fit is not None and fit["truncated"] and fit["kind"] == "visible":
        row.update(edge_truncated=True, fit_kind="visible", arc_deg=fit["arc_deg"], outline_visible_frac=np.nan,
                   fit_reliable=False)
    elif fit is not None and fit["truncated"]:
        a, b = max(fit["a"], fit["b"]), min(fit["a"], fit["b"])
        th = fit["theta"] if fit["a"] >= fit["b"] else fit["theta"] + np.pi / 2
        orient = np.nan if fit["kind"] == "circle" else _orientation_from_row_axis(th)
        row.update(x=fit["x"], y=fit["y"], area_px2=np.pi * a * b, r_eq=np.sqrt(a * b),
                   major_axis=2 * a, minor_axis=2 * b, orientation_deg=orient,
                   eccentricity=np.sqrt(1 - (b / a) ** 2), edge_truncated=True, fit_kind=fit["kind"],
                   arc_deg=fit["arc_deg"], outline_visible_frac=fit["visible_frac"],
                   fit_reliable=fit["arc_deg"] >= cfg.edge_reliable_arc_deg)
    return row


def finish_measurements(df: pd.DataFrame, native_hw, cfg: RCNNConfig) -> pd.DataFrame:
    """Add in_analysis (exclusion options) and micrometre columns."""
    if len(df):
        Hn, Wn = native_hw
        keep = np.ones(len(df), bool)
        if cfg.drop_edge_bubbles:
            keep &= ~df["edge_truncated"].to_numpy(bool)
        if cfg.drop_unreliable_fits:
            keep &= df["fit_reliable"].to_numpy(bool)
        if cfg.inner_margin > 0:
            mgn = cfg.inner_margin
            keep &= ((df["x"] >= mgn - 0.5) & (df["x"] <= Wn - 0.5 - mgn)
                     & (df["y"] >= mgn - 0.5) & (df["y"] <= Hn - 0.5 - mgn)).to_numpy()
        df["in_analysis"] = keep
    if len(df) and cfg.um_per_px:
        u = cfg.um_per_px
        df["r_eq_um"] = df["r_eq"] * u
        df["area_um2"] = df["area_px2"] * u ** 2
        df["area_visible_um2"] = df["area_visible_px2"] * u ** 2
        df["major_axis_um"] = df["major_axis"] * u
        df["minor_axis_um"] = df["minor_axis"] * u
    return df


def measure_instances(pred: dict, cfg: RCNNConfig, frame: str = "") -> pd.DataFrame:
    """Per bubble: centre, area, equivalent radius, ellipse axes (native px and um).

    Interior bubbles: measured from the mask (centroid, area, regionprops ellipse).
    Bubbles cut by the image edge (edge_truncated=True): measured from the outline fitted to the
    visible arc, so x, y, area and axes describe the FULL bubble (centre may be outside the image).
    area_visible_px2 is always the visible mask area. fit_reliable=False marks edge fits from a short
    visible arc (< edge_reliable_arc_deg). in_analysis applies the exclusion options
    (drop_edge_bubbles, drop_unreliable_fits, inner_margin).
    """
    s = cfg.model_scale
    rows = []
    for i, (m, sc) in enumerate(zip(pred["masks"], pred["scores"])):
        row = measure_mask(m, cfg)
        if row is not None:
            rows.append(dict(frame=frame, bubble_id=i, score=float(sc), **row))
    df = pd.DataFrame(rows)
    native_hw = (pred["masks"].shape[1] / s, pred["masks"].shape[2] / s) if len(pred["masks"]) else (0, 0)
    return finish_measurements(df, native_hw, cfg)


def predict_frame(model, path: str, cfg: RCNNConfig):
    img = bs.load_image(path)
    x = prepare_image(img, cfg)
    pred = predict(model, x, cfg)
    return img, x, pred, measure_instances(pred, cfg, os.path.basename(path))


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def match_instances(pred_masks, gt_masks, iou_thr=0.5):
    """Greedy one-to-one matching (pred assumed sorted by score). Returns list of (p, g, iou)."""
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return []
    bp, bg = _bboxes(pred_masks), _bboxes(gt_masks)
    ap = pred_masks.reshape(len(pred_masks), -1).sum(1)
    ag = gt_masks.reshape(len(gt_masks), -1).sum(1)
    iou = np.zeros((len(pred_masks), len(gt_masks)))
    for p in range(len(pred_masks)):
        near = np.flatnonzero((bg[:, 0] < bp[p, 1]) & (bg[:, 1] > bp[p, 0]) & (bg[:, 2] < bp[p, 3]) & (bg[:, 3] > bp[p, 2]))
        for g in near:
            iou[p, g] = _pair_iou(pred_masks[p], gt_masks[g], bp[p], bg[g], ap[p], ag[g])
    used, pairs = set(), []
    for p in range(len(pred_masks)):
        row = iou[p].copy()
        if used:
            row[list(used)] = -1
        g = int(np.argmax(row))
        if row[g] >= iou_thr:
            used.add(g)
            pairs.append((p, g, float(iou[p, g])))
    return pairs


def evaluate(model, samples: list, cfg: RCNNConfig, iou_thr=0.5, return_details=False):
    tp = fp = fn = 0
    details = []
    for s in samples:
        pred = predict(model, s.image, cfg)
        pairs = match_instances(pred["masks"], s.masks, iou_thr)
        tp += len(pairs)
        fp += len(pred["masks"]) - len(pairs)
        fn += len(s.masks) - len(pairs)
        for p, g, iou in pairs:
            rp = np.sqrt(pred["masks"][p].sum() / np.pi) / cfg.model_scale
            rg = np.sqrt(s.masks[g].sum() / np.pi) / cfg.model_scale
            details.append(dict(sample=s.name, iou=iou, r_pred=rp, r_gt=rg))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9))
    if return_details:
        out["matches"] = pd.DataFrame(details)
    return out


# ----------------------------------------------------------------------------
# Human-in-the-loop: predictions -> editable circles / ellipses
# ----------------------------------------------------------------------------
SHAPE_KINDS = ("auto", "ellipse", "circle")


def _mask_moment_ellipse(crop, offset, s):
    """Ellipse (native x, y, a, b, theta) with the area, centroid and orientation of a mask (model resolution)."""
    rr, cc = np.nonzero(crop)
    if len(rr) < 3:
        return None
    y = model_to_native(rr + offset[0], s)
    x = model_to_native(cc + offset[1], s)
    cov = np.cov(np.c_[x, y].T, bias=True) + np.eye(2) / (12 * s * s)      # + extent of one pixel
    ev, evec = np.linalg.eigh(cov)
    if not np.isfinite(ev).all() or ev[0] <= 0:
        return None
    a, b = 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0])
    k = np.sqrt(len(rr) / s ** 2 / (np.pi * a * b))                         # keep the mask area
    return float(x.mean()), float(y.mean()), float(a * k), float(b * k), float(np.arctan2(evec[1, 1], evec[0, 1]))


def _shape_iou(p, m, s, rows, cols):
    """IoU of the ellipse p = (x, y, a, b, theta) (native), cut to the frame, with the mask m (model resolution)."""
    H, W = m.shape
    px, py = _ellipse_points(*p, n=int(np.clip(8 * (p[2] + p[3]), 90, 2000)))
    q = native_to_model(np.c_[py, px], s)
    r0 = int(max(min(q[:, 0].min(), rows[0]) - 1, 0)); r1 = int(min(max(q[:, 0].max(), rows[-1]) + 2, H))
    c0 = int(max(min(q[:, 1].min(), cols[0]) - 1, 0)); c1 = int(min(max(q[:, 1].max(), cols[-1]) + 2, W))
    fm = np.zeros((r1 - r0, c1 - c0), bool)
    rr, cc = draw.polygon(q[:, 0] - r0, q[:, 1] - c0, fm.shape)
    fm[rr, cc] = True
    mm = m[r0:r1, c0:c1]
    inter = np.count_nonzero(fm & mm)
    return inter / (np.count_nonzero(fm) + np.count_nonzero(mm) - inter + 1e-9)


def fit_shape(m: np.ndarray, cfg: RCNNConfig, kind: str = "auto") -> Optional[dict]:
    """Circle or ellipse (napari ellipse shape, native coordinates) for one predicted mask (model resolution).

    Bubbles inside the image: the circle and the ellipse fitted to the outline and the moment ellipse of the mask
    (same area, centre and orientation) are compared, and the one that overlaps the mask best is used.
    Bubbles cut by the image edge: the edge-aware fit to the visible arc (outline_fit), so the centre may lie
    outside the image. Slivers (visible arc < edge_min_fit_arc_deg): the circle through the visible arc; its size
    is uncertain, the measurement flags it fit_reliable=False.
    kind: 'auto' (an ellipse, saved as a circle when rounder than cfg.circle_min_axis_ratio), 'ellipse' (never
    rounded) or 'circle' (always a circle).
    Returns dict(type='ellipse', data=..., fit_kind='circle'|'ellipse'|'sliver'), or None for an empty mask or an
    edge sliver whose arc is too straight to give a size.
    """
    if kind not in SHAPE_KINDS:
        raise ValueError(f"kind must be one of {SHAPE_KINDS}, not {kind!r}")
    s = cfg.model_scale
    H, W = m.shape
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    if not len(rows):
        return None
    # crop around the bubble (1 px margin): much faster than the full frame
    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, H)
    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, W)
    crop = m[r0:r1, c0:c1]
    f = outline_fit(crop, cfg, offset=(r0, c0), full_hw=(H, W))
    sliver = False
    if f is not None and f["truncated"]:
        xc, yc, r = f["circle"]
        sliver = f["kind"] == "visible"
        if sliver or kind == "circle":
            if not (np.isfinite([xc, yc, r]).all() and 0 < r <= max(H, W) / s):   # (nearly) straight arc
                return None
            best = (xc, yc, r, r, 0.0)
        else:
            best = (f["x"], f["y"], f["a"], f["b"], f["theta"])
        if not sliver and _shape_iou(best, m, s, rows, cols) < 0.5:
            f = None            # the edge fit does not match the mask (e.g. a streak across the frame): see below
    if f is None or not f["truncated"]:
        sliver = False
        cands = []
        mom = _mask_moment_ellipse(crop, (r0, c0), s)
        if mom is not None:
            cands.append(mom if kind != "circle" else (mom[0], mom[1], np.sqrt(mom[2] * mom[3]),
                                                        np.sqrt(mom[2] * mom[3]), 0.0))
        if f is not None:
            xc, yc, r = f["circle"]
            if np.isfinite([xc, yc, r]).all() and r > 0:
                cands.append((xc, yc, r, r, 0.0))
            if kind != "circle" and f["kind"] == "ellipse" and max(f["a"], f["b"]) < 3 * r:
                cands.append((f["x"], f["y"], f["a"], f["b"], f["theta"]))
        if not cands:
            return None
        best = max(cands, key=lambda p: _shape_iou(p, m, s, rows, cols))
    xc, yc, a, b, th = best
    if kind == "circle" or (kind == "auto" and min(a, b) / max(a, b) >= cfg.circle_min_axis_ratio):
        a = b = float(np.sqrt(a * b))
    return ellipse_shape(xc, yc, a, b, th, fit_kind="sliver" if sliver else ("circle" if a == b else "ellipse"))


def masks_to_shapes(pred: dict, cfg: RCNNConfig, kind: str = "auto") -> list:
    """Predicted masks -> circles / ellipses (napari ellipse shapes, native coords) for correction in napari.
    kind: 'auto' (default; circle where the bubble is round, else ellipse), 'ellipse' or 'circle' (see fit_shape)."""
    out = []
    for m in pred["masks"]:
        sh = fit_shape(m, cfg, kind=kind)
        if sh is not None:
            out.append(sh)
    return out


def measure_shapes(shapes: list, native_hw, cfg: RCNNConfig, frame: str = "") -> pd.DataFrame:
    """Measure circle / ellipse shapes (annotations, reviewed results or masks_to_shapes output): one row per bubble.

    The exact circle / ellipse parameters are used, so a shape that extends beyond the image (a bubble cut by the
    edge) keeps its full centre and size. Its visible part gives area_visible_px2, outline_visible_frac and arc_deg
    (= 360 x visible fraction of the outline); fit_reliable is False when arc_deg < edge_reliable_arc_deg, and
    for model drafts of slivers (fit_kind 'sliver': only a small piece of the bubble is in the image).
    Shapes entirely outside the image are skipped. bubble_id is the position of the shape in the list.
    Older polygons are measured as their ellipse (as_ellipse).
    """
    s = cfg.model_scale
    Hn, Wn = native_hw
    hw = (int(round(Hn * s)), int(round(Wn * s)))
    rows = []
    for i, sh0 in enumerate(shapes):
        sh = as_ellipse(sh0)
        if sh is None:
            continue
        x, y, a, b, th = ellipse_params(sh)
        if not (np.isfinite([x, y, a, b]).all() and b > 0):
            continue
        outline = shape_outline(sh, n_ellipse=360)                 # (row, col)
        m = rasterize(outline, hw, s)
        if not m.any():                                            # entirely outside the image
            continue
        inside = ((outline[:, 1] >= -0.5) & (outline[:, 1] <= Wn - 0.5)
                  & (outline[:, 0] >= -0.5) & (outline[:, 0] <= Hn - 0.5))
        vis = float(inside.mean())
        is_circle = b >= a * 0.999
        sliver = sh0.get("fit_kind") == "sliver"            # model draft of a bubble mostly outside the image
        rows.append(dict(
            frame=frame, bubble_id=i, shape_type="circle" if is_circle else "ellipse",
            x=x, y=y, area_px2=np.pi * a * b, r_eq=np.sqrt(a * b), major_axis=2 * a, minor_axis=2 * b,
            orientation_deg=np.nan if is_circle else _orientation_from_row_axis(th),
            eccentricity=np.sqrt(max(0.0, 1 - (b / a) ** 2)), solidity=1.0,
            area_visible_px2=float(m.sum()) / s ** 2, edge_truncated=bool(vis < 1 or sliver),
            fit_kind="sliver" if sliver else ("from_polygon" if sh.get("fit_kind") == "from_polygon"
                                              else ("circle" if is_circle else "ellipse")),
            arc_deg=360.0 * vis, outline_visible_frac=vis,
            fit_reliable=bool(not sliver and (vis == 1 or 360.0 * vis >= cfg.edge_reliable_arc_deg))))
    return finish_measurements(pd.DataFrame(rows), (Hn, Wn), cfg)


def predict_shapes(model, path: str, cfg: RCNNConfig, kind: str = "auto"):
    """Predict one frame and fit circles / ellipses. Returns (img, pred, shapes, df): the native image, the raw
    prediction (masks, scores), the shapes as saved for review, and their measurements (measure_shapes)."""
    img = bs.load_image(path)
    pred = predict(model, prepare_image(img, cfg), cfg)
    shapes = masks_to_shapes(pred, cfg, kind=kind)
    return img, pred, shapes, measure_shapes(shapes, img.shape, cfg, os.path.basename(path))


def write_prediction_annotation(json_path: str, image_name: str, shapes: list, **extra):
    """Save predictions as a draft annotation to correct in napari. It is used for training only once you ticked
    Reviewed (whole frame) or drew 'fully annotated' rectangles around the corrected parts (list_annotations)."""
    old = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            old = json.load(f)
    keep = {k: v for k, v in old.items() if k not in ("image", "bubbles", "rois", "draft", "edited_at", "reviewed")}
    save_annotation(json_path, image_name, shapes, [], **{**keep, **extra, "draft": True})


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_instances(img, pred_or_masks, cfg: RCNNConfig, ax=None, title=None, roi=None, color=None,
                   df: Optional[pd.DataFrame] = None, show_fit=True, view_pad=0):
    """Overlay instance outlines (model-resolution masks) on the native image.
    show_fit: dashed fitted outline for bubbles cut by the image edge (may extend outside).
    df: result of measure_instances; bubbles with in_analysis=False are drawn grey.
    view_pad: extend the view this many native px beyond the image to see fitted outlines outside it."""
    import matplotlib.pyplot as plt
    masks = pred_or_masks["masks"] if isinstance(pred_or_masks, dict) else pred_or_masks
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img, cmap="gray")
    cmap = plt.get_cmap("tab20")
    s = cfg.model_scale
    excluded = set(df.loc[~df["in_analysis"], "bubble_id"]) if df is not None and "in_analysis" in df else set()
    for i, m in enumerate(masks):
        col = "0.6" if i in excluded else (color or cmap(i % 20))
        for c in measure.find_contours(np.pad(m, 1).astype(np.float32), 0.5):
            c = model_to_native(c - 1, s)
            ax.plot(c[:, 1], c[:, 0], lw=0.9, color=col)
        if show_fit:
            fit = outline_fit(m, cfg)
            if fit is not None and fit["truncated"] and fit["kind"] != "visible":
                px, py = _ellipse_points(fit["x"], fit["y"], fit["a"], fit["b"], fit["theta"])
                ax.plot(np.r_[px, px[:1]], np.r_[py, py[:1]], ls="--", lw=0.8, color=col)
    if cfg.inner_margin > 0:
        mg = cfg.inner_margin
        ax.add_patch(plt.Rectangle((mg - 0.5, mg - 0.5), img.shape[1] - 2 * mg, img.shape[0] - 2 * mg,
                                   fill=False, color="yellow", lw=1, ls=":"))
    if roi is not None:
        ax.set_xlim(roi[0], roi[1])
        ax.set_ylim(roi[3], roi[2])
    else:
        ax.set_xlim(-0.5 - view_pad, img.shape[1] - 0.5 + view_pad)
        ax.set_ylim(img.shape[0] - 0.5 + view_pad, -0.5 - view_pad)
        if view_pad:
            ax.add_patch(plt.Rectangle((-0.5, -0.5), img.shape[1], img.shape[0], fill=False, color="w", lw=0.8))
    ax.set_title(title or f"{len(masks)} bubbles" + ("  (dashed: fitted outline of edge bubbles)" if show_fit else ""))
    ax.axis("off")
    return ax


def plot_annotation(ann: dict, ax=None, roi=None):
    import matplotlib.pyplot as plt
    img = bs.load_image(annotation_image(ann))
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img, cmap="gray")
    cmap = plt.get_cmap("tab20")
    for i, b in enumerate(ann["bubbles"]):
        o = shape_outline(b)
        o = np.vstack([o, o[:1]])
        ax.plot(o[:, 1], o[:, 0], lw=0.9, color=cmap(i % 20))
    for r0, c0, r1, c1 in ann["rois"]:
        ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, color="yellow", lw=1.5, ls="--"))
    if roi is not None:
        ax.set_xlim(roi[0], roi[1])
        ax.set_ylim(roi[3], roi[2])
    ax.set_title(f"{os.path.basename(ann['json_path'])}: {len(ann['bubbles'])} bubbles, {len(ann['rois'])} ROIs (yellow)")
    ax.axis("off")
    return ax
