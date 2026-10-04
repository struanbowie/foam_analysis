"""
Trainable bubble instance segmentation that handles overlapping bubbles.

Model: torchvision Mask R-CNN (ResNet50-FPN v2, COCO-pretrained). It predicts one mask per
object, so masks may overlap (nested / crossing bubbles). Masks are free-form (non-spherical
bubbles). The FPN with custom anchor sizes, plus upsampling and scale-jitter augmentation,
handles the large size range.

Annotations are napari *Shapes* (one ellipse / polygon per bubble, overlaps allowed), stored as
JSON next to the image they belong to:

    {
      "image": "frame.tif",                 # relative to the JSON file
      "bubbles": [{"type": "ellipse", "data": [[row, col] x 4]},
                  {"type": "polygon", "data": [[row, col], ...]}],
      "rois": [[r0, c0, r1, c1], ...]       # fully annotated regions; [] = whole image
    }

All annotation coordinates are in NATIVE image pixels (napari convention: pixel centres at
integer coordinates). The model works at ``model_scale`` x native resolution.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
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
    detections_per_img: int = 400
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

    # --- bubbles cut by the image edge ---
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
        for k in ("channels", "anchor_ratios", "scale_jitter"):
            d[k] = tuple(d[k])
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
def shape_outline(shape: dict, n_ellipse: int = 128) -> np.ndarray:
    """Return the outline polygon (N, 2) in (row, col) for an annotation shape."""
    d = np.asarray(shape["data"], float)
    t = shape["type"]
    if t == "ellipse":
        # napari stores ellipses as the 4 corners of their (possibly rotated) bounding box
        c = d.mean(axis=0)
        e1, e2 = (d[1] - d[0]) / 2.0, (d[3] - d[0]) / 2.0
        th = np.linspace(0, 2 * np.pi, n_ellipse, endpoint=False)
        return c + np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    if t in ("polygon", "rectangle"):
        return d
    raise ValueError(f"unsupported shape type {t!r} (use ellipse or polygon)")


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
    return ann


def save_annotation(json_path: str, image_name: str, bubbles: list, rois: list):
    with open(json_path, "w") as f:
        json.dump(dict(image=image_name, bubbles=bubbles, rois=rois), f)


def list_annotations(cfg: RCNNConfig) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(cfg.annotation_dir, "*.json"))):
        a = load_annotation(p)
        if len(a["bubbles"]):
            out.append(a)
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
            save_annotation(js, name, [], [])
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


def build_samples(annotations: list, cfg: RCNNConfig) -> list:
    """Prepare every annotated ROI as a Sample (whole image if no ROI was drawn)."""
    s = cfg.model_scale
    samples = []
    for ann in annotations:
        img = bs.load_image(ann["image_path"])
        x = prepare_image(img, cfg)
        H, W = x.shape[1:]
        outlines = [shape_outline(b) for b in ann["bubbles"]]
        rois = ann["rois"] or [[0, 0, img.shape[0] - 1, img.shape[1] - 1]]
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


def split_samples(samples: list, cfg: RCNNConfig):
    """Hold out whole ROIs for validation. Needs >= 3 ROIs; with fewer, everything is used for
    training (no validation) so a tiny dataset is not halved."""
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(samples))
    if len(samples) < 3 or cfg.val_fraction <= 0:
        print(f"{len(samples)} annotated region(s): no validation split (draw >= 3 'fully annotated' rectangles to get one)")
        return list(samples), []
    n_val = max(1, int(round(cfg.val_fraction * len(samples))))
    val = [samples[i] for i in idx[:n_val]]
    train = [samples[i] for i in idx[n_val:]]
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
        rng = np.random.default_rng((self.cfg.seed, idx, os.getpid()))
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
    return torch.device(cfg.device)


def save_model(model, cfg: RCNNConfig, name="bubble_rcnn"):
    import torch
    os.makedirs(cfg.model_dir, exist_ok=True)
    path = os.path.join(cfg.model_dir, name + ".pt")
    torch.save(model.state_dict(), path)
    cfg.to_json(os.path.join(cfg.model_dir, name + "_config.json"))
    return path


def load_model(path, cfg: Optional[RCNNConfig] = None, device=None):
    import torch
    if cfg is None:
        cfg = RCNNConfig.from_json(path.replace(".pt", "_config.json"))
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
def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on masks; returns kept indices (sorted by score)."""
    order = np.argsort(-scores)
    flat = masks.reshape(len(masks), -1)
    areas = flat.sum(1)
    keep = []
    for i in order:
        ok = True
        for j in keep:
            inter = np.logical_and(flat[i], flat[j]).sum()
            if inter and inter / (areas[i] + areas[j] - inter) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return np.array(keep, int)


def predict(model, x: np.ndarray, cfg: RCNNConfig, score_thresh=None):
    """x: (3,H,W) model input. Returns dict(masks (n,H,W) bool, scores (n,))."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    st = cfg.score_thresh if score_thresh is None else score_thresh
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


def _ellipse_params(em):
    if hasattr(em, "axis_lengths"):
        (xc, yc), (a, b), th = em.center, em.axis_lengths, em.theta
    else:
        xc, yc, a, b, th = em.params
    return float(xc), float(yc), float(a), float(b), float(th)


def _ellipse_points(xc, yc, a, b, th, n=360):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    x = xc + a * np.cos(t) * ct - b * np.sin(t) * st
    y = yc + a * np.cos(t) * st + b * np.sin(t) * ct
    return x, y


def outline_fit(mask: np.ndarray, cfg: RCNNConfig) -> Optional[dict]:
    """Fit an ellipse/circle to one predicted mask, in NATIVE coordinates.

    Masks of bubbles cut by the image edge have a straight side along the border. Those border
    points are dropped, and only the visible arc is fitted, so the centre may lie OUTSIDE the
    image. A circle is used unless the arc is long enough (edge_min_arc_deg) AND an ellipse fits it
    clearly better (edge_ellipse_gain): on short noisy arcs ellipse fits are unstable.
    Returns dict(x, y, a, b, theta, kind, truncated, arc_deg, visible_frac) or None.
    """
    s = cfg.model_scale
    H, W = mask.shape
    cs = measure.find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
    if not cs:
        return None
    c = max(cs, key=len) - 1                      # model (row, col); border runs at -0.5 / H-0.5
    on_edge = (c[:, 0] <= 0) | (c[:, 0] >= H - 1) | (c[:, 1] <= 0) | (c[:, 1] >= W - 1)
    truncated = on_edge.sum() >= 3
    arc = c[~on_edge] if truncated else c
    if len(arc) < 6:
        return None
    xy = model_to_native(arc, s)[:, ::-1]         # native (x, y)
    xc, yc, r = _fit_circle(xy)
    ang = np.arctan2(xy[:, 1] - yc, xy[:, 0] - xc)
    arc_deg = 10.0 * len(np.unique(np.floor((ang + np.pi) / np.deg2rad(10)).astype(int)))
    fit = dict(x=xc, y=yc, a=r, b=r, theta=0.0, kind="circle")
    if not truncated or arc_deg >= cfg.edge_min_arc_deg:
        em = _fit_ellipse(xy)
        if em is not None:
            ex, ey, a, b, th = _ellipse_params(em)
            ok = np.isfinite([ex, ey, a, b]).all() and min(a, b) > 0
            if ok and truncated:   # partial arc: sanity checks + ellipse must clearly beat the circle
                rms_c = np.sqrt(np.mean((np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - r) ** 2))
                rms_e = np.sqrt(np.mean(np.asarray(em.residuals(xy)) ** 2))
                ok = (max(a, b) / min(a, b) <= cfg.edge_max_axis_ratio and max(a, b) < 3 * r
                      and np.hypot(ex - xc, ey - yc) < r and rms_c > cfg.edge_ellipse_gain * max(rms_e, 1e-6))
            if ok:
                fit = dict(x=ex, y=ey, a=a, b=b, theta=th, kind="ellipse")
    # fraction of the fitted outline inside the image
    Hn, Wn = H / s, W / s
    px, py = _ellipse_points(fit["x"], fit["y"], fit["a"], fit["b"], fit["theta"])
    vis = (px >= -0.5) & (px <= Wn - 0.5) & (py >= -0.5) & (py <= Hn - 0.5)
    fit.update(truncated=bool(truncated), arc_deg=float(min(arc_deg, 360.0)), visible_frac=float(vis.mean()))
    return fit


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
        rp = measure.regionprops(m.astype(np.uint8))[0]
        cy, cx = model_to_native(rp.centroid, s)
        area_vis = rp.area / s ** 2
        fit = outline_fit(m, cfg)
        row = dict(frame=frame, bubble_id=i, score=float(sc), x=cx, y=cy, area_px2=area_vis,
                   r_eq=np.sqrt(area_vis / np.pi), major_axis=rp.axis_major_length / s,
                   minor_axis=rp.axis_minor_length / s, orientation_deg=np.degrees(rp.orientation),
                   eccentricity=rp.eccentricity, solidity=rp.solidity, area_visible_px2=area_vis,
                   edge_truncated=False, fit_kind="mask", arc_deg=360.0, outline_visible_frac=1.0,
                   fit_reliable=True)
        if fit is not None and fit["truncated"]:
            a, b = max(fit["a"], fit["b"]), min(fit["a"], fit["b"])
            th = fit["theta"] if fit["a"] >= fit["b"] else fit["theta"] + np.pi / 2
            orient = np.degrees(np.arctan2(np.cos(th), np.sin(th)))          # angle from the row axis
            orient = (orient + 90) % 180 - 90
            if fit["kind"] == "circle":
                orient = np.nan
            row.update(x=fit["x"], y=fit["y"], area_px2=np.pi * a * b, r_eq=np.sqrt(a * b),
                       major_axis=2 * a, minor_axis=2 * b, orientation_deg=orient,
                       eccentricity=np.sqrt(1 - (b / a) ** 2), edge_truncated=True, fit_kind=fit["kind"],
                       arc_deg=fit["arc_deg"], outline_visible_frac=fit["visible_frac"],
                       fit_reliable=fit["arc_deg"] >= cfg.edge_reliable_arc_deg)
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        Hn, Wn = pred["masks"].shape[1] / s, pred["masks"].shape[2] / s
        keep = np.ones(len(df), bool)
        if cfg.drop_edge_bubbles:
            keep &= ~df["edge_truncated"].to_numpy()
        if cfg.drop_unreliable_fits:
            keep &= df["fit_reliable"].to_numpy()
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
    P = pred_masks.reshape(len(pred_masks), -1).astype(np.float32)
    G = gt_masks.reshape(len(gt_masks), -1).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    used, pairs = set(), []
    for p in range(len(P)):
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
# Human-in-the-loop: predictions -> editable annotation shapes
# ----------------------------------------------------------------------------
def _fit_ellipse(xy):
    """skimage EllipseModel fit, compatible with old (estimate) and new (from_estimate) APIs."""
    if hasattr(measure.EllipseModel, "from_estimate"):
        em = measure.EllipseModel.from_estimate(xy)
        return em if em else None
    em = measure.EllipseModel()
    return em if em.estimate(xy) else None


def masks_to_shapes(pred: dict, cfg: RCNNConfig, kind="polygon", tol=0.4, min_score=None) -> list:
    """Convert predicted masks to annotation shapes (native coords) for correction in napari.
    kind='polygon' keeps the predicted shape; 'ellipse' gives easier-to-edit ellipses."""
    s = cfg.model_scale
    out = []
    for m, sc in zip(pred["masks"], pred["scores"]):
        if min_score is not None and sc < min_score:
            continue
        cs = measure.find_contours(np.pad(m, 1).astype(np.float32), 0.5)
        if not cs:
            continue
        c = max(cs, key=len) - 1
        c = model_to_native(c, s)
        if kind == "ellipse":
            fit = outline_fit(m, cfg)    # edge-aware: centre may lie outside the image
            if fit is not None:
                xc, yc, a, b, th = fit["x"], fit["y"], fit["a"], fit["b"], fit["theta"]
                ca, sa = np.cos(th), np.sin(th)
                ux, uy = np.array([ca, sa]) * a, np.array([-sa, ca]) * b
                corners_xy = [(xc - ux[0] - uy[0], yc - ux[1] - uy[1]), (xc + ux[0] - uy[0], yc + ux[1] - uy[1]),
                              (xc + ux[0] + uy[0], yc + ux[1] + uy[1]), (xc - ux[0] + uy[0], yc - ux[1] + uy[1])]
                out.append(dict(type="ellipse", data=[[y, x] for x, y in corners_xy]))
                continue
        c = measure.approximate_polygon(c, tol)
        if len(c) >= 4:
            out.append(dict(type="polygon", data=c[:-1].tolist()))
    return out


def write_prediction_annotation(json_path: str, image_name: str, shapes: list):
    """Save predictions as a draft annotation (no ROI: you add ROIs once a region is fully checked)."""
    save_annotation(json_path, image_name, shapes, [])


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
            if fit is not None and fit["truncated"]:
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
    img = bs.load_image(ann["image_path"])
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
