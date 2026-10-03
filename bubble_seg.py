"""
Bubble segmentation pipeline for high-speed foam / bubble images.

Stages
------
1. Load + mask burned-in overlays (timestamp, scale bar) and user exclusions.
2. Preprocess: masked background flattening, denoising, contrast stretch,
   optional CLAHE, and a noise-normalised dark-ridge ("rim") map.
3. Cellpose-SAM segmentation, optionally at several diameters
   (multi-scale passes; their masks may overlap each other).
4. Robust circle fit for every mask, refined against the rim map, so that
   partially occluded bubbles still get a full circle (radius, centre).
5. Optional ridge-Hough "completion" to recover circles Cellpose missed
   (typically large bubbles fragmented by the bubbles in front of them).
6. Quality control + duplicate suppression -> one table of bubbles.

All length parameters in ``Config`` are given in ORIGINAL image pixels; they
are converted internally to the working resolution (``work_scale``; the defaults
are for the clean 400x250 native exports, processed upsampled 2.5x).
All results are reported in original pixels (and micrometres if the pixel
size is known).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, draw, exposure, feature, filters, io, measure, morphology, restoration, segmentation, transform


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # --- I/O -----------------------------------------------------------------
    image_path: str = "animations/jpg_r563_svd_normalised_tr050/r563_svd_normalised_tr050_tid0_000.jpg"
    output_dir: str = "outputs"
    # glob for the whole series (used for the temporal static mask / batch runs)
    frame_glob: str = "animations/jpg_r563_svd_normalised_tr050/*.jpg"

    # --- pixel size ----------------------------------------------------------
    um_per_px: Optional[float] = 3.2           # HPV-X2 effective pixel size; None -> measure a burned-in scale bar
    scale_bar_um: float = 100.0                # physical length of the burned-in bar
    scale_bar_box: tuple = (0.85, 1.0, 0.80, 1.0)  # (y0, y1, x0, x1) fractions to search

    # --- regions to ignore -----------------------------------------------------
    # boxes as (y0, y1, x0, x1) FRACTIONS of the image size, e.g. burned-in text (none in the clean exports)
    exclude_boxes: list = field(default_factory=list)
    # polygons as lists of (x, y) ORIGINAL-pixel vertices, e.g. a fibre you never want
    exclude_polygons: list = field(default_factory=list)

    # --- static structures (fibres, film wrinkles) ------------------------------
    # 'none' | 'lines' (single frame: long straight ridges) | 'temporal' (series) | 'both'
    static_mode: str = "lines"
    line_min_length: float = 24.0      # px; straight ridge segments longer than this are "static"
    line_max_gap: float = 4.0          # px; gap allowed inside one segment
    line_ridge_z: float = 3.0          # ridge z-score used to find lines
    line_width: float = 3.2            # px; width painted around each detected line
    temporal_n_frames: int = 40        # frames sampled from frame_glob for the temporal mask
    temporal_percentile: float = 20.0  # ridge must be present in >= (100 - p)% of frames
    temporal_ridge_z: float = 3.0
    static_dilate: float = 0.8         # px dilation of the final static mask

    # --- preprocessing ---------------------------------------------------------
    work_scale: float = 2.5            # resampling factor for processing (>1 upsamples the 400x250 native frames)
    bg_sigma: float = 24.0             # px; Gaussian scale of the illumination background
    bg_mode: str = "divide"            # 'divide' | 'subtract'
    denoise: str = "gaussian"          # 'gaussian' | 'nlm' | 'tv' | 'none'
    denoise_sigma: float = 1.0         # px (gaussian)
    nlm_h: float = 0.8                 # NLM strength, multiples of the estimated noise sigma
    tv_weight: float = 0.05            # TV-Chambolle weight
    stretch_percentiles: tuple = (0.5, 99.5)
    use_clahe: bool = True
    clahe_kernel: float = 51.2         # px
    clahe_clip: float = 0.01
    ridge_sigmas: tuple = (0.8, 1.2, 1.6)  # px; ~half the rim thickness in original px

    # --- Cellpose-SAM ------------------------------------------------------------
    use_cellpose: bool = True
    cp_model: str = "cpsam"            # built-in name ('cpsam', 'cpsam_v2') or path to a fine-tuned model
    cp_gpu: bool = True
    cp_input: str = "clahe"            # 'norm' | 'clahe' | 'ridge' | 'clahe+ridge'
    # one Cellpose pass per entry; diameter in ORIGINAL px, None = model default (no rescale)
    cp_diameters: list = field(default_factory=lambda: [None, 24.0])
    cp_flow_threshold: float = 0.4     # higher -> more (less certain) masks
    cp_cellprob_threshold: float = 0.0  # lower (e.g. -2) -> more / larger masks
    cp_min_size: int = 15              # working px
    cp_max_size_fraction: float = 0.4
    cp_tile_norm_blocksize: float = 0  # px; >0 = local normalisation inside Cellpose
    cp_augment: bool = False           # test-time flips (slower, a bit better)
    cp_batch_size: int = 8
    cp_niter: Optional[int] = None

    # --- circle fitting + quality control -----------------------------------------
    rim_z: float = 2.5                 # ridge z-score for a pixel to count as "rim"
    rim_search: float = 0.4            # px; radial tolerance when looking for the rim
    ransac_tol: float = 0.8            # px; inlier distance for RANSAC circle fit
    ransac_iters: int = 300
    min_radius: float = 1.0            # px
    max_radius: float = 80.0           # px
    min_arc_support_mask: float = 0.35  # mask-based (Cellpose/watershed): fraction of visible circumference on a rim
    min_arc_support_hough: float = 0.70
    min_ring_contrast_mask: float = 1.0     # rim z minus mean z just inside/outside the ring
    min_ring_contrast_hough: float = 2.0
    min_mask_circle_iou: float = 0.05  # mask-based: the mask must actually overlap its fitted circle
    max_r_over_mask_radius: float = 4.0  # mask-based: fitted r / mask equivalent radius (occluded masks are small)
    min_visible_frac: float = 0.4      # fraction of circumference not excluded/static
    dup_frac: float = 0.3              # circles closer than this (x radius) in centre AND radius are duplicates

    # --- classical watershed baseline (optional extra candidate source) ----------------
    use_watershed: bool = False
    ws_rim_z: float = 2.0              # pixels below this rim z-score count as bubble interior
    ws_min_distance: float = 1.6       # px between seeds
    ws_min_inner_radius: float = 1.2   # px; seeds need at least this distance to a rim
    ws_min_area: float = 4.8           # px^2

    # --- Hough completion -------------------------------------------------------------
    use_hough: bool = True
    hough_radii: tuple = (5.0, 44.0)   # px range (small ones are left to Cellpose)
    hough_radius_step: float = 0.4      # px
    hough_edge_z: float = 3.0
    hough_peak_threshold: float = 0.12  # fraction of circumference voting (low: scoring does the filtering)
    hough_peaks_per_radius: int = 200

    # --- misc -----------------------------------------------------------------------
    random_seed: int = 0

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


# ----------------------------------------------------------------------------
# I/O and masks
# ----------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """Load an image as float32 greyscale in [0, 1]."""
    raw = io.imread(path)
    if np.issubdtype(raw.dtype, np.integer):
        raw = raw.astype(np.float32) / np.iinfo(raw.dtype).max
    if raw.ndim == 3:
        raw = color.rgb2gray(raw[..., :3])
    im = raw.astype(np.float32)
    if im.max() > 1.0:
        im /= im.max()
    return im


def _box_slices(shape, box):
    h, w = shape
    y0, y1, x0, x1 = box
    return slice(int(round(y0 * h)), int(round(y1 * h))), slice(int(round(x0 * w)), int(round(x1 * w)))


def measure_scale_bar(img: np.ndarray, cfg: Config) -> Optional[float]:
    """Find the bright horizontal scale bar in ``cfg.scale_bar_box``; return um/px."""
    sy, sx = _box_slices(img.shape, cfg.scale_bar_box)
    sub = img[sy, sx]
    bw = sub > 0.8 * sub.max()
    best = None
    for rp in measure.regionprops(measure.label(bw)):
        h = rp.bbox[2] - rp.bbox[0]
        w = rp.bbox[3] - rp.bbox[1]
        if w > 5 * max(h, 1) and w > 20 and (best is None or w > best):
            best = w
    return None if best is None else cfg.scale_bar_um / best


def build_valid_mask(shape, cfg: Config) -> np.ndarray:
    """True where the image may contain bubbles (excludes overlay boxes and polygons)."""
    valid = np.ones(shape, bool)
    for box in cfg.exclude_boxes:
        sy, sx = _box_slices(shape, box)
        valid[sy, sx] = False
    for poly in cfg.exclude_polygons:
        p = np.asarray(poly, float)
        rr, cc = draw.polygon(p[:, 1], p[:, 0], shape)
        valid[rr, cc] = False
    return valid


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def _masked_gaussian(x, m, sigma):
    num = filters.gaussian(np.where(m, x, 0.0), sigma)
    den = filters.gaussian(m.astype(np.float32), sigma)
    return num / np.maximum(den, 1e-3)


def _robust_z(x, m):
    v = x[m]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-12
    return (x - med) / mad


def preprocess(img: np.ndarray, valid: np.ndarray, cfg: Config) -> dict:
    """Return a dict of intermediate images at working resolution.

    keys: work, valid, flat, denoised, norm, clahe, ridge_z, scale
    """
    s = cfg.work_scale
    work = transform.rescale(img, s, anti_aliasing=s < 1, preserve_range=True).astype(np.float32) if s != 1 else img.copy()
    vmask = transform.resize(valid, work.shape, order=0, anti_aliasing=False).astype(bool)

    bg = _masked_gaussian(work, vmask, cfg.bg_sigma * s)
    if cfg.bg_mode == "divide":
        flat = work / np.maximum(bg, 1e-6)
    else:
        flat = work - bg + np.median(work[vmask])
    flat = np.where(vmask, flat, np.median(flat[vmask])).astype(np.float32)

    if cfg.denoise == "gaussian":
        den = filters.gaussian(flat, cfg.denoise_sigma * s)
    elif cfg.denoise == "nlm":
        noise = 1.4826 * np.median(np.abs(flat - filters.gaussian(flat, 1)))
        den = restoration.denoise_nl_means(flat, h=cfg.nlm_h * noise, sigma=noise,
                                           patch_size=5, patch_distance=7, fast_mode=True)
    elif cfg.denoise == "tv":
        den = restoration.denoise_tv_chambolle(flat, weight=cfg.tv_weight)
    else:
        den = flat
    den = den.astype(np.float32)

    lo, hi = np.percentile(den[vmask], cfg.stretch_percentiles)
    norm = np.clip((den - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

    if cfg.use_clahe:
        k = max(8, int(round(cfg.clahe_kernel * s)))
        clahe = exposure.equalize_adapthist(norm, kernel_size=k, clip_limit=cfg.clahe_clip).astype(np.float32)
    else:
        clahe = norm

    ridge = filters.sato(norm, sigmas=[max(0.5, r * s) for r in cfg.ridge_sigmas], black_ridges=True)
    ridge_z = np.clip(_robust_z(ridge, vmask), 0, None) * vmask

    return dict(work=work, valid=vmask, flat=flat, denoised=den, norm=norm, clahe=clahe,
                ridge_z=ridge_z.astype(np.float32), scale=s)


# ----------------------------------------------------------------------------
# Static structures (fibres, wrinkles)
# ----------------------------------------------------------------------------
def line_static_mask(ridge_z: np.ndarray, cfg: Config, scale: float) -> np.ndarray:
    """Long straight ridge segments (fibres). A circle would need r >~ L^2 / (8 * rim width)
    to produce a straight segment of length L, so real bubbles are not hit."""
    edges = morphology.skeletonize(ridge_z > cfg.line_ridge_z)
    lines = transform.probabilistic_hough_line(
        edges, threshold=10, line_length=max(5, int(cfg.line_min_length * scale)),
        line_gap=max(1, int(cfg.line_max_gap * scale)), rng=cfg.random_seed)
    m = np.zeros(ridge_z.shape, bool)
    for (x0, y0), (x1, y1) in lines:
        rr, cc = draw.line(y0, x0, y1, x1)
        m[rr, cc] = True
    w = max(1, int(round(cfg.line_width * scale / 2)))
    return ndi.binary_dilation(m, morphology.disk(w)) if m.any() else m


def temporal_static_mask(frame_paths: Sequence[str], cfg: Config, valid_full: np.ndarray) -> np.ndarray:
    """Ridges present in most frames of the series (computed at working resolution).
    NB: a bubble that does not move or change size over the sampled frames will also be
    flagged; keep temporal_percentile low or sample frames spread over the whole series."""
    paths = list(frame_paths)
    if len(paths) == 0:
        raise FileNotFoundError("no frames for the temporal static mask")
    idx = np.linspace(0, len(paths) - 1, min(cfg.temporal_n_frames, len(paths))).astype(int)
    stack = []
    for i in idx:
        pre = preprocess(load_image(paths[i]), valid_full, cfg)
        stack.append(pre["ridge_z"] > cfg.temporal_ridge_z)
    frac = np.mean(stack, axis=0)
    return frac >= (1 - cfg.temporal_percentile / 100.0)


def build_static_mask(pre: dict, cfg: Config, valid_full: np.ndarray, temporal: Optional[np.ndarray] = None) -> np.ndarray:
    s = pre["scale"]
    m = np.zeros(pre["ridge_z"].shape, bool)
    if cfg.static_mode in ("lines", "both"):
        m |= line_static_mask(pre["ridge_z"], cfg, s)
    if cfg.static_mode in ("temporal", "both"):
        if temporal is None:
            temporal = temporal_static_mask(sorted(glob.glob(cfg.frame_glob)), cfg, valid_full)
        m |= temporal
    if cfg.static_dilate > 0 and m.any():
        m = ndi.binary_dilation(m, morphology.disk(max(1, int(round(cfg.static_dilate * s)))))
    return m


# ----------------------------------------------------------------------------
# Cellpose
# ----------------------------------------------------------------------------
_MODEL_CACHE = {}


def load_cellpose_model(cfg: Config):
    key = (cfg.cp_model, cfg.cp_gpu)
    if key not in _MODEL_CACHE:
        from cellpose import models
        _MODEL_CACHE[key] = models.CellposeModel(gpu=cfg.cp_gpu, pretrained_model=cfg.cp_model)
    return _MODEL_CACHE[key]


def cellpose_input(pre: dict, cfg: Config) -> np.ndarray:
    rz = np.clip(pre["ridge_z"] / 10.0, 0, 1).astype(np.float32)
    if cfg.cp_input == "norm":
        return pre["norm"]
    if cfg.cp_input == "clahe":
        return pre["clahe"]
    if cfg.cp_input == "ridge":
        return 1.0 - rz          # dark rims on bright background, like the raw image
    if cfg.cp_input == "clahe+ridge":
        return np.stack([pre["clahe"], 1.0 - rz, np.zeros_like(rz)], axis=-1)
    raise ValueError(f"unknown cp_input {cfg.cp_input!r}")


def run_cellpose(model, x: np.ndarray, cfg: Config, scale: float) -> list:
    """Run one Cellpose pass per entry in cfg.cp_diameters.
    Returns a list of dicts {diameter, labels, flows}."""
    normalize = {"tile_norm_blocksize": int(round(cfg.cp_tile_norm_blocksize * scale))}
    out = []
    for d in cfg.cp_diameters:
        dw = None if d is None else d * scale
        masks, flows, _ = model.eval(
            x, channel_axis=2 if x.ndim == 3 else None, diameter=dw,
            flow_threshold=cfg.cp_flow_threshold, cellprob_threshold=cfg.cp_cellprob_threshold,
            min_size=cfg.cp_min_size, max_size_fraction=cfg.cp_max_size_fraction,
            normalize=normalize, augment=cfg.cp_augment, batch_size=cfg.cp_batch_size,
            niter=cfg.cp_niter)
        out.append(dict(diameter=d, labels=masks.astype(np.int32), flows=flows))
    return out


# ----------------------------------------------------------------------------
# Circle geometry
# ----------------------------------------------------------------------------
def fit_circle_lsq(x, y, w=None):
    """Algebraic (Kasa) least-squares circle fit, optional weights. Returns xc, yc, r."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    A = np.c_[x, y, np.ones_like(x)] * w[:, None]
    b = (x ** 2 + y ** 2) * w
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = a / 2, bb / 2
    return xc, yc, float(np.sqrt(max(c + xc ** 2 + yc ** 2, 0)))


def fit_circle_ransac(x, y, tol, iters, rng, r_min=0.0, r_max=np.inf):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return None
    best, best_in = None, None
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        try:
            xc, yc, r = fit_circle_lsq(x[i], y[i])
        except np.linalg.LinAlgError:
            continue
        if not (r_min <= r <= r_max) or not np.isfinite(r):
            continue
        inl = np.abs(np.hypot(x - xc, y - yc) - r) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (xc, yc, r), inl
    if best is None or best_in.sum() < 3:
        return None
    return fit_circle_lsq(x[best_in], y[best_in])


def refine_circle_on_ridge(xc, yc, r, ridge_z, usable, cfg_rim_z, tol, n_iter=3):
    """Weighted LSQ re-fit to rim pixels within an annulus around the current circle."""
    h, w = ridge_z.shape
    for _ in range(n_iter):
        y0, y1 = int(max(0, yc - r - tol - 1)), int(min(h, yc + r + tol + 2))
        x0, x1 = int(max(0, xc - r - tol - 1)), int(min(w, xc + r + tol + 2))
        if y1 <= y0 or x1 <= x0:
            break
        yy, xx = np.mgrid[y0:y1, x0:x1]
        z = ridge_z[y0:y1, x0:x1]
        sel = (np.abs(np.hypot(xx - xc, yy - yc) - r) < tol) & (z > cfg_rim_z) & usable[y0:y1, x0:x1]
        if sel.sum() < max(8, 0.5 * r):
            break
        nxc, nyc, nr = fit_circle_lsq(xx[sel], yy[sel], z[sel])
        if not np.isfinite(nr) or abs(nr - r) > tol or np.hypot(nxc - xc, nyc - yc) > tol:
            break
        xc, yc, r = nxc, nyc, nr
    return xc, yc, r


def _ring_samples(xc, yc, r, img, offsets=(0,), order=1, cval=0.0, m=None):
    m = m or max(48, int(2 * np.pi * r))
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    vals = []
    for d in offsets:
        rr = max(r + d, 0.5)
        vals.append(ndi.map_coordinates(img, [yc + rr * np.sin(th), xc + rr * np.cos(th)],
                                        order=order, mode="constant", cval=cval))
    return np.array(vals)


def circle_scores(xc, yc, r, ridge_z, usable, rim_z, search):
    """arc_support: fraction of the VISIBLE circumference lying on a rim;
    ring_contrast: mean rim z on the ring minus mean z just inside/outside;
    visible_frac: fraction of the circumference not excluded / static / off-image."""
    m = max(48, int(2 * np.pi * r))
    offs = np.arange(-int(np.ceil(search)), int(np.ceil(search)) + 1)
    on = _ring_samples(xc, yc, r, ridge_z, offs, m=m).max(axis=0)
    vis = _ring_samples(xc, yc, r, usable.astype(np.float32), (0,), order=0, m=m).ravel() > 0.5
    if vis.sum() == 0:
        return 0.0, 0.0, 0.0
    gap = max(2 * search, 0.25 * r)
    inner = _ring_samples(xc, yc, max(r - gap, 0.5), ridge_z, m=m).ravel() if r > gap + 1 else np.zeros_like(on)
    outer = _ring_samples(xc, yc, r + gap, ridge_z, m=m).ravel()
    support = float((on[vis] > rim_z).mean())
    contrast = float(on[vis].mean() - 0.5 * (inner[vis].mean() + outer[vis].mean()))
    return support, contrast, float(vis.mean())


def disk_iou(region_mask, xc, yc, r, offset):
    """IoU between a region mask (in its bbox) and the fitted disk."""
    oy, ox = offset
    h, w = region_mask.shape
    yy, xx = np.mgrid[oy:oy + h, ox:ox + w]
    disk = np.hypot(xx - xc, yy - yc) <= r
    # the disk may extend outside the bbox: count its full area for the union
    inter = np.logical_and(region_mask, disk).sum()
    union = region_mask.sum() + np.pi * r * r - inter
    return float(inter / max(union, 1))


# ----------------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------------
def masks_to_candidates(labels, pre, static, cfg: Config, source="cellpose", pass_diameter=None, rng=None):
    """One circle candidate per Cellpose mask."""
    s = pre["scale"]
    rz = pre["ridge_z"]
    usable = pre["valid"] & ~static
    rz_max = ndi.maximum_filter(rz, size=3)
    rng = rng or np.random.default_rng(cfg.random_seed)
    tol = cfg.ransac_tol * s
    search = max(1.0, cfg.rim_search * s)
    rmin, rmax = cfg.min_radius * s, cfg.max_radius * s
    out = []
    for rp in measure.regionprops(labels):
        oy, ox = rp.bbox[0], rp.bbox[1]
        reg = rp.image
        bnd = segmentation.find_boundaries(np.pad(reg, 1), mode="inner")[1:-1, 1:-1]
        by, bx = np.nonzero(bnd)
        by = by + oy
        bx = bx + ox
        if len(bx) < 5:
            continue
        ok = (rz_max[by, bx] > cfg.rim_z) & usable[by, bx]
        # prefer boundary points that sit on a real rim and are not on static structures
        if ok.sum() >= max(6, 0.25 * len(bx)):
            px, py = bx[ok], by[ok]
        else:
            px, py = bx, by
        fit = fit_circle_ransac(px, py, tol, cfg.ransac_iters, rng,
                                r_min=max(rmin, 0.3 * rp.equivalent_diameter_area / 2), r_max=2.0 * rmax)
        if fit is None:
            continue
        xc, yc, r = refine_circle_on_ridge(*fit, rz, usable, cfg.rim_z, search)
        sup, con, vis = circle_scores(xc, yc, r, rz, usable, cfg.rim_z, search)
        out.append(dict(
            source=source, pass_diameter=pass_diameter, label=rp.label,
            x=xc / s, y=yc / s, r=r / s,
            mask_area=rp.area / s ** 2, mask_eq_radius=rp.equivalent_diameter_area / 2 / s,
            mask_cx=rp.centroid[1] / s, mask_cy=rp.centroid[0] / s,
            solidity=rp.solidity, eccentricity=rp.eccentricity,
            mask_circle_iou=disk_iou(reg, xc, yc, r, (oy, ox)),
            arc_support=sup, ring_contrast=con, visible_frac=vis))
    return out


def watershed_labels(pre, static, cfg: Config) -> np.ndarray:
    """Classical baseline: watershed of the distance-to-rim map (no deep learning)."""
    s = pre["scale"]
    inside = (pre["ridge_z"] < cfg.ws_rim_z) & pre["valid"] & ~static
    dist = ndi.distance_transform_edt(inside)
    pk = feature.peak_local_max(dist, min_distance=max(2, int(round(cfg.ws_min_distance * s))),
                                threshold_abs=cfg.ws_min_inner_radius * s, labels=measure.label(inside))
    markers = np.zeros(dist.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(-dist, markers, mask=inside)
    areas = np.bincount(lab.ravel())
    lab[areas[lab] < cfg.ws_min_area * s ** 2] = 0
    return segmentation.relabel_sequential(lab)[0]


def hough_candidates(pre, static, cfg: Config):
    """Circle Hough on the skeletonised rim map, scored at working resolution."""
    s = pre["scale"]
    rz = pre["ridge_z"]
    usable = pre["valid"] & ~static
    edges = morphology.skeletonize((rz > cfg.hough_edge_z) & usable)
    r0, r1 = cfg.hough_radii
    radii = np.unique(np.arange(np.ceil(r0 * s), r1 * s + 1e-9, max(1.0, round(cfg.hough_radius_step * s))).astype(int))
    radii = radii[radii >= 3]
    search = max(1.0, cfg.rim_search * s)
    cands = []
    md = max(2, int(round(0.2 * radii.min())))
    for chunk in np.array_split(radii, max(1, len(radii) // 16)):
        H = transform.hough_circle(edges, chunk, normalize=True)
        _, cx, cy, rr = transform.hough_circle_peaks(
            H, chunk, min_xdistance=md, min_ydistance=md, threshold=cfg.hough_peak_threshold,
            num_peaks=cfg.hough_peaks_per_radius, total_num_peaks=cfg.hough_peaks_per_radius * len(chunk))
        del H
        for xc, yc, r in zip(cx, cy, rr):
            xc, yc, r = refine_circle_on_ridge(float(xc), float(yc), float(r), rz, usable, cfg.rim_z, search)
            sup, con, vis = circle_scores(xc, yc, r, rz, usable, cfg.rim_z, search)
            if sup < cfg.min_arc_support_hough or con < cfg.min_ring_contrast_hough:
                continue
            cands.append(dict(source="hough", pass_diameter=np.nan, label=0,
                              x=xc / s, y=yc / s, r=r / s, arc_support=sup, ring_contrast=con, visible_frac=vis))
    return cands


def select_bubbles(cands: list, cfg: Config) -> pd.DataFrame:
    """QC filter + duplicate suppression. Ties are won by Cellpose, then watershed, then Hough."""
    df = pd.DataFrame(cands)
    if df.empty:
        return df
    is_h = df["source"] == "hough"
    min_sup = np.where(is_h, cfg.min_arc_support_hough, cfg.min_arc_support_mask)
    min_con = np.where(is_h, cfg.min_ring_contrast_hough, cfg.min_ring_contrast_mask)
    df["qc_pass"] = ((df["r"] >= cfg.min_radius) & (df["r"] <= cfg.max_radius)
                     & (df["arc_support"] >= min_sup)
                     & (df["ring_contrast"] >= min_con)
                     & (df["visible_frac"] >= cfg.min_visible_frac))
    if "mask_circle_iou" in df:
        mask_ok = ((df["mask_circle_iou"] >= cfg.min_mask_circle_iou)
                   & (df["r"] <= cfg.max_r_over_mask_radius * df["mask_eq_radius"]))
        df["qc_pass"] &= is_h | mask_ok
    good = df[df["qc_pass"]].copy()
    bonus = good["source"].map({"cellpose": 0.2, "watershed": 0.1}).fillna(0.0)
    good["quality"] = good["arc_support"] * np.clip(good["ring_contrast"] / 3.0, 0, 1) + bonus
    good = good.sort_values("quality", ascending=False)
    kept = []
    for i, row in good.iterrows():
        dup = False
        for j in kept:
            k = good.loc[j]
            rr = max(row.r, k.r)
            if np.hypot(row.x - k.x, row.y - k.y) < cfg.dup_frac * rr and abs(row.r - k.r) < cfg.dup_frac * rr:
                dup = True
                break
        if not dup:
            kept.append(i)
    df["selected"] = False
    df.loc[kept, "selected"] = True
    return df


def add_physical_units(df: pd.DataFrame, um_per_px: Optional[float]) -> pd.DataFrame:
    df = df.copy()
    df["area_circle_px2"] = np.pi * df["r"] ** 2
    if um_per_px:
        df["r_um"] = df["r"] * um_per_px
        df["area_circle_um2"] = df["area_circle_px2"] * um_per_px ** 2
        if "mask_area" in df:
            df["mask_area_um2"] = df["mask_area"] * um_per_px ** 2
    return df


# ----------------------------------------------------------------------------
# Whole-frame driver
# ----------------------------------------------------------------------------
def process_frame(path: str, cfg: Config, model=None, temporal_static=None, um_per_px=None, keep_debug=True):
    """Run the full pipeline on one frame. Returns (bubbles_df, all_candidates_df, debug)."""
    img = load_image(path)
    valid = build_valid_mask(img.shape, cfg)
    if um_per_px is None:
        um_per_px = cfg.um_per_px or measure_scale_bar(img, cfg)
    pre = preprocess(img, valid, cfg)
    static = build_static_mask(pre, cfg, valid, temporal_static)
    rng = np.random.default_rng(cfg.random_seed)
    cands, passes = [], []
    if cfg.use_cellpose:
        model = model or load_cellpose_model(cfg)
        passes = run_cellpose(model, cellpose_input(pre, cfg), cfg, pre["scale"])
        for p in passes:
            cands += masks_to_candidates(p["labels"], pre, static, cfg, pass_diameter=p["diameter"], rng=rng)
    if cfg.use_watershed:
        ws = watershed_labels(pre, static, cfg)
        passes.append(dict(diameter="watershed", labels=ws, flows=None))
        cands += masks_to_candidates(ws, pre, static, cfg, source="watershed", rng=rng)
    if cfg.use_hough:
        cands += hough_candidates(pre, static, cfg)
    allc = select_bubbles(cands, cfg)
    if not allc.empty:
        allc = add_physical_units(allc, um_per_px)
        allc.insert(0, "frame", os.path.basename(path))
    bubbles = allc[allc["selected"]].reset_index(drop=True) if not allc.empty else allc
    if not bubbles.empty:
        bubbles.insert(1, "bubble_id", np.arange(len(bubbles)))
    debug = dict(img=img, valid=valid, pre=pre, static=static, passes=passes, um_per_px=um_per_px) if keep_debug else {}
    return bubbles, allc, debug


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
def plot_preprocessing(pre, static=None, figsize=(18, 10)):
    import matplotlib.pyplot as plt
    panels = [("working image", pre["work"], "gray", None), ("flattened + denoised", pre["norm"], "gray", None),
              ("CLAHE", pre["clahe"], "gray", None), ("rim map (ridge z-score)", pre["ridge_z"], "magma", (0, 10))]
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for ax, (t, im, cm, lim) in zip(axs.ravel(), panels):
        ax.imshow(im, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        ax.set_title(t)
        ax.axis("off")
    if static is not None and static.any():
        axs[1, 1].contour(static, levels=[0.5], colors="cyan", linewidths=0.6)
        axs[1, 1].set_title("rim map (cyan: static / excluded structures)")
    fig.tight_layout()
    return fig


def plot_masks(pre, passes, figsize=(18, 6)):
    import matplotlib.pyplot as plt
    n = max(1, len(passes))
    fig, axs = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for ax, p in zip(axs.ravel(), passes):
        ax.imshow(pre["clahe"], cmap="gray")
        b = segmentation.find_boundaries(p["labels"], mode="inner")
        ov = np.zeros(b.shape + (4,))
        ov[b] = (1, 0.2, 0.2, 1)
        ax.imshow(ov)
        name = "watershed baseline" if p["diameter"] == "watershed" else f"Cellpose pass, diameter={p['diameter']}"
        ax.set_title(f"{name}  ({p['labels'].max()} masks)")
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_bubbles(img, bubbles, rejected=None, roi=None, figsize=(18, 11), lw=0.9, title=None):
    """Overlay fitted circles on the original image. roi = (x0, x1, y0, y1) in original px."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors = {"cellpose": "lime", "watershed": "deepskyblue", "hough": "orange"}
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap="gray")
    if rejected is not None:
        for _, b in rejected.iterrows():
            ax.add_patch(Circle((b.x, b.y), b.r, fill=False, color="red", lw=0.4, ls="--", alpha=0.6))
    for _, b in bubbles.iterrows():
        ax.add_patch(Circle((b.x, b.y), b.r, fill=False, color=colors.get(b.source, "cyan"), lw=lw))
    if roi is not None:
        ax.set_xlim(roi[0], roi[1])
        ax.set_ylim(roi[3], roi[2])
    ax.set_title(title or f"{len(bubbles)} bubbles  (green: Cellpose, blue: watershed, orange: Hough"
                 + (", red dashed: rejected)" if rejected is not None else ")"))
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Fine-tuning helpers
# ----------------------------------------------------------------------------
def export_tiles_for_annotation(image, labels, out_dir, tile=256, n_tiles=8, valid=None, seed=0, min_masks=3):
    """Save random image tiles (+ the current masks as a starting point) for correction in
    the Cellpose GUI / napari. Files: tile_XXX.tif and tile_XXX_masks.tif."""
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    saved = []
    tries = 0
    while len(saved) < n_tiles and tries < 50 * n_tiles:
        tries += 1
        y = int(rng.integers(0, max(1, h - tile)))
        x = int(rng.integers(0, max(1, w - tile)))
        if valid is not None and valid[y:y + tile, x:x + tile].mean() < 0.95:
            continue
        lab = labels[y:y + tile, x:x + tile]
        if len(np.unique(lab)) - 1 < min_masks:      # skip near-empty tiles
            continue
        base = os.path.join(out_dir, f"tile_{len(saved):03d}")
        tifffile.imwrite(base + ".tif", (np.clip(image[y:y + tile, x:x + tile], 0, 1) * 65535).astype(np.uint16))
        tifffile.imwrite(base + "_masks.tif", segmentation.relabel_sequential(lab)[0].astype(np.uint16))
        saved.append(base)
    return saved
