"""
Image preprocessing shared by the bubble pipeline.

* load_image        : any frame (jpg / png / tif) -> float32 greyscale in [0, 1]
* build_valid_mask  : optional regions to ignore (boxes / polygons)
* preprocess        : masked background flattening, denoising, contrast stretch, CLAHE and a
                      noise-normalised dark-ridge ("rim") map

Used by bubble_rcnn.prepare_image (the model's input channels: norm, clahe, ridge) and by
annotate_bubbles.py (the "rim map" helper layer). Length parameters in Config are in native
image pixels and are scaled internally by work_scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage import color, draw, exposure, filters, io, restoration, transform


@dataclass
class Config:
    # --- regions to ignore -----------------------------------------------------
    # boxes as (y0, y1, x0, x1) FRACTIONS of the image size, e.g. burned-in text (none in the clean exports)
    exclude_boxes: list = field(default_factory=list)
    # polygons as lists of (x, y) native-pixel vertices
    exclude_polygons: list = field(default_factory=list)

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
    ridge_sigmas: tuple = (0.8, 1.2, 1.6)  # px; ~half the rim thickness


# ----------------------------------------------------------------------------
# Loading and masks
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
