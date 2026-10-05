"""
Statistics and plots for exported bubble results (bubble_analysis.ipynb).

Input: the `bubbles` / `frames` tables from bubble_io.load_results, with a time column `t`
(train number, or real time if TRAIN_TIMES is given in the notebook).

Metrics per frame (see frame_metrics):
    n_bubbles, density_per_mm2          number of bubbles, per mm^2 of field of view
    r_mean_um (+ r_sem_um), r_median_um, r_std_um, r_max_um
    r32_um                              Sauter mean radius sum(r^3)/sum(r^2), weights large bubbles (volume/surface)
    polydispersity                      r_std / r_mean
    area_total_um2                      sum of bubble areas (overlapping bubbles counted fully)
    coverage_frac                       fraction of the frame covered by at least one bubble (union of outlines)
    volume_um3                          sum of 4/3 pi r^3: sphere-equivalent gas volume proxy
    aspect_median, frac_noncircular     minor/major axis; share of bubbles with aspect < 0.8
    cx_um, cy_um                        area-weighted centroid of the bubble population
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import bubble_rcnn as br

# reference palette (dataviz skill): categorical slots 1-4 and text / grid inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2,
    })


def train_colors(n):
    """Sequential blues, light (early) -> dark (late), for ordered trains."""
    import matplotlib.pyplot as plt
    return [plt.get_cmap("Blues")(v) for v in np.linspace(0.35, 1.0, max(n, 1))]


# ----------------------------------------------------------------------------
# Shapes and metrics
# ----------------------------------------------------------------------------
def load_shapes(run_dir, frame):
    with open(os.path.join(run_dir, "shapes", frame + ".json")) as f:
        return json.load(f).get("bubbles", [])


def coverage_fraction(shapes, native_hw, scale=2.0):
    """Fraction of the frame covered by the union of the outlines (rasterised at `scale` x)."""
    hw = (int(round(native_hw[0] * scale)), int(round(native_hw[1] * scale)))
    cov = np.zeros(hw, bool)
    for s in shapes:
        cov |= br.rasterize(br.shape_outline(s), hw, scale)
    return float(cov.mean())


def frame_metrics(bubbles, frames, run_dir=None, um_per_px=3.2):
    """One row per frame with the metrics listed in the module docstring."""
    rows = []
    for _, fr in frames.sort_values(["t", "frame"]).iterrows():
        b = bubbles[bubbles["frame"] == fr["frame"]]
        r = b["r_eq_um"].to_numpy()
        a = b["area_um2"].to_numpy()
        hw = (fr["image_height_px"], fr["image_width_px"])
        fov_mm2 = hw[0] * hw[1] * (um_per_px * 1e-3) ** 2
        aspect = (b["minor_axis"] / b["major_axis"].clip(lower=1e-9)).to_numpy()
        row = dict(train=fr["train"], frame_idx=fr.get("frame_idx"), t=fr["t"], frame=fr["frame"], n_bubbles=len(b),
                   density_per_mm2=len(b) / fov_mm2,
                   r_mean_um=r.mean() if len(r) else np.nan,
                   r_sem_um=r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan,
                   r_median_um=np.median(r) if len(r) else np.nan,
                   r_std_um=r.std(ddof=1) if len(r) > 1 else np.nan,
                   r_max_um=r.max() if len(r) else np.nan,
                   r32_um=(r ** 3).sum() / (r ** 2).sum() if len(r) else np.nan,
                   area_total_um2=a.sum(), volume_um3=(4 / 3 * np.pi * r ** 3).sum(),
                   aspect_median=np.median(aspect) if len(aspect) else np.nan,
                   frac_noncircular=float(np.mean(aspect < 0.8)) if len(aspect) else np.nan,
                   cx_um=np.average(b["x"], weights=a) * um_per_px if len(b) else np.nan,
                   cy_um=np.average(b["y"], weights=a) * um_per_px if len(b) else np.nan,
                   n_edge=int(b["edge_truncated"].sum()))
        row["polydispersity"] = row["r_std_um"] / row["r_mean_um"]
        if run_dir is not None:
            ids = set(b["bubble_id"])
            shapes = [s for i, s in enumerate(load_shapes(run_dir, fr["frame"])) if i in ids]
            row["coverage_frac"] = coverage_fraction(shapes, hw)
        rows.append(row)
    return pd.DataFrame(rows)


def size_class_counts(bubbles, edges_um):
    """Bubbles per frame in radius classes [e0, e1), [e1, e2), ..."""
    labels = [f"{lo:g}–{hi:g} µm" if np.isfinite(hi) else f"≥ {lo:g} µm" for lo, hi in zip(edges_um[:-1], edges_um[1:])]
    cls = pd.cut(bubbles["r_eq_um"], bins=edges_um, labels=labels, right=False)
    out = bubbles.assign(size_class=cls).groupby(["t", "frame", "size_class"], observed=False).size()
    return out.unstack("size_class").reset_index(), labels


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def _xaxis(ax, m, label):
    ax.set_xlabel(label)
    if len(m) <= 12:
        ax.set_xticks(m["t"])


def _end_labels(ax, x, ys, texts, min_gap_frac=0.06):
    """Direct labels at the line ends, nudged apart vertically so they never overlap."""
    ys = np.asarray(ys, float)
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = np.argsort(ys)
    pos = ys[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= max(0.0, pos[-1] - hi) if len(pos) else 0.0          # keep inside the axes
    for k, i in enumerate(order):
        ax.annotate(texts[i], (x, ys[i]), xytext=(x, pos[k]), textcoords="data", va="center", fontsize=9,
                    color=INK, annotation_clip=False)
    ax.set_ylim(lo, hi)


def _log_ticks(ax, ticks, axis="x"):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(lambda v, _: f"{v:g}")
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def plot_count(m, time_label, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    ax.plot(m["t"], m["n_bubbles"], "-o", color=C[0], ms=8, mec="white", mew=1.5)
    for _, r in m.iterrows():
        ax.annotate(f"{r.n_bubbles:.0f}", (r.t, r.n_bubbles), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, m["n_bubbles"].max() * 1.15)
    ax.set_ylabel("bubbles per frame")
    sec = ax.secondary_yaxis("right", functions=(lambda n: n / m["n_bubbles"].iloc[0] * m["density_per_mm2"].iloc[0],
                                                 lambda d: d / m["density_per_mm2"].iloc[0] * m["n_bubbles"].iloc[0]))
    sec.set_ylabel("per mm²")                    # same quantity, rescaled (not a second data series)
    _xaxis(ax, m, time_label)
    ax.set_title("Bubble count")
    return ax


def plot_mean_size(m, time_label, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    series = [("r_mean_um", "mean", C[0]), ("r_median_um", "median", C[1]), ("r32_um", "Sauter r₃₂", C[2])]
    for col, lab, col_ in series:
        ax.plot(m["t"], m[col], "-o", color=col_, ms=7, mec="white", mew=1.2, label=lab)
    ax.fill_between(m["t"], m["r_mean_um"] - m["r_sem_um"], m["r_mean_um"] + m["r_sem_um"], color=C[0], alpha=0.15, lw=0)
    ax.set_ylim(bottom=0)
    _end_labels(ax, m["t"].iloc[-1] + 0.15, [m[c].iloc[-1] for c, _, _ in series], [l for _, l, _ in series])
    ax.set_ylabel("equivalent radius [µm]")
    _xaxis(ax, m, time_label)
    ax.set_xlim(m["t"].min() - 0.3, m["t"].max() + 0.9 * (m["t"].max() - m["t"].min()) / max(len(m) - 1, 1) + 0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Average bubble size (band: mean ± s.e.)")
    return ax


def log_bins(values, n=24, lo=None, hi=None):
    lo = lo or max(np.nanmin(values) * 0.9, 1e-3)
    hi = hi or np.nanmax(values) * 1.1
    return np.geomspace(lo, hi, n + 1)


def plot_histograms(bubbles, m, var="r_eq_um", xlabel="equivalent radius [µm]", bins=None, ncols=4):
    """Small multiples: one histogram per frame (shared axes), first frame as grey outline for comparison."""
    import matplotlib.pyplot as plt
    bins = bins if bins is not None else log_bins(bubbles[var])
    n = len(m)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=True, sharey=True, squeeze=False)
    ref = bubbles.loc[bubbles["frame"] == m["frame"].iloc[0], var]
    for ax, (_, r) in zip(axs.ravel(), m.iterrows()):
        v = bubbles.loc[bubbles["frame"] == r["frame"], var]
        ax.hist(v, bins=bins, color=C[0], alpha=0.85, edgecolor="white", linewidth=0.6)
        if r["frame"] != m["frame"].iloc[0]:
            ax.hist(ref, bins=bins, histtype="step", color=INK2, lw=1.2, ls="--")
        ax.axvline(np.median(v), color=C[1], lw=1.5)
        ax.set_xscale("log")
        _log_ticks(ax, [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
                        if bins[0] <= t <= bins[-1]])
        ax.set_title(f"train {r['train']}  (n = {len(v)}, median {np.median(v):.1f})", fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("bubbles")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f"Size distribution per frame (bars; orange line: median; dashed: train {m['train'].iloc[0]})",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_cdfs(bubbles, m, var="r_eq_um", xlabel="equivalent radius [µm]", ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    cols = train_colors(len(m))
    for c, (_, r) in zip(cols, m.iterrows()):
        v = np.sort(bubbles.loc[bubbles["frame"] == r["frame"], var].to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1.8, label=f"train {r['train']}")
    ax.set_xscale("log")
    _log_ticks(ax, [t for t in (2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
                    if bubbles[var].min() * 0.8 <= t <= bubbles[var].max() * 1.2])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of bubbles ≤ size")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_title("Cumulative size distribution (light → dark: early → late)")
    return ax


def plot_scaled_distribution(bubbles, m, ax=None, bins=None):
    """PDF of r / <r>: curves collapsing onto each other means self-similar coarsening."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    cols = train_colors(len(m))
    bins = bins if bins is not None else np.geomspace(0.15, 6, 22)
    centers = np.sqrt(bins[1:] * bins[:-1])
    for c, (_, r) in zip(cols, m.iterrows()):
        v = bubbles.loc[bubbles["frame"] == r["frame"], "r_eq_um"].to_numpy()
        h, _ = np.histogram(v / v.mean(), bins=bins, density=True)
        ax.plot(centers, h, "-", color=c, lw=1.8, label=f"train {r['train']}")
    ax.set_xscale("log")
    _log_ticks(ax, [0.2, 0.3, 0.5, 1, 2, 3, 5])
    ax.set_xlabel("r / ⟨r⟩")
    ax.set_ylabel("probability density")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Scaled size distribution (collapse = self-similar coarsening)")
    return ax


def plot_size_classes(counts, labels, time_label, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    for lab, col in zip(labels, C):
        ax.plot(counts["t"], counts[lab], "-o", color=col, ms=6, mec="white", mew=1.2, label=lab)
    ax.set_ylim(bottom=0)
    _end_labels(ax, counts["t"].iloc[-1] + 0.15, [counts[l].iloc[-1] for l in labels], labels)
    ax.set_ylabel("bubbles per frame")
    ax.set_xlabel(time_label)
    if len(counts) <= 12:
        ax.set_xticks(counts["t"])
    span = counts["t"].max() - counts["t"].min()
    ax.set_xlim(counts["t"].min() - 0.03 * span - 0.3, counts["t"].max() + 0.25 * span + 0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Bubbles per size class")
    return ax


def plot_relative(m, time_label, ax=None):
    """Count, mean radius, total area and volume proxy relative to the first frame (common base = 1)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    series = [("n_bubbles", "count"), ("r_mean_um", "mean radius"), ("area_total_um2", "total area"),
              ("volume_um3", "volume (Σ 4/3πr³)")]
    ends = []
    for (col, lab), c in zip(series, C):
        y = m[col] / m[col].iloc[0]
        ax.plot(m["t"], y, "-o", color=c, ms=6, mec="white", mew=1.2, label=lab)
        ends.append(y.iloc[-1])
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    _end_labels(ax, m["t"].iloc[-1] + 0.15, ends, [l for _, l in series])
    ax.set_ylabel(f"relative to train {m['train'].iloc[0]}")
    _xaxis(ax, m, time_label)
    span = m["t"].max() - m["t"].min()
    ax.set_xlim(m["t"].min() - 0.3, m["t"].max() + 0.3 * span + 0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Coalescence check (relative change)")
    return ax


# ----------------------------------------------------------------------------
# Overlays and spatial plots
# ----------------------------------------------------------------------------
def area_norm(bubbles, var="area_um2"):
    from matplotlib.colors import LogNorm
    v = bubbles[var]
    return LogNorm(vmin=max(v.min(), 1e-6), vmax=v.max())


def plot_area_overlay(img, shapes, values, norm, cmap="RdYlGn_r", ax=None, alpha=0.45, title=None):
    """Bubbles filled by colour of `values` (e.g. area); large bubbles drawn first so small ones stay visible."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    ax = ax or plt.subplots(figsize=(10, 6.5))[1]
    ax.imshow(img, cmap="gray")
    cm = plt.get_cmap(cmap)
    order = np.argsort(-np.asarray(values))
    polys = [br.shape_outline(shapes[i])[:, ::-1] for i in order]     # (x, y)
    cols = [cm(norm(values[i])) for i in order]
    ax.add_collection(PolyCollection(polys, facecolors=[(*c[:3], alpha) for c in cols],
                                     edgecolors=[(*c[:3], 1.0) for c in cols], linewidths=0.6))
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_title(title or f"{len(shapes)} bubbles", fontsize=10)
    ax.axis("off")
    return ax


def plot_density_maps(bubbles, m, native_hw, um_per_px=3.2, cell_um=80, ncols=4):
    """Small multiples: number of bubble centres per cell (shared colour scale)."""
    import matplotlib.pyplot as plt
    H, W = native_hw[0] * um_per_px, native_hw[1] * um_per_px
    ex, ey = np.arange(0, W + cell_um, cell_um), np.arange(0, H + cell_um, cell_um)
    hs = [np.histogram2d(bubbles.loc[bubbles["frame"] == f, "y"] * um_per_px,
                         bubbles.loc[bubbles["frame"] == f, "x"] * um_per_px, bins=[ey, ex])[0] for f in m["frame"]]
    vmax = max(h.max() for h in hs)
    nrows = int(np.ceil(len(m) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.7 * nrows), squeeze=False)
    for ax, h, (_, r) in zip(axs.ravel(), hs, m.iterrows()):
        im = ax.imshow(h, extent=[0, ex[-1], ey[-1], 0], cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"train {r['train']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axs.ravel()[len(m):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label=f"bubbles per {cell_um}×{cell_um} µm²")
    fig.suptitle("Where the bubbles are (bubble centres)", fontweight="bold", fontsize=11)
    return fig


def plot_drift(m, native_hw, um_per_px=3.2, img=None, min_zoom_um=40):
    """Path of the area-weighted centroid between trains: full field (left) and zoom with step arrows (right)."""
    import matplotlib.pyplot as plt
    W, H = native_hw[1] * um_per_px, native_hw[0] * um_per_px
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    cols = train_colors(len(m))
    x, y = m["cx_um"].to_numpy(), m["cy_um"].to_numpy()
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(min_zoom_um, (x.max() - x.min()) * 1.3, (y.max() - y.min()) * 1.3) / 2 + 3
    for k, ax in enumerate(axs):
        if img is not None:
            ax.imshow(img, cmap="gray", extent=[0, W, H, 0], alpha=0.5)
        ax.grid(False)
        if k == 0:
            ax.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, color=C[1], lw=1.5))
            ax.plot(x, y, "-", color=C[1], lw=1.5)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_title("Population centroid (orange box: zoom)")
        else:
            for i in range(len(m) - 1):
                ax.annotate("", (x[i + 1], y[i + 1]), (x[i], y[i]),
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=6, shrinkB=6))
            for c, (_, r) in zip(cols, m.iterrows()):
                ax.plot(r.cx_um, r.cy_um, "o", color=c, ms=10, mec="white", mew=1.2, zorder=3)
                ax.annotate(f"{r.train}", (r.cx_um, r.cy_um), xytext=(7, 5), textcoords="offset points",
                            fontsize=9, color=INK, zorder=4)
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
            step = np.hypot(np.diff(x), np.diff(y))
            ax.set_title(f"Zoom: steps between trains (total {step.sum():.0f} µm, net {np.hypot(x[-1] - x[0], y[-1] - y[0]):.0f} µm)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
    fig.tight_layout()
    return fig


def plot_shape(bubbles, m, time_label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    asp = bubbles["minor_axis"] / bubbles["major_axis"].clip(lower=1e-9)
    ax[0].scatter(bubbles["r_eq_um"], asp, s=6, color=C[0], alpha=0.35, lw=0)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.02)
    _log_ticks(ax[0], [t for t in (2, 5, 10, 20, 50, 100) if bubbles["r_eq_um"].min() * 0.8 <= t <= bubbles["r_eq_um"].max() * 1.2])
    ax[0].set_xlabel("equivalent radius [µm]"); ax[0].set_ylabel("aspect ratio (minor / major)")
    ax[0].set_title("Shape vs size (all frames)")
    ax[1].plot(m["t"], m["frac_noncircular"] * 100, "-o", color=C[0], ms=7, mec="white", mew=1.2)
    ax[1].set_ylim(bottom=0); ax[1].set_ylabel("non-circular bubbles [%] (aspect < 0.8)")
    _xaxis(ax[1], m, time_label)
    ax[1].set_title("Elongated bubbles (fresh mergers relax to round)")
    fig.tight_layout()
    return fig
