"""
Statistics and plots for exported bubble results (bubble_analysis.ipynb).

Input: the `bubbles` / `frames` tables from bubble_io.load_results, with a time column `t`
(see train_times: the train number, or real times given in the notebook). Results may hold frames of
several runs (column `run`); labels then name the run as well as the train.

Metrics per frame (see frame_metrics):
    n_bubbles, density_per_mm2          number of bubbles, per mm^2 of field of view
    r_mean_um (+ r_sem_um), r_median_um, r_std_um, r_max_um
    r32_um                              Sauter mean radius sum(r^3)/sum(r^2), weights large bubbles (volume/surface)
    polydispersity                      r_std / r_mean
    area_total_um2                      sum of bubble areas (overlapping bubbles counted fully); fov_mm2 field of view
    coverage_frac                       fraction of the frame covered by at least one bubble (union of outlines)
    volume_um3                          sum of 4/3 pi r^3: sphere-equivalent gas volume proxy
    aspect_median, frac_noncircular     minor/major axis; share of bubbles with aspect < 0.8
    cx_um, cy_um                        area-weighted centroid of the bubble population
"""
from __future__ import annotations

import json
import os
import re

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


def train_label(r):
    lab = r.get("label") if hasattr(r, "get") else None
    return lab if isinstance(lab, str) and lab else f"train {r['train']}"


def train_times(frames, times=None, verbose=True):
    """Time `t` of every frame from its run and train.
    times: None (the train number; with several runs the trains are numbered on from one run to the next, in
    the order of the run names), or a dict {train: t} or {(run, train): t} (real times, e.g. minutes)."""
    runs = sorted(frames["run"].unique()) if "run" in frames else [None]
    if times:
        def get(r):
            for k in ((r.get("run"), r["train"]), r["train"]):
                if k in times:
                    return times[k]
            raise KeyError(f"no time for run {r.get('run')} train {r['train']} in TRAIN_TIMES")
        return frames.apply(get, axis=1).astype(float)
    if len(runs) <= 1:
        return frames["train"].astype(float)
    offset, off = {}, 0
    for run in runs:
        offset[run] = off
        off += int(frames.loc[frames["run"] == run, "train"].max()) + 1
    if verbose:
        print("several runs: t = train number counted on from one run to the next (" + ", ".join(
            f"{run}: train 0 = {o}" for run, o in offset.items()) + "); set TRAIN_TIMES for real times")
    return (frames["train"] + frames["run"].map(offset)).astype(float)


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
    multi_run = "run" in frames and frames["run"].nunique() > 1
    for _, fr in frames.sort_values(["t", "frame"]).iterrows():
        b = bubbles[bubbles["frame"] == fr["frame"]]
        r = b["r_eq_um"].to_numpy()
        a = b["area_um2"].to_numpy()
        hw = (fr["image_height_px"], fr["image_width_px"])
        fov_mm2 = hw[0] * hw[1] * (um_per_px * 1e-3) ** 2
        aspect = (b["minor_axis"] / b["major_axis"].clip(lower=1e-9)).to_numpy()
        run = fr.get("run")
        row = dict(run=run, train=fr["train"], frame_idx=fr.get("frame_idx"), t=fr["t"], frame=fr["frame"],
                   label=f"{run} train {fr['train']}" if multi_run else f"train {fr['train']}",
                   tag=f"{run}/{fr['train']}" if multi_run else f"{fr['train']}", n_bubbles=len(b),
                   density_per_mm2=len(b) / fov_mm2,
                   r_mean_um=r.mean() if len(r) else np.nan,
                   r_sem_um=r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan,
                   r_median_um=np.median(r) if len(r) else np.nan,
                   r_std_um=r.std(ddof=1) if len(r) > 1 else np.nan,
                   r_max_um=r.max() if len(r) else np.nan,
                   r32_um=(r ** 3).sum() / (r ** 2).sum() if len(r) else np.nan,
                   area_total_um2=a.sum(), fov_mm2=fov_mm2, volume_um3=(4 / 3 * np.pi * r ** 3).sum(),
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


def _many(m):
    return len(m) > 15


def _series(ax, x, y, color, label=None, many=False, smooth=1, ms=7):
    """One time series: markers for few points; for many points a thin line, optionally with a centred
    running mean (raw values faint) so frame-to-frame measurement noise does not hide the trend."""
    x, y = np.asarray(x), pd.Series(np.asarray(y, float))
    if smooth and smooth > 1:
        ax.plot(x, y, "-", color=color, lw=0.8, alpha=0.35)
        ys = y.rolling(int(smooth), center=True, min_periods=1).mean()
        ax.plot(x, ys, "-", color=color, lw=2.2, label=label)
        return ys.to_numpy()
    if many:
        ax.plot(x, y, "-", color=color, lw=1.6, label=label)
    else:
        ax.plot(x, y, "-o", color=color, ms=ms, mec="white", mew=1.2, label=label)
    return y.to_numpy()


def _label_x(m):
    span = m["t"].max() - m["t"].min()
    return m["t"].iloc[-1] + 0.02 * (span if span else 1)


def _room_right(ax, m, frac=0.18):
    span = m["t"].max() - m["t"].min() or 1
    ax.set_xlim(m["t"].min() - 0.03 * span, m["t"].max() + frac * span)


def select_snapshots(m, n=8):
    """n evenly spaced frames of m (always including the first and last)."""
    if len(m) <= n:
        return m.reset_index(drop=True)
    return m.iloc[np.unique(np.linspace(0, len(m) - 1, n).round().astype(int))].reset_index(drop=True)


def _log_ticks(ax, ticks, axis="x"):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(lambda v, _: f"{v:g}")
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def plot_count(m, time_label, ax=None, smooth=1):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    many = _many(m)
    _series(ax, m["t"], m["n_bubbles"], C[0], many=many, smooth=smooth, ms=8)
    if not many:
        for _, r in m.iterrows():
            ax.annotate(f"{r.n_bubbles:.0f}", (r.t, r.n_bubbles), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, m["n_bubbles"].max() * 1.15)
    ax.set_ylabel("bubbles per frame")
    k = (m["density_per_mm2"] / m["n_bubbles"])[m["n_bubbles"] > 0]       # 1 / field of view [mm²]
    if len(k):
        k = float(k.iloc[0])
        sec = ax.secondary_yaxis("right", functions=(lambda n: n * k, lambda d: d / k))
        sec.set_ylabel("per mm²")                # same quantity, rescaled (not a second data series)
    _xaxis(ax, m, time_label)
    ax.set_title("Bubble count" + (f" (line: {smooth}-frame running mean)" if smooth and smooth > 1 else ""))
    return ax


def plot_mean_size(m, time_label, ax=None, smooth=1):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    many = _many(m)
    series = [("r_mean_um", "mean", C[0]), ("r_median_um", "median", C[1]), ("r32_um", "Sauter r₃₂", C[2])]
    ends = [_series(ax, m["t"], m[col], col_, label=lab, many=many, smooth=smooth)[-1] for col, lab, col_ in series]
    ax.fill_between(m["t"], m["r_mean_um"] - m["r_sem_um"], m["r_mean_um"] + m["r_sem_um"], color=C[0], alpha=0.15, lw=0)
    ax.set_ylim(bottom=0)
    _end_labels(ax, _label_x(m), ends, [l for _, l, _ in series])
    ax.set_ylabel("equivalent radius [µm]")
    _xaxis(ax, m, time_label)
    _room_right(ax, m)
    ax.set_title("Average bubble size (band: mean ± s.e.)")
    return ax


def plot_total_area(m, time_label, ax=None, smooth=1):
    """Total bubble area per frame (sum of all bubbles, overlaps counted fully) and, if computed, the area covered
    by bubbles (overlaps counted once), in mm²; right axis: the same as % of the field of view."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    many = _many(m)
    series = [(m["area_total_um2"] * 1e-6, "total (Σ areas)", C[0])]
    if "coverage_frac" in m:
        series.append((m["coverage_frac"] * m["fov_mm2"], "covered (overlaps once)", C[1]))
    ends = [_series(ax, m["t"], y, col, label=lab, many=many, smooth=smooth)[-1] for y, lab, col in series]
    ax.set_ylim(bottom=0)
    _end_labels(ax, _label_x(m), ends, [lab for _, lab, _ in series])
    ax.set_ylabel("bubble area [mm²]")
    fov = float(m["fov_mm2"].iloc[0])
    sec = ax.secondary_yaxis("right", functions=(lambda v: v / fov * 100, lambda p: p * fov / 100))
    sec.set_ylabel("% of the field of view")      # same quantity, rescaled (not a second data series)
    _xaxis(ax, m, time_label)
    _room_right(ax, m, 0.45)
    ax.set_title("Total bubble area" + (f" (line: {smooth}-frame running mean)" if smooth and smooth > 1 else ""))
    return ax


def log_bins(values, n=24, lo=None, hi=None):
    lo = lo or max(np.nanmin(values) * 0.9, 1e-3)
    hi = hi or np.nanmax(values) * 1.1
    return np.geomspace(lo, hi, n + 1)


def plot_histograms(bubbles, m, var="r_eq_um", xlabel="equivalent radius [µm]", bins=None, ncols=4, frame_label=train_label):
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
        ax.set_title(f"{frame_label(r)}  (n = {len(v)}, median {np.median(v):.1f})", fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("bubbles")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f"Size distribution per frame (bars; orange line: median; dashed: {frame_label(m.iloc[0])})",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_cdfs(bubbles, m, var="r_eq_um", xlabel="equivalent radius [µm]", ax=None, frame_label=train_label):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    cols = train_colors(len(m))
    for c, (_, r) in zip(cols, m.iterrows()):
        v = np.sort(bubbles.loc[bubbles["frame"] == r["frame"], var].to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1.8, label=frame_label(r))
    ax.set_xscale("log")
    _log_ticks(ax, [t for t in (2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
                    if bubbles[var].min() * 0.8 <= t <= bubbles[var].max() * 1.2])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of bubbles ≤ size")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_title("Cumulative size distribution (light → dark: early → late)")
    return ax


def plot_scaled_distribution(bubbles, m, ax=None, bins=None, frame_label=train_label):
    """PDF of r / <r>: curves collapsing onto each other means self-similar coarsening."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    cols = train_colors(len(m))
    bins = bins if bins is not None else np.geomspace(0.15, 6, 22)
    centers = np.sqrt(bins[1:] * bins[:-1])
    for c, (_, r) in zip(cols, m.iterrows()):
        v = bubbles.loc[bubbles["frame"] == r["frame"], "r_eq_um"].to_numpy()
        h, _ = np.histogram(v / v.mean(), bins=bins, density=True)
        ax.plot(centers, h, "-", color=c, lw=1.8, label=frame_label(r))
    ax.set_xscale("log")
    _log_ticks(ax, [0.2, 0.3, 0.5, 1, 2, 3, 5])
    ax.set_xlabel("r / ⟨r⟩")
    ax.set_ylabel("probability density")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Scaled size distribution (collapse = self-similar coarsening)")
    return ax


def plot_size_classes(counts, labels, time_label, ax=None, smooth=1):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    many = _many(counts)
    ends = [_series(ax, counts["t"], counts[lab], col, label=lab, many=many, smooth=smooth, ms=6)[-1]
            for lab, col in zip(labels, C)]
    ax.set_ylim(bottom=0)
    _end_labels(ax, _label_x(counts), ends, labels)
    ax.set_ylabel("bubbles per frame")
    ax.set_xlabel(time_label)
    if len(counts) <= 12:
        ax.set_xticks(counts["t"])
    _room_right(ax, counts, 0.25)
    ax.set_title("Bubbles per size class")
    return ax


def plot_relative(m, time_label, ax=None, smooth=1):
    """Count, mean radius, total area and volume proxy relative to the first frame (common base = 1)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    series = [("n_bubbles", "count"), ("r_mean_um", "mean radius"), ("area_total_um2", "total area"),
              ("volume_um3", "volume (Σ 4/3πr³)")]
    many = _many(m)
    ends = []
    for (col, lab), c in zip(series, C):
        base = m[col].iloc[: max(1, int(smooth))].mean() if smooth and smooth > 1 else m[col].iloc[0]
        ends.append(_series(ax, m["t"], m[col] / base, c, label=lab, many=many, smooth=smooth, ms=6)[-1])
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    _end_labels(ax, _label_x(m), ends, [l for _, l in series])
    ax.set_ylabel("relative to the first frame" if many else f"relative to {train_label(m.iloc[0])}")
    _xaxis(ax, m, time_label)
    _room_right(ax, m, 0.3)
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


def plot_density_maps(bubbles, m, native_hw, um_per_px=3.2, cell_um=80, ncols=4, frame_label=train_label):
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
        ax.set_title(frame_label(r), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axs.ravel()[len(m):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label=f"bubbles per {cell_um}×{cell_um} µm²")
    fig.suptitle("Where the bubbles are (bubble centres)", fontweight="bold", fontsize=11)
    return fig


def plot_drift(m, native_hw, um_per_px=3.2, img=None, min_zoom_um=40, time_label="time"):
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
        elif _many(m):     # many frames: path coloured by time, first / last labelled
            from matplotlib.collections import LineCollection
            pts = np.c_[x, y].reshape(-1, 1, 2)
            from matplotlib.colors import ListedColormap
            blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.35, 1, 256)))   # skip the near-white end
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=blues,
                                norm=plt.Normalize(m["t"].min(), m["t"].max()), lw=2)
            lc.set_array(m["t"].to_numpy()[:-1])
            ax.add_collection(lc)
            for i, lab in ((0, "start"), (len(m) - 1, "end")):
                ax.plot(x[i], y[i], "o", color=blues(0.0 if i == 0 else 1.0), ms=9, mec="white", mew=1.2, zorder=3)
                ax.annotate(lab, (x[i], y[i]), xytext=(7, 5), textcoords="offset points", fontsize=9, color=INK)
            fig.colorbar(lc, ax=ax, shrink=0.8, label=time_label)
        else:
            for i in range(len(m) - 1):
                ax.annotate("", (x[i + 1], y[i + 1]), (x[i], y[i]),
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=6, shrinkB=6))
            for c, (_, r) in zip(cols, m.iterrows()):
                ax.plot(r.cx_um, r.cy_um, "o", color=c, ms=10, mec="white", mew=1.2, zorder=3)
                ax.annotate(str(r.get("tag", r.train)), (r.cx_um, r.cy_um), xytext=(7, 5), textcoords="offset points",
                            fontsize=9, color=INK, zorder=4)
        if k == 1:
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
            step = np.hypot(np.diff(x), np.diff(y))
            what = "path" if _many(m) else "steps between trains"
            ax.set_title(f"Zoom: {what} (total {step.sum():.0f} µm, net {np.hypot(x[-1] - x[0], y[-1] - y[0]):.0f} µm)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
    fig.tight_layout()
    return fig


def plot_shape(bubbles, m, time_label, smooth=1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    asp = bubbles["minor_axis"] / bubbles["major_axis"].clip(lower=1e-9)
    ax[0].scatter(bubbles["r_eq_um"], asp, s=6, color=C[0], alpha=0.35, lw=0)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.02)
    _log_ticks(ax[0], [t for t in (2, 5, 10, 20, 50, 100) if bubbles["r_eq_um"].min() * 0.8 <= t <= bubbles["r_eq_um"].max() * 1.2])
    ax[0].set_xlabel("equivalent radius [µm]"); ax[0].set_ylabel("aspect ratio (minor / major)")
    ax[0].set_title("Shape vs size (all frames)")
    _series(ax[1], m["t"], m["frac_noncircular"] * 100, C[0], many=_many(m), smooth=smooth)
    ax[1].set_ylim(bottom=0); ax[1].set_ylabel("non-circular bubbles [%] (aspect < 0.8)")
    _xaxis(ax[1], m, time_label)
    ax[1].set_title("Elongated bubbles (fresh mergers relax to round)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Many frames: size-distribution heatmap and overlay GIF
# ----------------------------------------------------------------------------
def plot_size_heatmap(bubbles, m, time_label, var="r_eq_um", label="equivalent radius [µm]", bins=None, ax=None):
    """Size distribution of every frame as one column (fraction of that frame's bubbles per size bin),
    with the median overlaid. Shows the whole evolution in one picture."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(12, 4.2))[1]
    bins = bins if bins is not None else log_bins(bubbles[var], n=30)
    H = np.array([np.histogram(bubbles.loc[bubbles["frame"] == f, var], bins=bins)[0] for f in m["frame"]], float)
    H /= np.maximum(H.sum(axis=1, keepdims=True), 1)
    t = m["t"].to_numpy()
    dt = np.median(np.diff(t)) if len(t) > 1 else 1
    te = np.r_[t - dt / 2, t[-1] + dt / 2]
    pc = ax.pcolormesh(te, bins, H.T, cmap="Blues", shading="flat")
    med = [bubbles.loc[bubbles["frame"] == f, var].median() for f in m["frame"]]
    ax.plot(t, med, color=C[1], lw=1.5, label="median")
    ax.set_yscale("log")
    _log_ticks(ax, [v for v in (2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
                    if bins[0] <= v <= bins[-1]], axis="y")
    ax.set_xlabel(time_label); ax.set_ylabel(label); ax.grid(False)
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(pc, ax=ax, label="fraction of the frame's bubbles")
    ax.set_title("Size distribution over time (each column = one frame)")
    return ax


def save_gif(figures, out_path, fps=10, dpi=90):
    """Write matplotlib figures (any iterable; each is closed after rendering) as an animated GIF."""
    import io as _io
    import matplotlib.pyplot as plt
    from PIL import Image
    frames_out = []
    for fig in figures:
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames_out.append(Image.open(buf).convert("RGB").quantize(colors=200, method=Image.Quantize.MEDIANCUT))
    if not frames_out:
        raise ValueError("no frames for the GIF")
    size = frames_out[0].size
    frames_out = [f if f.size == size else f.resize(size) for f in frames_out]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frames_out[0].save(out_path, save_all=True, append_images=frames_out[1:], duration=int(1000 / fps), loop=0)
    return out_path, len(frames_out)


def make_overlay_gif(run_dir, bubbles, m, norm, out_path, cmap="RdYlGn_r", alpha=0.45, every=1, fps=10,
                     width_in=8.0, dpi=90, time_fmt=lambda r: f"frame {r['frame_idx']}"):
    """Animated GIF of the area-coloured overlays (one shared colour scale) for every `every`-th frame of m."""
    import matplotlib.pyplot as plt
    import bubble_io as bio

    def figures():
        for _, r in m.iloc[::max(1, int(every))].iterrows():
            img, shapes = bio.load_frame(run_dir, r["frame"])
            b = bubbles[bubbles["frame"] == r["frame"]]
            h = width_in * img.shape[0] / img.shape[1]
            fig, ax = plt.subplots(figsize=(width_in * 1.15, h + 0.5))
            plot_area_overlay(img, [shapes[i] for i in b["bubble_id"]], b["area_um2"].to_numpy(), norm,
                              cmap=cmap, ax=ax, alpha=alpha, title=f"{time_fmt(r)}  ·  {len(b)} bubbles")
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, label="bubble area [µm²]")
            yield fig

    return save_gif(figures(), out_path, fps=fps, dpi=dpi)


# ----------------------------------------------------------------------------
# Several runs on one plot (bubble_compare_runs.ipynb)
# ----------------------------------------------------------------------------
# categorical palette (dataviz reference, 8 slots, fixed order): one colour per run, assigned in run order
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RUN_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TRAIN_COLS = ["n_bubbles", "density_per_mm2", "r_mean_um", "r_median_um", "r32_um", "r_max_um", "polydispersity",
              "area_total_um2", "coverage_frac", "volume_um3", "aspect_median", "frac_noncircular", "fov_mm2"]


def load_runs(results_dirs, reviewed_only=False, in_analysis_only=True, compute_coverage=True, um_per_px=3.2):
    """Load several results folders and compute the per-frame metrics of each.
    Returns (bubbles, m): all bubbles and one row per frame, both with a `run` column. Folders without exported
    frames are skipped (with a message)."""
    import bubble_io as bio
    bubbles, metrics = [], []
    for d in results_dirs:
        b, f = bio.load_results(d, reviewed_only=reviewed_only, in_analysis_only=in_analysis_only)
        if not len(f):
            print(f"{d}: no exported frames, skipped (run the export in bubble_inference.ipynb first)")
            continue
        f = f.assign(t=f["train"].astype(float))
        b = b.assign(t=b["train"].astype(float), results_dir=d)
        m = frame_metrics(b, f, run_dir=d if compute_coverage else None, um_per_px=um_per_px)
        bubbles.append(b)
        metrics.append(m.assign(results_dir=d))
    if not metrics:
        raise ValueError("no frames in any of the results folders")
    return pd.concat(bubbles, ignore_index=True), pd.concat(metrics, ignore_index=True)


def per_train(m):
    """One row per (run, train): metrics averaged over the frames of the train (n_frames; *_sem when > 1 frame)."""
    cols = [c for c in TRAIN_COLS if c in m]
    g = m.groupby(["run", "train"], sort=True)
    out = g[cols].mean()
    sem = g[cols].sem().add_suffix("_sem")
    return pd.concat([out, sem], axis=1).assign(n_frames=g.size()).reset_index()


def find_scan_trains(mt, scan=None, min_jump=1.3, verbose=True):
    """Train of each run in which the new bubble population of the laser scan appears: the largest rise of the
    bubble count from one train to the next (scan: {run: train} to set it by hand, for some or all runs).
    Returns {run: train or None}; runs whose largest rise is below min_jump x get None (no clear scan)."""
    scan = dict(scan or {})
    out, rows = {}, []
    for run, g in mt.sort_values("train").groupby("run"):
        n, tr = g["n_bubbles"].to_numpy(float), g["train"].to_numpy()
        ratio = n[1:] / np.maximum(n[:-1], 1)
        k = int(np.argmax(ratio)) + 1 if len(ratio) else None
        if run in scan:
            out[run], how = scan[run], "given"
        elif k is not None and ratio[k - 1] >= min_jump:
            out[run], how = int(tr[k]), "found"
        else:
            out[run], how = None, "no clear jump"
        s = out[run]
        if s is not None and (tr == s).any() and (tr < s).any():
            before, after = n[tr < s][-1], n[tr == s][0]
            rows.append(dict(run=run, scan_train=s, how=how, count_before=int(before), count_at_scan=int(after),
                             jump=round(after / max(before, 1), 2)))
        else:
            rows.append(dict(run=run, scan_train=s, how=how))
    if verbose:
        from IPython.display import display
        display(pd.DataFrame(rows))
    return out


def align_runs(mt, scan, align=True, normalise=False, cols=None):
    """Add x (train, or trains since the scan: 0 = first train after it) and, with normalise, divide every metric
    by its mean over the run's trains before the scan (1 = as before the scan). Runs without a scan train are
    dropped when aligning or normalising."""
    mt = mt.copy()
    mt["scan_train"] = mt["run"].map(scan)
    if align or normalise:
        drop = sorted(mt.loc[mt["scan_train"].isna(), "run"].unique())
        if drop:
            print("no scan train for", ", ".join(drop), "-> left out (set SCAN_TRAINS)")
        mt = mt[mt["scan_train"].notna()].copy()
    mt["x"] = mt["train"] - mt["scan_train"] if align else mt["train"]
    if normalise:
        cols = cols or [c for c in TRAIN_COLS if c in mt]
        for run, g in mt.groupby("run"):
            base = g.loc[g["train"] < g["scan_train"], cols].mean()
            mt.loc[g.index, cols] = g[cols] / base.where(base != 0)
            sems = [c + "_sem" for c in cols if c + "_sem" in mt]
            mt.loc[g.index, sems] = g[sems] / base.where(base != 0)[[s[:-4] for s in sems]].to_numpy()
    return mt


def run_colors(runs):
    runs = sorted(runs)
    if len(runs) > len(RUN_COLORS):
        raise ValueError(f"{len(runs)} runs: at most {len(RUN_COLORS)} runs per plot (split them into groups)")
    return {r: (RUN_COLORS[i], RUN_MARKERS[i]) for i, r in enumerate(runs)}


def plot_runs(mt, col, ylabel, ax=None, align=True, normalise=False, title=None, colors=None, legend=True,
              scale=1.0):
    """One line per run: the metric col of each train against x (see align_runs). Error bars = s.e. over the
    frames of a train (several frames per train). The laser scan is marked by a dashed line when aligned."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    colors = colors or run_colors(mt["run"].unique())
    for run, g in mt.sort_values("x").groupby("run"):
        c, mk = colors[run]
        y = g[col] * scale
        ax.plot(g["x"], y, "-", marker=mk, color=c, ms=7, mec="white", mew=1, lw=2, label=run)
        if col + "_sem" in g and g[col + "_sem"].notna().any():
            ax.errorbar(g["x"], y, yerr=g[col + "_sem"] * scale, fmt="none", ecolor=c, elinewidth=1, capsize=2)
    if align:
        ax.axvline(-0.5, color=INK2, lw=1, ls="--")
        ax.annotate("laser scan", (-0.5, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=8, color=INK2)
    if normalise:      # log scale: a x30 jump and a x1.5 jump stay readable on one plot
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ticks = (0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20, 30, 50, 100) if hi / lo > 4 else \
            (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3)
        _log_ticks(ax, [t for t in ticks if lo <= t <= hi] or [1], axis="y")
    xs = np.sort(mt["x"].unique())
    if len(xs) <= 16:
        ax.set_xticks(xs)
    ax.set_xlabel("trains since the laser scan (0 = first train after it)" if align else "train")
    ax.set_ylabel(re.sub(r"\s*\[[^]]*\]", "", ylabel) + " (relative to before the scan)" if normalise else ylabel)
    if not normalise:
        ax.set_ylim(bottom=0)
    ax.set_title(title or (re.sub(r"\s*\[[^]]*\]", "", ylabel) if normalise else ylabel))
    if legend:
        ax.legend(fontsize=8, ncol=2 if len(colors) > 4 else 1)
    return ax


def plot_runs_grid(mt, panels, align=True, normalise=False, ncols=2, colors=None, suptitle=None):
    """Small multiples: one panel per (col, label[, scale]) with every run in each panel; one shared legend."""
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows + 0.6), squeeze=False, layout="constrained")
    colors = colors or run_colors(mt["run"].unique())
    for ax, p in zip(axs.ravel(), panels):
        col, label = p[0], p[1]
        plot_runs(mt, col, label, ax=ax, align=align, normalise=normalise, colors=colors, legend=False,
                  scale=p[2] if len(p) > 2 else 1.0)
    for ax in axs.ravel()[len(panels):]:
        ax.set_visible(False)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    fig.legend(h, l, loc="outside lower center", ncol=min(len(l), 8), frameon=False)
    return fig


def scan_jump_table(mt, scan, cols=("n_bubbles", "r_mean_um", "r_median_um", "r32_um", "area_total_um2", "coverage_frac")):
    """Per run: each metric just before the scan (last train before it), in the scan train, and their ratio."""
    rows = []
    for run, g in mt.groupby("run"):
        s = scan.get(run)
        if s is None or not (g["train"] < s).any() or not (g["train"] == s).any():
            continue
        before = g[g["train"] < s].sort_values("train").iloc[-1]
        after = g[g["train"] == s].iloc[0]
        row = dict(run=run, scan_train=s)
        for c in cols:
            if c in g:
                row[f"{c} before"], row[f"{c} after"] = before[c], after[c]
                row[f"{c} ratio"] = after[c] / before[c] if before[c] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_size_before_after(bubbles, scan, var="r_eq_um", xlabel="equivalent radius [µm]", ncols=3):
    """Per run: cumulative size distribution of the last train before the scan and of the scan train."""
    import matplotlib.pyplot as plt
    runs = [r for r in sorted(bubbles["run"].unique()) if scan.get(r) is not None]
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), sharex=True, sharey=True, squeeze=False)
    lo, hi = bubbles[var].min() * 0.8, bubbles[var].max() * 1.2
    for ax, run in zip(axs.ravel(), runs):
        b = bubbles[bubbles["run"] == run]
        s = scan[run]
        pre = sorted(t for t in b["train"].unique() if t < s)
        for tr, lab, c in ((pre[-1] if pre else None, "before", C[0]), (s, "after", C[1])):
            if tr is None:
                continue
            bt = b[b["train"] == tr]
            for k, (_, bf) in enumerate(bt.groupby("frame")):
                v = np.sort(bf[var].to_numpy())
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1.8,
                        label=f"{lab} (train {tr}, median {np.median(v):.1f})" if k == 0 else None)
        ax.set_xscale("log")
        _log_ticks(ax, [t for t in (2, 3, 5, 10, 20, 30, 50, 100, 200, 500) if lo <= t <= hi])
        ax.set_ylim(0, 1)
        ax.set_title(run, fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.tick_params(labelbottom=True)
    for ax in axs.ravel()[len(runs):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel("fraction of bubbles ≤ size")
    fig.suptitle("Size distribution just before and just after the laser scan", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig
