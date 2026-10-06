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


def train_label(r):
    return f"train {r['train']}"


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
    sec = ax.secondary_yaxis("right", functions=(lambda n: n / m["n_bubbles"].iloc[0] * m["density_per_mm2"].iloc[0],
                                                 lambda d: d / m["density_per_mm2"].iloc[0] * m["n_bubbles"].iloc[0]))
    sec.set_ylabel("per mm²")                    # same quantity, rescaled (not a second data series)
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
    ax.set_ylabel("relative to the first frame" if many else f"relative to train {m['train'].iloc[0]}")
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
                ax.annotate(f"{r.train}", (r.cx_um, r.cy_um), xytext=(7, 5), textcoords="offset points",
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


def make_overlay_gif(run_dir, bubbles, m, norm, out_path, cmap="RdYlGn_r", alpha=0.45, every=1, fps=10,
                     width_in=8.0, dpi=90, time_fmt=lambda r: f"frame {r['frame_idx']}"):
    """Animated GIF of the area-coloured overlays (one shared colour scale) for every `every`-th frame of m."""
    import io as _io
    import matplotlib.pyplot as plt
    from PIL import Image
    import bubble_io as bio
    rows = m.iloc[::max(1, int(every))]
    frames_out = []
    for _, r in rows.iterrows():
        img, shapes = bio.load_frame(run_dir, r["frame"])
        b = bubbles[bubbles["frame"] == r["frame"]]
        h = width_in * img.shape[0] / img.shape[1]
        fig, ax = plt.subplots(figsize=(width_in * 1.15, h + 0.5))
        plot_area_overlay(img, [shapes[i] for i in b["bubble_id"]], b["area_um2"].to_numpy(), norm,
                          cmap=cmap, ax=ax, alpha=alpha, title=f"{time_fmt(r)}  ·  {len(b)} bubbles")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, label="bubble area [µm²]")
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

