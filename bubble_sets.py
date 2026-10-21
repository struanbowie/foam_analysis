"""
Run sets: consecutive runs that continue one another (e.g. r562 -> r563 -> r564), analysed as ONE timeline
(bubble_run_sets.ipynb). Train 8 of r562 is followed by train 0 of r563, and so on.

A set is either a results folder named `<name>_set` (written by bubble_inference.ipynb; its runs are taken from
the frame names inside, in run-number order) or an explicit list of runs, {"name": ["r562", "r563", "r564"]}.

timeline() gives every frame its position `seq` (0, 1, 2, ... through all runs and trains) and, when train and
run intervals are known, a time `t_min` since the first train of the set.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import bubble_compare as bc
from bubble_stats import C, INK, INK2, GRID


def _run_key(run):
    digits = "".join(ch for ch in str(run) if ch.isdigit())
    return (int(digits) if digits else 0, str(run))


def runs_in_folder(run_dir):
    """Runs present in a results folder, from frames.csv or else the shape file names."""
    p = os.path.join(run_dir, "frames.csv")
    if os.path.exists(p) and os.path.getsize(p) > 10:
        names = pd.read_csv(p, usecols=["frame"])["frame"]
    else:
        names = pd.Series([os.path.splitext(os.path.basename(j))[0]
                           for j in glob.glob(os.path.join(run_dir, "shapes", "*.json"))])
    return sorted({bc.run_name(n, run_dir) for n in names}, key=_run_key)


def discover_sets(results_root="results"):
    """{set name: [runs]} for every `<name>_set` folder in results_root."""
    out = {}
    for d in sorted(glob.glob(os.path.join(results_root, "*_set"))):
        if os.path.isdir(d):
            runs = runs_in_folder(d)
            if runs:
                out[os.path.basename(d)] = runs
    return out


def load_sets(run_sets, results_root="results", frame_idx=10, reviewed_only=False, in_analysis_only=True,
              export_missing=True, verbose=True):
    """Load every set. A set named like a `<name>_set` folder is read from that folder first; runs missing there
    come from their own folders (results/<run>/). Returns (bubbles, frames), both with columns set, run, train."""
    bl, fl = [], []
    for name, runs in run_sets.items():
        dirs = []
        if os.path.isdir(os.path.join(results_root, name)):
            dirs.append(os.path.join(results_root, name))
        dirs += [os.path.join(results_root, r) for r in runs if os.path.isdir(os.path.join(results_root, r))]
        if verbose:
            print(f"--- {name}: runs {', '.join(runs)} from {', '.join(dirs)}")
        try:
            b, f = bc.load_runs(dirs, frame_idx=frame_idx, reviewed_only=reviewed_only,
                                in_analysis_only=in_analysis_only, export_missing=export_missing, verbose=verbose)
        except FileNotFoundError as e:
            print(f"WARNING {name}: skipped ({e})")
            continue
        f = f[f["run"].isin(runs)]
        b = b[b["frame"].isin(f["frame"])]
        missing = [r for r in runs if r not in set(f["run"])]
        if missing:
            print(f"WARNING {name}: no frames found for {missing}")
        bl.append(b.assign(set=name))
        fl.append(f.assign(set=name))
    if not fl:
        raise FileNotFoundError("no set could be loaded")
    return pd.concat(bl, ignore_index=True), pd.concat(fl, ignore_index=True)


def timeline(frames, run_sets, train_minutes=None, run_gap_minutes=None, run_trains=None):
    """Order of every frame within its set: runs in the listed order, trains in order within a run.

    seq       0, 1, 2, ... through the whole set (missing trains leave no gap; use `slot` for that)
    slot      position counting every train of every run, including trains with no frame, from run_trains
              ({run: number of trains}; default: highest train found + 1)
    t_min     minutes since train 0 of the first run, if train_minutes (interval between trains) is given;
              run_gap_minutes is the extra time between the last train of a run and train 0 of the next
              (default: one train interval). Either may be a dict per run (run_gap_minutes: gap BEFORE that run).
    """
    out = []
    for name, runs in run_sets.items():
        f = frames[frames["set"] == name].copy()
        if f.empty:
            continue
        order = {r: i for i, r in enumerate(runs)}
        f["run_pos"] = f["run"].map(order)
        f = f.sort_values(["run_pos", "train", "frame"]).reset_index(drop=True)
        f["seq"] = np.arange(len(f))
        n_tr = {r: (run_trains or {}).get(r, int(f.loc[f["run"] == r, "train"].max()) + 1 if (f["run"] == r).any() else 0)
                for r in runs}
        start, start_t, t0 = {}, {}, 0.0
        s = 0
        for i, r in enumerate(runs):
            start[r] = s
            s += n_tr[r]
            if train_minutes is not None:
                dt = train_minutes.get(r) if isinstance(train_minutes, dict) else train_minutes
                if i > 0:
                    prev = runs[i - 1]
                    dprev = train_minutes.get(prev) if isinstance(train_minutes, dict) else train_minutes
                    gap = (run_gap_minutes.get(r) if isinstance(run_gap_minutes, dict) else run_gap_minutes)
                    t0 = start_t[prev] + (n_tr[prev] - 1) * dprev + (gap if gap is not None else dprev)
                start_t[r] = t0
        f["slot"] = [start[r] + t for r, t in zip(f["run"], f["train"])]
        if train_minutes is not None:
            f["t_min"] = [start_t[r] + t * (train_minutes.get(r) if isinstance(train_minutes, dict) else train_minutes)
                          for r, t in zip(f["run"], f["train"])]
        out.append(f)
    return pd.concat(out, ignore_index=True)


def set_metrics(bubbles, frames_tl, um_per_px=3.2, small_um=10.0):
    """bubble_compare.metrics for every frame, with set / seq / slot / t_min attached."""
    rows = []
    for name, f in frames_tl.groupby("set", sort=False):
        b = bubbles[bubbles["set"] == name]
        m = bc.metrics(b, f.assign(t=f["train"].astype(float)), um_per_px=um_per_px, small_um=small_um)
        keep = [c for c in ("frame", "set", "seq", "slot", "t_min", "run_pos") if c in f]
        m = m.merge(f[keep], on="frame", how="left")
        rows.append(m)
    return pd.concat(rows, ignore_index=True).sort_values(["set", "seq"]).reset_index(drop=True)


def boundaries(m, cols=("n_bubbles", "r_mean_um", "r32_um", "coverage_frac", "area_total_um2")):
    """Change across each run boundary (last train of a run -> first train of the next), compared with the steps
    inside the runs: ratio, and the robust z-score of ln(ratio) against all within-run steps of the set.
    |z| < 2: the next run continues smoothly; larger: a jump at the boundary (e.g. the sample moved, a new layer,
    or a long pause between runs)."""
    rows = []
    for name, g in m.sort_values("seq").groupby("set", sort=False):
        g = g.reset_index(drop=True)
        same = (g["run"].to_numpy()[1:] == g["run"].to_numpy()[:-1])
        for c in cols:
            if c not in g:
                continue
            v = g[c].to_numpy(float)
            lr = np.log(np.clip(v[1:], 1e-12, None) / np.clip(v[:-1], 1e-12, None))
            inside = lr[same & np.isfinite(lr)]
            med = np.median(inside) if len(inside) else 0.0
            mad = (np.median(np.abs(inside - med)) * 1.4826) if len(inside) > 2 else np.nan
            if not np.isfinite(mad) or mad == 0:
                mad = np.std(inside) if len(inside) > 1 else np.nan
            for i in np.where(~same)[0]:
                rows.append(dict(set=name, metric=c, from_run=g.loc[i, "run"], from_train=int(g.loc[i, "train"]),
                                 to_run=g.loc[i + 1, "run"], to_train=int(g.loc[i + 1, "train"]),
                                 before=v[i], after=v[i + 1], ratio=float(np.exp(lr[i])),
                                 typical_step=float(np.exp(np.median(np.abs(inside)))) if len(inside) else np.nan,
                                 z=float((lr[i] - med) / mad) if np.isfinite(mad) and mad else np.nan))
    return pd.DataFrame(rows)


def run_summary(m, cols=("n_bubbles", "r_mean_um", "r32_um", "coverage_frac")):
    """Per run of each set: first / last / mean of each metric and the net change within the run."""
    rows = []
    for (name, run), g in m.sort_values("seq").groupby(["set", "run"], sort=False):
        row = dict(set=name, run=run, trains=f"{int(g['train'].min())}–{int(g['train'].max())}", n_frames=len(g))
        for c in cols:
            if c in g:
                row[f"{c}_first"], row[f"{c}_last"], row[f"{c}_mean"] = g[c].iloc[0], g[c].iloc[-1], g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def whole_set_trend(m, col, x="seq", log=True):
    """Spearman ρ and a linear fit over the whole set (log y: % change per step of x)."""
    from scipy import stats
    rows = []
    for name, g in m.groupby("set", sort=False):
        g = g[[x, col]].dropna()
        g = g[g[col] > 0] if log else g
        if len(g) < 4:
            continue
        y = np.log(g[col]) if log else g[col]
        rho, p = stats.spearmanr(g[x], y)
        lr = stats.linregress(g[x], y)
        rows.append(dict(set=name, metric=col, n=len(g), rho=rho, p=p, slope=lr.slope, slope_p=lr.pvalue,
                         pct_per_step=(np.exp(lr.slope) - 1) * 100 if log else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _xcol(m, x):
    return x if x in m and m[x].notna().all() else "seq"


def plot_timeline(m, col, ylabel, ax=None, x="seq", colors=None, title=None, annotate=False, logy=False):
    """One metric through a whole set: one line through every train of every run, markers in the run's colour,
    dashed lines at run boundaries with the run names on top. x: "seq" (frame order), "slot" (gaps for missing
    trains) or "t_min" (real time)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(12, 3.8))[1]
    x = _xcol(m, x)
    g = m.sort_values("seq")
    colors = colors or bc.run_colors(g["run"].unique())
    ax.plot(g[x], g[col], "-", color=INK2, lw=1.2, zorder=2)
    for run, r in g.groupby("run", sort=False):
        ax.plot(r[x], r[col], "o", color=colors[run], ms=7, mec="white", mew=1.2, zorder=3, label=run)
        if annotate:
            for xv, yv in zip(r[x], r[col]):
                ax.annotate(f"{yv:.0f}", (xv, yv), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=7, color=INK2)
    runs = list(dict.fromkeys(g["run"]))
    xs = g[x].to_numpy()
    for i, run in enumerate(runs):
        sel = (g["run"] == run).to_numpy()
        lo, hi = xs[sel].min(), xs[sel].max()
        if i > 0:
            prev_hi = xs[(g["run"] == runs[i - 1]).to_numpy()].max()
            ax.axvline((prev_hi + lo) / 2, color=INK2, lw=0.9, ls="--", zorder=1)
        ax.text((lo + hi) / 2, 1.01, run, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, color=colors[run], fontweight="bold")
    if x in ("seq", "slot"):
        ax.set_xticks(xs, [str(int(t)) for t in g["train"]], fontsize=8)
        ax.set_xlabel("train (runs in order)")
    else:
        ax.set_xlabel("time since the first train [min]")
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10, pad=16)
    return ax


def plot_timelines(m, specs, x="seq", annotate_first=True):
    """Small multiples: one row per metric (specs: [(column, label)]), one column per set."""
    import matplotlib.pyplot as plt
    sets = list(dict.fromkeys(m["set"]))
    colors = bc.run_colors(m["run"].unique())
    fig, axs = plt.subplots(len(specs), len(sets), figsize=(max(7, 0.45 * m.groupby("set").size().max() + 3) * len(sets),
                                                          3.0 * len(specs)), squeeze=False)
    for j, s in enumerate(sets):
        g = m[m["set"] == s]
        for i, (col, label) in enumerate(specs):
            plot_timeline(g, col, label, ax=axs[i, j], x=x, colors=colors,
                          title=s if i == 0 else None, annotate=annotate_first and i == 0)
            if i < len(specs) - 1:
                axs[i, j].set_xlabel("")
    fig.tight_layout()
    return fig


def plot_relative_runs(m, col, ylabel, x_within="train"):
    """Each run of each set against its own train number, divided by its first train: do consecutive runs
    evolve the same way within a run?"""
    import matplotlib.pyplot as plt
    sets = list(dict.fromkeys(m["set"]))
    colors = bc.run_colors(m["run"].unique())
    fig, axs = plt.subplots(1, len(sets), figsize=(6.5 * len(sets), 3.8), squeeze=False)
    for ax, s in zip(axs[0], sets):
        for run, g in m[m["set"] == s].sort_values("seq").groupby("run", sort=False):
            ax.plot(g[x_within], g[col] / g[col].iloc[0], "-o", color=colors[run], ms=5, mec="white", label=run)
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("train within the run"); ax.set_ylabel(f"{ylabel} ÷ first train")
        ax.set_title(s, fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_size_timeline(bubbles, m, var="r_eq_um", label="equivalent radius [µm]"):
    """Size distribution of every frame of each set as one column, in timeline order, with the median and run
    boundaries."""
    import matplotlib.pyplot as plt
    import bubble_stats as bst
    sets = list(dict.fromkeys(m["set"]))
    bins = bst.log_bins(bubbles[var], n=30)
    fig, axs = plt.subplots(len(sets), 1, figsize=(max(9, 0.4 * m.groupby("set").size().max() + 3), 3.6 * len(sets)),
                            squeeze=False)
    for ax, s in zip(axs[:, 0], sets):
        g = m[m["set"] == s].sort_values("seq")
        H = np.array([np.histogram(bubbles.loc[bubbles["frame"] == f, var], bins=bins)[0] for f in g["frame"]], float)
        H /= np.maximum(H.sum(axis=1, keepdims=True), 1)
        xe = np.arange(len(g) + 1) - 0.5
        pc = ax.pcolormesh(xe, bins, H.T, cmap="Blues", shading="flat")
        ax.plot(np.arange(len(g)), [bubbles.loc[bubbles["frame"] == f, var].median() for f in g["frame"]],
                color=C[1], lw=1.5, label="median")
        runs = g["run"].to_numpy()
        for i in np.where(runs[1:] != runs[:-1])[0]:
            ax.axvline(i + 0.5, color=INK, lw=1, ls="--")
        for run in dict.fromkeys(runs):
            idx = np.where(runs == run)[0]
            ax.text(idx.mean(), 1.01, run, transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
        ax.set_yscale("log")
        bst._log_ticks(ax, [v for v in (2, 3, 5, 10, 20, 30, 50, 100, 200) if bins[0] <= v <= bins[-1]], axis="y")
        ax.set_xticks(np.arange(len(g)), [str(int(t)) for t in g["train"]], fontsize=8)
        ax.set_xlabel("train (runs in order)"); ax.set_ylabel(label); ax.grid(False)
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(pc, ax=ax, label="fraction of the frame's bubbles")
        ax.set_title(f"{s}: size distribution through the set", fontsize=10, pad=16)
    fig.tight_layout()
    return fig
