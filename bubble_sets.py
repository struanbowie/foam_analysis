"""
Run sets: consecutive runs that continue one another (e.g. r562 -> r563 -> r564), analysed as ONE timeline
(bubble_run_sets.ipynb). Train 8 of r562 is followed by train 0 of r563, and so on.

A set is either a results folder named `<name>_set` (written by bubble_inference.ipynb; its runs are taken from
the frame names inside, in run-number order) or an explicit list of runs, {"name": ["r562", "r563", "r564"]}.

timeline() gives every frame its position `seq` (0, 1, 2, ... through all runs and trains) and, when train and
run start times are known, a time `t_s` (seconds) since the first train of the set.
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


def run_starts(runs, n_trains, train_seconds=15.0, run_start_s=None, run_gap_s=120.0):
    """Start time (s) of train 0 of every run, relative to train 0 of the first run.
    run_start_s: {run: seconds} set by hand (any subset; the first run defaults to 0). Runs without a value start
    run_gap_s after the last train of the previous run (n_trains[prev] - 1 intervals after its start)."""
    run_start_s = run_start_s or {}
    out = {}
    for i, r in enumerate(runs):
        if r in run_start_s:
            out[r] = float(run_start_s[r])
        elif i == 0:
            out[r] = 0.0
        else:
            prev = runs[i - 1]
            out[r] = out[prev] + (n_trains[prev] - 1) * train_seconds + run_gap_s
    return out


def timeline(frames, run_sets, train_seconds=15.0, run_start_s=None, run_gap_s=120.0, run_trains=None):
    """Order and time of every frame within its set: runs in the listed order, trains in order within a run.

    seq       0, 1, 2, ... through the whole set (missing trains leave no gap; use `slot` for that)
    slot      position counting every train of every run, including trains with no frame, from run_trains
              ({run: number of trains}; default: highest train found + 1)
    t_s       seconds since train 0 of the first run: run start (run_starts) + train * train_seconds
    t_run_s   seconds since train 0 of the frame's own run
    run_start_s may be {run: s} for one set, or {set: {run: s}} for several.
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
        start, s = {}, 0
        for r in runs:
            start[r] = s
            s += n_tr[r]
        f["slot"] = [start[r] + t for r, t in zip(f["run"], f["train"])]
        rs = run_start_s or {}
        rs = rs.get(name, {}) if rs and isinstance(next(iter(rs.values())), dict) else rs
        t0 = run_starts(runs, n_tr, train_seconds, rs, run_gap_s)
        f["t_run_s"] = f["train"].astype(float) * train_seconds
        f["t_s"] = f["run"].map(t0) + f["t_run_s"]
        out.append(f)
    return pd.concat(out, ignore_index=True)


def set_metrics(bubbles, frames_tl, um_per_px=3.2, small_um=10.0):
    """bubble_compare.metrics for every frame, with set / seq / slot / t_s / t_run_s attached."""
    rows = []
    for name, f in frames_tl.groupby("set", sort=False):
        b = bubbles[bubbles["set"] == name]
        m = bc.metrics(b, f.assign(t=f["train"].astype(float)), um_per_px=um_per_px, small_um=small_um)
        keep = [c for c in ("frame", "set", "seq", "slot", "t_s", "t_run_s", "run_pos") if c in f]
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
                                 gap_s=g.loc[i + 1, "t_s"] - g.loc[i, "t_s"] if "t_s" in g else np.nan,
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


XLABEL = {"seq": "train (runs in order)", "slot": "train (runs in order)", "t_s": "time since train 0 of the first run [s]"}


def plot_timeline(m, col, ylabel, ax=None, x="seq", colors=None, title=None, annotate=False, logy=False, fits=None):
    """One metric through a whole set: one line through every train of every run, markers in the run's colour,
    dashed lines at run boundaries with the run names on top. x: "seq" (frame order), "slot" (gaps for missing
    trains) or "t_s" (timestamps). fits: decay-fit table from fit_decays (drawn for its metric only)."""
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
    if fits is not None and len(fits):
        _draw_fits(ax, g, fits[fits["metric"] == col], x, colors)
    if x in ("seq", "slot"):
        ax.set_xticks(xs, [str(int(t)) for t in g["train"]], fontsize=8)
    ax.set_xlabel(XLABEL.get(x, x))
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10, pad=16)
    return ax


def plot_timelines(m, specs, x="seq", annotate_first=True, fits=None):
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
                          title=s if i == 0 else None, annotate=annotate_first and i == 0,
                          fits=fits[fits["set"] == s] if fits is not None and len(fits) else None)
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
        ax.set_xlabel({"train": "train within the run", "t_run_s": "time since train 0 of the run [s]"}.get(x_within, x_within))
        ax.set_ylabel(f"{ylabel} ÷ first train")
        ax.set_title(s, fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_size_timeline(bubbles, m, var="r_eq_um", label="equivalent radius [µm]", x="seq", width=None):
    """Size distribution of every frame of each set as one column (x = "seq": one column per train; x = "t_s":
    at its timestamp, `width` seconds wide, default the smallest spacing), with the median and run boundaries."""
    import matplotlib.pyplot as plt
    import bubble_stats as bst
    x = _xcol(m, x)
    sets = list(dict.fromkeys(m["set"]))
    bins = bst.log_bins(bubbles[var], n=30)
    fig, axs = plt.subplots(len(sets), 1, figsize=(max(9, 0.4 * m.groupby("set").size().max() + 3), 3.6 * len(sets)),
                            squeeze=False)
    for ax, s in zip(axs[:, 0], sets):
        g = m[m["set"] == s].sort_values("seq")
        xc = np.arange(len(g), dtype=float) if x == "seq" else g[x].to_numpy(float)
        w = 1.0 if x == "seq" else (width or (np.min(np.diff(xc)) if len(xc) > 1 else 1.0))
        H = np.array([np.histogram(bubbles.loc[bubbles["frame"] == f, var], bins=bins)[0] for f in g["frame"]], float)
        H /= np.maximum(H.sum(axis=1, keepdims=True), 1)
        vmax = H.max() if H.size else 1
        for k in range(len(g)):
            pc = ax.pcolormesh([xc[k] - w / 2, xc[k] + w / 2], bins, H[k][:, None], cmap="Blues", vmin=0, vmax=vmax,
                               shading="flat")
        runs = g["run"].to_numpy()
        med = [bubbles.loc[bubbles["frame"] == f, var].median() for f in g["frame"]]
        for run in dict.fromkeys(runs):
            idx = np.where(runs == run)[0]
            ax.plot(xc[idx], np.asarray(med)[idx], color=C[1], lw=1.5, label="median" if idx[0] == 0 else None)
            ax.text(xc[idx].mean(), 1.01, run, transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
        for i in np.where(runs[1:] != runs[:-1])[0]:
            ax.axvline((xc[i] + xc[i + 1]) / 2, color=INK, lw=1, ls="--")
        ax.set_yscale("log")
        bst._log_ticks(ax, [v for v in (2, 3, 5, 10, 20, 30, 50, 100, 200) if bins[0] <= v <= bins[-1]], axis="y")
        if x in ("seq", "slot"):
            ax.set_xticks(xc, [str(int(t)) for t in g["train"]], fontsize=8)
        ax.set_xlim(xc.min() - w, xc.max() + w)
        ax.set_xlabel(XLABEL.get(x, x)); ax.set_ylabel(label); ax.grid(False)
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(pc, ax=ax, label="fraction of the frame's bubbles")
        ax.set_title(f"{s}: size distribution through the set", fontsize=10, pad=16)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Exponential decay of the bubble count after each scan
# ----------------------------------------------------------------------------
def _decay_ranges(spec):
    """Normalise DECAY_FITS to [(start_run, start_train, end_run, end_train)]. Accepted forms:
    {run: (first, last)} or {run: [(first, last), ...]}     within one run
    [(run, first, last), ...]                               within one run
    [(start_run, start_train, end_run, end_train), ...]     across runs, e.g. ("r562", 4, "r563", 3)
    [(("r562", 4), ("r563", 3)), ...]                       the same, as two (run, train) pairs
    Ranges are inclusive and follow the set's run order."""
    if not spec:
        return []
    out = []
    if isinstance(spec, dict):
        for run, v in spec.items():
            for lo, hi in ([v] if np.ndim(v) == 1 and len(v) == 2 and not isinstance(v[0], (list, tuple)) else v):
                out.append((run, int(lo), run, int(hi)))
        return out
    for r in spec:
        if len(r) == 2:                                   # ((run, train), (run, train))
            (ra, ta), (rb, tb) = r
        elif len(r) == 3:
            ra, ta, tb = r
            rb = ra
        else:
            ra, ta, rb, tb = r
        out.append((ra, int(ta), rb, int(tb)))
    return out


def range_label(ra, ta, rb, tb):
    return f"{ra} t{ta}–{tb}" if ra == rb else f"{ra} t{ta} → {rb} t{tb}"


def _select_range(g_all, ra, ta, rb, tb):
    """Rows of the set from (ra, ta) to (rb, tb), inclusive, in timeline order (may span several runs)."""
    for sname, g in g_all.groupby("set", sort=False):
        a = g[(g["run"] == ra) & (g["train"] == ta)]
        b = g[(g["run"] == rb) & (g["train"] == tb)]
        if len(a) and len(b):
            lo, hi = sorted((int(a["seq"].iloc[0]), int(b["seq"].iloc[0])))
            return g[g["seq"].between(lo, hi)].sort_values("seq"), None
    have = set(zip(g_all["run"], g_all["train"]))
    missing = [f"{r} t{t}" for r, t in ((ra, ta), (rb, tb)) if (r, t) not in have]
    return g_all.iloc[:0], f"no frame for {', '.join(missing)}" if missing else "start and end are in different sets"


def decay_model(t, n0, tau, n_inf=0.0):
    """N(t) = N_inf + (N0 - N_inf) exp(-t / tau), t from the first train of the fit range."""
    return n_inf + (n0 - n_inf) * np.exp(-t / tau)


def fit_decays(m, ranges, col="n_bubbles", offset=False, set_name=None):
    """Fit one exponential decay per range: ranges as in _decay_ranges, e.g. {"r561": (4, 9)} = trains 4..9 of
    r561, or ("r562", 4, "r563", 3) = r562 train 4 to r563 train 3 (inclusive). Time is the timestamp t_s from the
    first train of the range, so a range across runs includes the real pause between them.
    offset=False: N = N0 exp(-t/τ) (2 parameters; robust with 4-6 points).
    offset=True:  N = N∞ + (N0 - N∞) exp(-t/τ) (3 parameters; needs the curve to level off within the range).
    Returns one row per range: N0, τ [s] ± s.e., half-life, N∞, R², number of points, and the rate 1/τ."""
    from scipy.optimize import curve_fit
    rows = []
    g_all = m if set_name is None else m[m["set"] == set_name]
    for ra, ta, rb, tb in _decay_ranges(ranges):
        g, err = _select_range(g_all, ra, ta, rb, tb)
        g = g.dropna(subset=[col])
        row = dict(set=g["set"].iloc[0] if len(g) else set_name, run=ra, start_run=ra, start_train=ta, end_run=rb,
                   end_train=tb, label=range_label(ra, ta, rb, tb), metric=col, n_points=len(g), offset=offset,
                   seq_lo=int(g["seq"].min()) if len(g) else np.nan, seq_hi=int(g["seq"].max()) if len(g) else np.nan,
                   runs_spanned=", ".join(dict.fromkeys(g["run"])) if len(g) else "")
        if err:
            rows.append(dict(row, note=err))
            continue
        need = 4 if offset else 3
        if len(g) < need:
            rows.append(dict(row, note=f"only {len(g)} points (need {need})"))
            continue
        t = (g["t_s"] - g["t_s"].iloc[0]).to_numpy(float)
        y = g[col].to_numpy(float)
        # start values from a log-linear fit
        pos = y > 0
        sl = np.polyfit(t[pos], np.log(y[pos]), 1)[0] if pos.sum() >= 2 else -1 / max(t[-1], 1)
        tau0 = -1 / sl if sl < 0 else max(t[-1], 1.0) * 2
        try:
            if offset:
                p0 = [y[0], tau0, max(y.min() * 0.8, 0)]
                popt, pcov = curve_fit(decay_model, t, y, p0=p0, bounds=([0, 1e-3, 0], [np.inf, 1e6, np.inf]), maxfev=20000)
            else:
                popt, pcov = curve_fit(lambda tt, n0, tau: decay_model(tt, n0, tau), t, y, p0=[y[0], tau0],
                                       bounds=([0, 1e-3], [np.inf, 1e6]), maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
        except (RuntimeError, ValueError) as e:
            rows.append(dict(row, note=f"fit failed: {e}"))
            continue
        pred = decay_model(t, *popt)
        ss_res, ss_tot = ((y - pred) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        tau, tau_se = popt[1], perr[1]
        note = ""
        if tau >= 0.99e6 or tau > 20 * max(t[-1], 1.0):
            note = "no decay in this range (τ ≫ the range)"
        elif not np.isfinite(tau_se) or tau_se > 0.5 * tau:
            note = "τ poorly determined (s.e. > 50 %)"
        rows.append(dict(row, t0_s=g["t_s"].iloc[0], t_end_s=g["t_s"].iloc[-1], N0=popt[0], N0_se=perr[0],
                         tau_s=tau, tau_se=tau_se, half_life_s=tau * np.log(2), rate_per_min=60 / tau,
                         N_inf=popt[2] if offset else 0.0, N_inf_se=perr[2] if offset else np.nan,
                         r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                         rmse=float(np.sqrt(ss_res / len(y))), note=note))
    return pd.DataFrame(rows)


def fit_lines(m, ranges, col="r_mean_um", set_name=None):
    """Straight-line fit y = a + b·t of one metric over each range (same ranges as fit_decays, so DECAY_FITS can be
    reused); t = timestamp t_s from the first train of the range. Returns per range: start value a, slope b per s
    and per min ± s.e., relative change per min (b / a), R², p (slope ≠ 0) and the number of points."""
    from scipy import stats
    rows = []
    g_all = m if set_name is None else m[m["set"] == set_name]
    for ra, ta, rb, tb in _decay_ranges(ranges):
        g, err = _select_range(g_all, ra, ta, rb, tb)
        g = g.dropna(subset=[col])
        row = dict(set=g["set"].iloc[0] if len(g) else set_name, run=ra, start_run=ra, start_train=ta, end_run=rb,
                   end_train=tb, label=range_label(ra, ta, rb, tb), metric=col, n_points=len(g),
                   seq_lo=int(g["seq"].min()) if len(g) else np.nan, seq_hi=int(g["seq"].max()) if len(g) else np.nan)
        if err:
            rows.append(dict(row, note=err))
            continue
        if len(g) < 3:
            rows.append(dict(row, note=f"only {len(g)} points (need 3)"))
            continue
        t = (g["t_s"] - g["t_s"].iloc[0]).to_numpy(float)
        res = stats.linregress(t, g[col].to_numpy(float))
        rows.append(dict(row, t0_s=g["t_s"].iloc[0], t_end_s=g["t_s"].iloc[-1], intercept=res.intercept,
                         intercept_se=res.intercept_stderr, slope_per_s=res.slope, slope_se=res.stderr,
                         slope_per_min=60 * res.slope, slope_per_min_se=60 * res.stderr,
                         rel_per_min=60 * res.slope / res.intercept if res.intercept else np.nan,
                         r2=res.rvalue ** 2, p=res.pvalue, note=""))
    return pd.DataFrame(rows)


def _draw_fits(ax, g, fits, x, colors):
    """Fitted decay curves (fit_decays) or straight lines (fit_lines) on a timeline axes (x = t_s directly;
    x = seq / slot by interpolating time to position)."""
    for f in fits.itertuples():
        linear = hasattr(f, "slope_per_s")
        if not np.isfinite(getattr(f, "slope_per_s" if linear else "tau_s", np.nan)):
            continue
        tt = np.linspace(0, f.t_end_s - f.t0_s, 60)
        yy = f.intercept + f.slope_per_s * tt if linear else decay_model(tt, f.N0, f.tau_s, f.N_inf)
        r = g[g["seq"].between(f.seq_lo, f.seq_hi)].sort_values("t_s")
        if r.empty:
            continue
        xx = tt + f.t0_s if x == "t_s" else np.interp(tt + f.t0_s, r["t_s"], r[x])
        ax.plot(xx, yy, "-", color=colors.get(f.run, INK), lw=2.2, alpha=0.6, zorder=4)
        text = (f"{f.slope_per_min:+.2f}/min" if linear else
                f"τ = {f.tau_s:.0f} s" if not f.note else "no clear decay")
        ax.annotate(text, (xx[len(xx) // 2], yy[len(yy) // 2]), xytext=(4, 8),
                    textcoords="offset points", fontsize=8, color=colors.get(f.run, INK), fontweight="bold")


def plot_decays(m, fits, col="n_bubbles", ylabel="bubble count"):
    """All fitted ranges on one axes against the time since the start of each range: data, fit, τ in the legend.
    Left: linear y; right: log y (a straight line = a pure exponential)."""
    import matplotlib.pyplot as plt
    colors = bc.run_colors(m["run"].unique())
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, logy in zip(axs, (False, True)):
        for f in fits.itertuples():
            if not np.isfinite(getattr(f, "tau_s", np.nan)):
                continue
            g = m[(m["set"] == f.set) & m["seq"].between(f.seq_lo, f.seq_hi)].sort_values("t_s")
            t = g["t_s"] - f.t0_s
            c = colors.get(f.run, INK)
            ax.plot(t, g[col], "o", color=c, ms=7, mec="white", mew=1.2, zorder=3)
            tt = np.linspace(0, t.max(), 80)
            ax.plot(tt, decay_model(tt, f.N0, f.tau_s, f.N_inf), "--" if f.note else "-", color=c, lw=2,
                    label=f"{f.label}: " +
                          (f"τ = {f.tau_s:.0f} ± {f.tau_se:.0f} s" if not f.note else f.note))
        ax.set_xlabel("time since the first train of the fit range [s]")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log"); ax.set_title("log scale (straight line = exponential)", fontsize=10)
        else:
            ax.set_ylim(bottom=0); ax.set_title("Exponential decay after each scan", fontsize=10)
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def compare_taus(fits):
    """Are the decay times of the runs the same? Weighted mean τ, χ² test of equality (uses the fit s.e.),
    and the largest / smallest ratio."""
    from scipy import stats
    if "tau_s" not in fits:
        return None
    f = fits.dropna(subset=["tau_s", "tau_se"])
    f = f[(f["tau_se"] > 0) & (f["note"].fillna("") == "")]          # well-determined fits only
    if len(f) < 2:
        return None
    w = 1 / f["tau_se"] ** 2
    mean = (w * f["tau_s"]).sum() / w.sum()
    chi2 = (w * (f["tau_s"] - mean) ** 2).sum()
    dof = len(f) - 1
    return dict(n=len(f), runs=", ".join(f["run"]), tau_weighted_mean=mean, tau_mean_se=np.sqrt(1 / w.sum()), chi2=chi2, dof=dof,
                p_equal=stats.chi2.sf(chi2, dof), ratio_max_min=f["tau_s"].max() / f["tau_s"].min())
