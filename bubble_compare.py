"""
Compare bubble populations across runs and trains (bubble_runs_analysis.ipynb).

Input: one results folder per run (results/<run>/, written by bubble_inference.ipynb), each with one or more
frames per train. load_runs keeps one frame number per train (FRAME_IDX, e.g. 10) and stacks all runs, with a
`run` column (from the frame name, e.g. "r563_svd_..." -> "r563", else the folder name).

Per (run, train) metrics: everything bubble_stats.frame_metrics gives, plus (see extra_metrics)
    frac_small                  share of bubbles with r < small_um (a fresh population raises it)
    r_p90_um                    90th percentile radius
    log_width                   std of ln r (width of the size distribution, scale free)
    nn_mean_um                  mean distance from a bubble centre to the nearest other centre
    clark_evans                 nearest-neighbour index R (1 random, < 1 clustered, > 1 evenly spaced),
                                Donnelly edge correction, centres inside the frame only
    packing                     total bubble area / frame area (> coverage when bubbles overlap)

Comparisons: step classification between consecutive trains (coalescence / dissolution / new bubbles),
outlier steps (candidate scan-layer trains), trends per run and pooled, variance partition (run vs train),
distribution tests (Kruskal-Wallis, KS, Wasserstein distances), scaled-distribution collapse, coarsening exponent.
`Findings` collects one-line results from every section for the text box at the end of the notebook.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import fields

import numpy as np
import pandas as pd

import bubble_io as bio
import bubble_rcnn as br
import bubble_stats as bst
from bubble_stats import C, INK, INK2

# categorical run colours, fixed order (bubble_stats palette first)
RUN_COLORS = C + ["#7a4fd0", "#d6457a", "#7d5a3c", "#4a4a4a"]
_RUN_RE = re.compile(r"^(r\d+)")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_run_dirs(results_root="results", runs=None):
    """Top-level run folders under results_root that hold results (bubbles.csv or shapes/).
    Folders starting with '_' (e.g. the comparison output) are skipped. runs: only these folder names."""
    out = []
    for d in sorted(os.listdir(results_root)):
        p = os.path.join(results_root, d)
        if d.startswith(("_", ".")) or not os.path.isdir(p):
            continue
        if runs is not None and d not in runs:
            continue
        if os.path.exists(os.path.join(p, "bubbles.csv")) or os.path.isdir(os.path.join(p, "shapes")):
            out.append(p)
    return out


def config_from_run_info(run_dir):
    """The model/measurement config stored in run_info.json by bubble_inference.ipynb."""
    with open(os.path.join(run_dir, "run_info.json")) as f:
        d = dict(json.load(f)["config"])
    for k in ("channels", "anchor_ratios", "scale_jitter"):
        if k in d:
            d[k] = tuple(d[k])
    if "anchor_sizes" in d:
        d["anchor_sizes"] = tuple(tuple(a) for a in d["anchor_sizes"])
    known = {f.name for f in fields(br.RCNNConfig)}
    return br.RCNNConfig(**{k: v for k, v in d.items() if k in known})


def _needs_export(run_dir):
    p = os.path.join(run_dir, "bubbles.csv")
    return (not os.path.exists(p)) or os.path.getsize(p) < 10


def ensure_exported(run_dir, include_unreviewed=True, force=False, verbose=True):
    """Write bubbles.csv / frames.csv from the shape files if they are missing or empty (or force=True).
    Needs the frame .tif copies in shapes/ (present on Maxwell, not in git)."""
    if not (force or _needs_export(run_dir)):
        return False
    cfg = config_from_run_info(run_dir)
    bio.export_results(run_dir, cfg, include_unreviewed=include_unreviewed, verbose=verbose)
    return True


def run_name(frame, run_dir):
    m = _RUN_RE.match(str(frame))
    return m.group(1) if m else os.path.basename(os.path.normpath(run_dir))


def load_runs(run_dirs, frame_idx=10, reviewed_only=False, in_analysis_only=True, train_times=None,
              export_missing=True, include_unreviewed=True, verbose=True):
    """Stack the chosen frame of every train of every run.

    frame_idx: frame number kept in every train (None = all frames).
    train_times: None (t = train number), {train: t} for all runs, or {run: {train: t}}.
    Returns (bubbles, frames); both carry run, train, t, frame and run_dir.
    """
    bl, fl, seen = [], [], {}
    for d in run_dirs:
        if export_missing:
            try:
                if ensure_exported(d, include_unreviewed=include_unreviewed, verbose=verbose) and verbose:
                    print(f"  (exported {d} from its shape files)")
            except (FileNotFoundError, KeyError) as e:
                print(f"skipped {d}: cannot export ({e})")
                continue
        if _needs_export(d):
            print(f"skipped {d}: no bubbles.csv")
            continue
        b, f = bio.load_results(d, reviewed_only=reviewed_only, in_analysis_only=in_analysis_only)
        if frame_idx is not None:
            b, f = b[b["frame_idx"] == frame_idx], f[f["frame_idx"] == frame_idx]
        if f.empty:
            print(f"skipped {d}: no {'reviewed ' if reviewed_only else ''}frame {frame_idx}")
            continue
        # a frame name must be unique across runs (bubbles are matched to frames by name): skip frames already
        # loaded from another folder (e.g. a copy or an older export of the same run) and repeated rows
        dup = f["frame"].isin(list(seen)) | f["frame"].duplicated()
        if dup.any():
            src = sorted({seen[x] for x in f.loc[dup, "frame"] if x in seen})
            print(f"WARNING {d}: skipped {int(dup.sum())} frame(s) already loaded"
                  + (f" from {', '.join(src)}" if src else " (repeated rows in frames.csv)")
                  + "; set RUNS to choose which folder to use")
            f = f[~dup]
            if f.empty:
                continue
        seen.update({x: d for x in f["frame"]})
        f = f.assign(run=[run_name(x, d) for x in f["frame"]], run_dir=d)
        b = b.drop(columns=[c for c in ("run", "run_dir") if c in b]).merge(f[["frame", "run", "run_dir"]],
                                                                              on="frame", how="inner")
        bl.append(b)
        fl.append(f)
    if not fl:
        raise FileNotFoundError("no usable runs found")
    bubbles, frames = pd.concat(bl, ignore_index=True), pd.concat(fl, ignore_index=True)

    def tmap(run, train):
        if not train_times:
            return float(train)
        tt = train_times.get(run, train_times) if isinstance(next(iter(train_times.values())), dict) else train_times
        return float(tt.get(train, np.nan))

    frames["t"] = [tmap(r, t) for r, t in zip(frames["run"], frames["train"])]
    bubbles = bubbles.merge(frames[["frame", "t"]], on="frame", how="left")
    frames = frames.sort_values(["run", "t", "frame"]).reset_index(drop=True)
    if verbose:
        tab = frames.groupby("run").agg(trains=("train", lambda s: ",".join(map(str, sorted(s)))),
                                        n_frames=("frame", "size"))
        tab["bubbles"] = bubbles.groupby("run").size()
        print(f"{frames['run'].nunique()} runs, {len(frames)} frames, {len(bubbles)} bubbles")
        try:
            display(tab)          # noqa: F821 (notebook)
        except NameError:
            print(tab)
    return bubbles, frames


def drop_sparse(bubbles, frames, min_bubbles=30):
    """Drop frames with fewer than min_bubbles bubbles (e.g. trains recorded before the scan reached the field of
    view). Their few bubbles make ratios between trains explode and dominate trends. Returns (bubbles, frames,
    dropped) with dropped = [(run, train, n), ...]."""
    n = bubbles.groupby("frame").size().reindex(frames["frame"]).fillna(0).astype(int).to_numpy()
    low = n < min_bubbles
    dropped = [(r, int(t), int(k)) for r, t, k in zip(frames.loc[low, "run"], frames.loc[low, "train"], n[low])]
    keep = frames.loc[~low, "frame"]
    return (bubbles[bubbles["frame"].isin(keep)].reset_index(drop=True),
            frames[~low].reset_index(drop=True), dropped)


def run_colors(runs):
    runs = sorted(runs)
    if len(runs) <= len(RUN_COLORS):
        return {r: RUN_COLORS[i] for i, r in enumerate(runs)}
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab20")
    return {r: cm(i % 20) for i, r in enumerate(runs)}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def clark_evans(x, y, w, h):
    """Nearest-neighbour index R with Donnelly's edge correction (points in a w x h rectangle)."""
    from scipy.spatial import cKDTree
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    dm = d[:, 1].mean()
    expected = 0.5 * np.sqrt(w * h / n) + (0.0514 + 0.041 / np.sqrt(n)) * 2 * (w + h) / n
    return dm, dm / expected


def extra_metrics(bubbles, frame_row, um_per_px=3.2, small_um=10.0):
    r = bubbles["r_eq_um"].to_numpy()
    H, W = frame_row["image_height_px"], frame_row["image_width_px"]
    inside = (bubbles["x"] >= 0) & (bubbles["x"] <= W) & (bubbles["y"] >= 0) & (bubbles["y"] <= H)
    nn, ce = clark_evans(bubbles.loc[inside, "x"].to_numpy() * um_per_px, bubbles.loc[inside, "y"].to_numpy() * um_per_px,
                         W * um_per_px, H * um_per_px)
    fov = H * W * um_per_px ** 2
    return dict(frac_small=float(np.mean(r < small_um)) if len(r) else np.nan,
                r_p90_um=float(np.percentile(r, 90)) if len(r) else np.nan,
                log_width=float(np.std(np.log(r), ddof=1)) if len(r) > 1 else np.nan,
                nn_mean_um=nn, clark_evans=ce, packing=bubbles["area_um2"].sum() / fov)


def metrics(bubbles, frames, um_per_px=3.2, small_um=10.0, coverage=True):
    """One row per (run, train, frame): bubble_stats.frame_metrics + extra_metrics."""
    rows = []
    for (run, d), f in frames.groupby(["run", "run_dir"], sort=True):
        b = bubbles[(bubbles["run"] == run) & (bubbles["run_dir"] == d)]
        fi = f.set_index("frame")
        try:
            m = bst.frame_metrics(b, f, run_dir=d if coverage else None, um_per_px=um_per_px)
        except FileNotFoundError:
            m = bst.frame_metrics(b, f, run_dir=None, um_per_px=um_per_px)
        ex = [extra_metrics(b[b["frame"] == fr], fi.loc[fr], um_per_px, small_um)
              for fr in m["frame"]]
        m = pd.concat([m.drop(columns=[c for c in ("run",) if c in m]), pd.DataFrame(ex)], axis=1)
        m.insert(0, "run", run)
        rows.append(m)
    return pd.concat(rows, ignore_index=True).sort_values(["run", "t"]).reset_index(drop=True)


def pivot(m, col):
    return m.pivot_table(index="run", columns="train", values=col, aggfunc="mean")


# ----------------------------------------------------------------------------
# Changes between consecutive trains
# ----------------------------------------------------------------------------
def steps(m, tol_n=0.10, tol_v=0.25):
    """Change from one train to the next within each run, with a classification of what dominated:
    * new bubbles:  count up by > tol_n and mean radius down
    * coalescence:  count down by > tol_n, mean radius up, gas volume proxy within ±tol_v
    * growth:       volume up by > tol_v (count not up)
    * dissolution:  volume down by > tol_v (count not up)
    * little change / mixed otherwise.
    The volume proxy (Σ 4/3πr³) is noisy, hence the generous tol_v."""
    rows = []
    for run, g in m.sort_values("t").groupby("run"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            rn, rr = b.n_bubbles / max(a.n_bubbles, 1), b.r_mean_um / a.r_mean_um
            rv = b.volume_um3 / a.volume_um3 if a.volume_um3 > 0 else np.nan
            rmed = b.r_median_um / a.r_median_um
            if rn > 1 + tol_n and rr < 1:
                kind = "new bubbles"
            elif rn < 1 - tol_n and rr > 1 and abs(rv - 1) <= tol_v:
                kind = "coalescence"
            elif rv > 1 + tol_v and rn <= 1 + tol_n:
                kind = "growth"
            elif rv < 1 - tol_v and rn <= 1 + tol_n:
                kind = "dissolution"
            elif abs(rn - 1) <= tol_n and abs(rr - 1) <= tol_n:
                kind = "little change"
            else:
                kind = "mixed"
            rows.append(dict(run=run, train_from=a.train, train_to=b.train, t_from=a.t, t_to=b.t,
                             count_ratio=rn, r_mean_ratio=rr, r_median_ratio=rmed, volume_ratio=rv,
                             coverage_change=b.get("coverage_frac", np.nan) - a.get("coverage_frac", np.nan),
                             frac_small_change=b.frac_small - a.frac_small, kind=kind))
    return pd.DataFrame(rows)


def _robust_z(v):
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(v) or 1.0
    return (v - med) / mad


def outlier_steps(st, z_thresh=3.0):
    """Steps whose change in count, median radius or small-bubble share is far outside the typical step
    (robust z-score over all steps of all runs). Candidates for a new scan layer or another sudden event."""
    if st.empty:
        return st.assign(z=[], flagged=[], what=[], fresh=[])
    z = np.c_[_robust_z(np.log(st["count_ratio"])), _robust_z(np.log(st["r_median_ratio"])),
              _robust_z(st["frac_small_change"])]
    out = st.copy()
    out["z_count"], out["z_rmedian"], out["z_small"] = z.T
    out["z"] = np.nanmax(np.abs(z), axis=1)
    out["flagged"] = out["z"] > z_thresh

    def what(r):
        bits = []
        if abs(r.z_count) > z_thresh:
            bits.append(f"count ×{r.count_ratio:.2f}")
        if abs(r.z_rmedian) > z_thresh:
            bits.append(f"median r ×{r.r_median_ratio:.2f}")
        if abs(r.z_small) > z_thresh:
            bits.append(f"small share {r.frac_small_change * 100:+.0f} pts")
        if r.count_ratio > 1 and r.r_median_ratio < 1 and r.frac_small_change > 0:
            bits.append("→ fresh small bubbles (new layer?)")
        return ", ".join(bits)

    out["what"] = [what(r) if f else "" for r, f in zip(out.itertuples(), out["flagged"])]
    # a fresh population: count up, and smaller bubbles or more small ones; the train after such a jump is a
    # layer candidate. A jump straight back (count down) right after one is the layer ending, not a new layer.
    out["fresh"] = out["flagged"] & (out["count_ratio"] > 1) & ((out["r_median_ratio"] < 1) | (out["frac_small_change"] > 0))
    prev_fresh = out.groupby("run")["fresh"].shift(fill_value=False)
    after = out["flagged"] & ~out["fresh"] & prev_fresh
    out.loc[after, "what"] = out.loc[after, "what"] + " (back to normal after the jump before)"
    return out


def layer_set(out_steps=None, known=None):
    """(run, train) pairs to mark: the train after each fresh-population jump (outlier_steps) and/or known
    layer trains ({run: [trains]})."""
    s = set()
    if out_steps is not None and len(out_steps):
        s |= {(r.run, int(r.train_to)) for r in out_steps.itertuples() if r.fresh}
    if known:
        s |= {(run, int(t)) for run, ts in known.items() for t in ts}
    return s


# ----------------------------------------------------------------------------
# Trends and variance partition
# ----------------------------------------------------------------------------
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, res


def _r2(X, y):
    _, res = _ols(X, y)
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - (res ** 2).sum() / sst if sst > 0 else np.nan


def variance_partition(m, cols):
    """Share of the variation of each metric explained only by the run, only by the train, by both
    (confounded), and the rest (run × train interaction, i.e. runs evolving differently, plus noise)."""
    rows = []
    for c in cols:
        g = m[["run", "train", c]].dropna()
        if g["run"].nunique() < 2 or g["train"].nunique() < 2:
            continue
        y = g[c].to_numpy(float)
        one = np.ones((len(g), 1))
        Dr = pd.get_dummies(g["run"], drop_first=True).to_numpy(float)
        Dt = pd.get_dummies(g["train"], drop_first=True).to_numpy(float)
        r_full, r_run, r_tr = _r2(np.c_[one, Dr, Dt], y), _r2(np.c_[one, Dr], y), _r2(np.c_[one, Dt], y)
        rows.append(dict(metric=c, run_only=r_full - r_tr, train_only=r_full - r_run,
                         shared=r_run + r_tr - r_full, interaction_noise=1 - r_full))
    return pd.DataFrame(rows).set_index("metric")


def trends(m, cols, exclude=None, log_cols=None):
    """Monotonic trend of each metric with t, per run (Spearman ρ) and pooled over runs
    (OLS on run-centred values: common slope, per unit t). Metrics in log_cols are fitted on ln(y),
    and the slope is reported as % change per unit t. exclude: (run, train) pairs left out."""
    from scipy import stats
    log_cols = set(log_cols or [])
    d = m if not exclude else m[[(r, t) not in exclude for r, t in zip(m["run"], m["train"])]]
    rows = []
    for c in cols:
        for run, g in list(d.groupby("run")) + [("all (pooled)", d)]:
            g = g[["run", "t", c]].dropna()
            if len(g) < 3:
                continue
            y = np.log(g[c].clip(lower=1e-12)) if c in log_cols else g[c].astype(float)
            if run == "all (pooled)":
                yc = y - y.groupby(g["run"]).transform("mean")           # run-centred: removes run offsets
                tc = g["t"] - g.groupby("run")["t"].transform("mean")
                rho, p = stats.spearmanr(tc, yc)
                slope = (tc * yc).sum() / (tc ** 2).sum()
                res = yc - slope * tc
                dof = len(g) - g["run"].nunique() - 1
                se = np.sqrt((res ** 2).sum() / max(dof, 1) / (tc ** 2).sum())
                p_slope = 2 * stats.t.sf(abs(slope / se), max(dof, 1)) if se > 0 else np.nan
            else:
                rho, p = stats.spearmanr(g["t"], y)
                lr = stats.linregress(g["t"], y)
                slope, se, p_slope = lr.slope, lr.stderr, lr.pvalue
            rows.append(dict(metric=c, run=run, n=len(g), spearman_rho=rho, spearman_p=p, slope=slope,
                             slope_se=se, slope_p=p_slope,
                             pct_per_t=(np.exp(slope) - 1) * 100 if c in log_cols else np.nan))
    return pd.DataFrame(rows)


def coarsening_exponent(m, col="r32_um"):
    """Power-law fit r ∝ t^α per run (t > 0 only, so meaningful only with real times since the scan).
    Classical references: α = 1/3 Ostwald ripening (LSW), α ≈ 1/2 coalescence-dominated / 2-D foam coarsening."""
    from scipy import stats
    rows = []
    for run, g in m[m["t"] > 0].groupby("run"):
        g = g.dropna(subset=[col])
        if len(g) < 3:
            continue
        lr = stats.linregress(np.log(g["t"]), np.log(g[col]))
        rows.append(dict(run=run, alpha=lr.slope, alpha_se=lr.stderr, r2=lr.rvalue ** 2, n=len(g)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Size distributions
# ----------------------------------------------------------------------------
def kruskal_across(bubbles, group, within, var="r_eq_um"):
    """Kruskal-Wallis test of `var` across the levels of `group`, separately for each level of `within`
    (e.g. group="run", within="train": do runs differ at the same train?). epsilon² is the effect size
    (≈ share of rank variance explained; 0.01 small, 0.06 medium, 0.14 large). With hundreds of bubbles
    per frame tiny differences are 'significant', so read epsilon², not p."""
    from scipy import stats
    rows = []
    for w, g in bubbles.groupby(within):
        samples = [s[var].to_numpy() for _, s in g.groupby(group) if len(s) >= 5]
        if len(samples) < 2:
            continue
        H, p = stats.kruskal(*samples)
        n, k = sum(map(len, samples)), len(samples)
        rows.append(dict(**{within: w}, k=k, n=n, H=H, p=p, epsilon2=max(0.0, (H - k + 1) / (n - k))))
    return pd.DataFrame(rows)


def consecutive_ks(bubbles, m, var="r_eq_um"):
    """Two-sample KS test of the size distribution between consecutive trains of each run."""
    from scipy import stats
    rows = []
    for run, g in m.sort_values("t").groupby("run"):
        fr = g["frame"].tolist()
        for a, b, ta, tb in zip(fr[:-1], fr[1:], g["train"].iloc[:-1], g["train"].iloc[1:]):
            va, vb = bubbles.loc[bubbles["frame"] == a, var], bubbles.loc[bubbles["frame"] == b, var]
            if len(va) < 5 or len(vb) < 5:
                continue
            D, p = stats.ks_2samp(va, vb)
            rows.append(dict(run=run, train_from=ta, train_to=tb, D=D, p=p))
    return pd.DataFrame(rows)


def w1_matrix(bubbles, m, var="r_eq_um"):
    """Wasserstein (earth mover's) distance between the ln-radius distributions of every pair of frames.
    0 = identical; 0.1 ≈ a 10 % shift of all radii."""
    from scipy.stats import wasserstein_distance
    order = m.sort_values(["run", "t"])
    labels = [f"{r.run} t{r.train}" for r in order.itertuples()]
    vals = [np.log(bubbles.loc[bubbles["frame"] == f, var].to_numpy()) for f in order["frame"]]
    n = len(vals)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = wasserstein_distance(vals[i], vals[j]) if len(vals[i]) and len(vals[j]) else np.nan
    return pd.DataFrame(D, index=labels, columns=labels), order.reset_index(drop=True)


def w1_summary(D, order):
    """Mean distance between frames of the same run (different trains), of the same train (different runs),
    and of different runs and trains."""
    runs, trains = order["run"].to_numpy(), order["train"].to_numpy()
    same_run = (runs[:, None] == runs[None]) & (trains[:, None] != trains[None])
    same_train = (trains[:, None] == trains[None]) & (runs[:, None] != runs[None])
    other = (runs[:, None] != runs[None]) & (trains[:, None] != trains[None])
    A = D.to_numpy()
    f = lambda mask: float(np.nanmean(A[mask])) if mask.any() else np.nan
    return dict(same_run_diff_train=f(same_run), same_train_diff_run=f(same_train), different_both=f(other))


def scaled_curves(bubbles, m, bins=None):
    """Histogram density of r/<r> for every frame, and a collapse score: mean coefficient of variation
    across frames in the central bins (0 = perfect collapse)."""
    bins = bins if bins is not None else np.geomspace(0.15, 6, 22)
    centers = np.sqrt(bins[1:] * bins[:-1])
    curves = {}
    for r in m.itertuples():
        v = bubbles.loc[bubbles["frame"] == r.frame, "r_eq_um"].to_numpy()
        if len(v) >= 10:
            curves[(r.run, r.train)] = np.histogram(v / v.mean(), bins=bins, density=True)[0]
    H = np.array(list(curves.values()))
    central = (centers > 0.4) & (centers < 2.5)
    cv = np.nanmean(H[:, central].std(axis=0) / np.maximum(H[:, central].mean(axis=0), 1e-12)) if len(H) > 1 else np.nan
    return centers, curves, cv


# ----------------------------------------------------------------------------
# Pooled size distribution: asymmetric fits
# ----------------------------------------------------------------------------
# All fits are left-truncated at r_min (the smallest radius the model detects reliably): the likelihood of each
# bubble is f(r) / P(R >= r_min), so the missing small bubbles do not bias the fit. loc is fixed at 0.
def _lognorm2(p, x, rmin):
    """Two-component lognormal mixture (e.g. fresh + aged population). p = logit w, mu1, log s1, mu2, log s2."""
    from scipy import stats
    w = 1 / (1 + np.exp(-p[0]))
    d1 = stats.lognorm(np.exp(p[2]), scale=np.exp(p[1]))
    d2 = stats.lognorm(np.exp(p[4]), scale=np.exp(p[3]))
    pdf = w * d1.pdf(x) + (1 - w) * d2.pdf(x)
    norm = w * d1.sf(rmin) + (1 - w) * d2.sf(rmin)
    cdf = (w * (d1.cdf(x) - d1.cdf(rmin)) + (1 - w) * (d2.cdf(x) - d2.cdf(rmin))) / norm
    return pdf / norm, cdf, (w, d1, d2)


FIT_MODELS = {   # name: (scipy distribution name, number of shape parameters)
    "lognormal": ("lognorm", 1),
    "gamma": ("gamma", 1),
    "Weibull": ("weibull_min", 1),
    "generalised gamma": ("gengamma", 2),
    "2 x lognormal": (None, 4),
}


def _fit_one(name, x, w, rmin):
    """Weighted, left-truncated maximum likelihood fit. Returns (theta, nll, pdf(r), cdf(r), dist)."""
    from scipy import optimize, stats
    dname, nsh = FIT_MODELS[name]
    lx = np.log(x)
    if dname is None:
        q1, q2 = np.quantile(lx, [0.3, 0.8])
        th0 = np.array([0.0, q1, np.log(0.35), q2, np.log(0.5)])
        f = lambda th: -np.sum(w * np.log(np.maximum(_lognorm2(th, x, rmin)[0], 1e-300)))
        res = optimize.minimize(f, th0, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-6, fatol=1e-6))
        pdf_fn = lambda r, th=res.x: _lognorm2(th, r, rmin)[0]
        cdf_fn = lambda r, th=res.x: _lognorm2(th, r, rmin)[1]
        return res.x, res.fun, pdf_fn, cdf_fn, _lognorm2(res.x, x[:1], rmin)[2]
    dist = getattr(stats, dname)
    if dname == "lognorm":
        start = [lx.std(), 0, np.exp(lx.mean())]
    else:
        start = list(dist.fit(x, floc=0))
    th0 = np.log(np.r_[start[:nsh], start[-1]])

    def unpack(th):
        e = np.exp(th)
        return tuple(e[:nsh]) + (0.0, e[-1])

    def nll(th):
        a = unpack(th)
        v = -np.sum(w * (dist.logpdf(x, *a) - dist.logsf(rmin, *a)))
        return v if np.isfinite(v) else 1e300

    res = optimize.minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
    a = unpack(res.x)
    d = dist(*a)
    pdf_fn = lambda r: d.pdf(r) / d.sf(rmin)
    cdf_fn = lambda r: (d.cdf(r) - d.cdf(rmin)) / d.sf(rmin)
    return res.x, res.fun, pdf_fn, cdf_fn, d


def _describe(name, d):
    """Readable parameters, mode, mean and skewness of the untruncated fitted distribution."""
    from scipy import optimize
    from scipy.integrate import trapezoid
    if name == "2 x lognormal":
        w, d1, d2 = d
        (a1, b1), (a2, b2) = sorted([(d1, w), (d2, 1 - w)], key=lambda t: t[0].median())
        txt = (f"{b1:.0%} median {a1.median():.1f} µm (σ {a1.args[0]:.2f}) + {b2:.0%} median {a2.median():.1f} µm "
               f"(σ {a2.args[0]:.2f})")
        rr = np.geomspace(0.5, 500, 4000)
        pdf = w * d1.pdf(rr) + (1 - w) * d2.pdf(rr)
        mean = w * d1.mean() + (1 - w) * d2.mean()
        m2 = w * d1.moment(2) + (1 - w) * d2.moment(2)
        m3 = w * d1.moment(3) + (1 - w) * d2.moment(3)
        sd = np.sqrt(m2 - mean ** 2)
        skew = (m3 - 3 * mean * sd ** 2 - mean ** 3) / sd ** 3
        return txt, rr[np.argmax(pdf)], mean, skew
    a = d.args
    if name == "lognormal":
        txt = f"median {d.median():.1f} µm, σ(ln r) {a[0]:.3f}"
    elif name == "gamma":
        txt = f"shape k {a[0]:.3f}, scale {a[2]:.2f} µm"
    elif name == "Weibull":
        txt = f"shape k {a[0]:.3f}, scale λ {a[2]:.2f} µm"
    else:
        txt = f"a {a[0]:.3g}, c {a[1]:.3g}, scale {a[3]:.3g} µm" + (" (≈ lognormal limit)" if a[0] > 100 else "")
    mode = optimize.minimize_scalar(lambda r: -d.pdf(r), bounds=(1e-3, d.ppf(0.9)), method="bounded").x
    mode = 0.0 if mode < 0.01 else mode          # shape < 1: density falls from r = 0
    rr = np.geomspace(max(d.ppf(1e-9), 1e-6), d.ppf(1 - 1e-7), 200001)  # moments numerically (scipy overflows near the lognormal limit)
    pdf = d.pdf(rr)
    pdf /= trapezoid(pdf, rr)
    mean = trapezoid(rr * pdf, rr)
    sd = np.sqrt(trapezoid((rr - mean) ** 2 * pdf, rr))
    return txt, mode, mean, trapezoid((rr - mean) ** 3 * pdf, rr) / sd ** 3


def _wks(x, w, cdf_fn):
    """Weighted Kolmogorov-Smirnov distance between the data and a fitted cdf."""
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    ecdf = np.cumsum(ws) / ws.sum()
    F = cdf_fn(xs)
    return float(np.max(np.maximum(np.abs(ecdf - F), np.abs(ecdf - ws / ws.sum() - F))))


def pooled_sample(bubbles, m, rmin=5.0, exclude=None, weight="bubble", var="r_eq_um"):
    """Radii (>= rmin) of every frame in m, minus the (run, train) pairs in exclude, with likelihood weights:
    weight="bubble" counts every bubble once (busy frames dominate), "frame" gives every frame the same total weight,
    "run" every run. Weights are scaled to sum to the number of bubbles, so AIC stays comparable."""
    keep = m if not exclude else m[[(r, t) not in exclude for r, t in zip(m["run"], m["train"])]]
    b = bubbles[bubbles["frame"].isin(keep["frame"]) & (bubbles[var] >= rmin)]
    x = b[var].to_numpy(float)
    if weight == "bubble":
        w = np.ones(len(x))
    else:
        key = b["frame"] if weight == "frame" else b["run"]
        w = 1.0 / key.map(key.value_counts()).to_numpy(float)
    w *= len(x) / w.sum()
    return b, x, w


def fit_pooled(x, w, rmin=5.0, models=None):
    """Fit every model in FIT_MODELS (or the names in models). Returns (table, fits) with AIC, ΔAIC, KS D,
    readable parameters, mode, mean and skewness; fits[name] = (pdf, cdf) of the truncated fitted distribution."""
    rows, fits = [], {}
    for name in models or FIT_MODELS:
        th, nll, pdf_fn, cdf_fn, d = _fit_one(name, x, w, rmin)
        txt, mode, mean, skew = _describe(name, d)
        k = FIT_MODELS[name][1] + 1
        rows.append(dict(model=name, k=k, nll=nll, AIC=2 * k + 2 * nll, KS_D=_wks(x, w, cdf_fn),
                         params=txt, mode_um=mode, mean_um=mean, skewness=skew))
        fits[name] = (pdf_fn, cdf_fn)
    t = pd.DataFrame(rows)
    t["dAIC"] = t["AIC"] - t["AIC"].min()
    t = t.sort_values("AIC").reset_index(drop=True)
    return t[["model", "k", "dAIC", "KS_D", "params", "mode_um", "mean_um", "skewness", "AIC"]], fits


def bootstrap_runs(b, rmin=5.0, model="lognormal", weight="bubble", n_boot=100, seed=0, var="r_eq_um"):
    """Resample whole runs with replacement and refit: spread of the fitted parameters from run-to-run variation.
    Returns a DataFrame with one row per resample (mode, mean, skewness and the raw parameters)."""
    rng = np.random.default_rng(seed)
    runs = np.array(sorted(b["run"].unique()))
    groups = {r: g for r, g in b.groupby("run")}
    out = []
    for _ in range(n_boot):
        g = pd.concat([groups[r] for r in rng.choice(runs, len(runs))], ignore_index=True)
        x = g[var].to_numpy(float)
        if weight == "bubble":
            w = np.ones(len(x))
        else:
            key = (g["frame"] if weight == "frame" else g["run"]).to_numpy()
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            w = 1.0 / cnt[inv]
        w *= len(x) / w.sum()
        th, nll, _, _, d = _fit_one(model, x, w, rmin)
        _, mode, mean, skew = _describe(model, d)
        out.append(dict(mode_um=mode, mean_um=mean, skewness=skew, **{f"theta{i}": v for i, v in enumerate(th)}))
    return pd.DataFrame(out)


def run_histograms(b, bins, var="r_eq_um", min_n=50, scaled=False):
    """Density histogram per run on shared bins (runs with at least min_n bubbles). scaled: r / <r> of each frame."""
    H = {}
    for run, g in b.groupby("run"):
        if len(g) < min_n:
            continue
        v = g[var] / g.groupby("frame")[var].transform("mean") if scaled else g[var]
        H[run] = np.histogram(v, bins=bins, density=True)[0]
    return pd.DataFrame(H, index=np.sqrt(bins[1:] * bins[:-1])).T


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _mark_layers(ax, piv, layers, color=None):
    import matplotlib.pyplot as plt
    if not layers:
        return
    rows, cols = list(piv.index), list(piv.columns)
    for run, t in layers:
        if run in rows and t in cols:
            ax.add_patch(plt.Rectangle((cols.index(t) - 0.5, rows.index(run) - 0.5), 1, 1, fill=False,
                                       ec=color or INK, lw=2.2))


def plot_heatmaps(m, specs, layers=None, ncols=3, cmap="Blues"):
    """Small multiples: one run × train grid per metric, values written in the cells, each with its own
    colour scale. specs: [(column, title, format)]. Black boxes: layer trains."""
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(specs) / ncols))
    nr = m["run"].nunique()
    nt = m["train"].nunique()
    fig, axs = plt.subplots(nrows, ncols, figsize=(max(4.2, 0.62 * nt + 1.6) * ncols, (0.42 * nr + 1.3) * nrows),
                            squeeze=False)
    for ax, (col, title, fmt) in zip(axs.ravel(), specs):
        P = pivot(m, col)
        A = P.to_numpy(float)
        im = ax.imshow(A, cmap=cmap, aspect="auto")
        lo, hi = np.nanmin(A), np.nanmax(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    dark = (A[i, j] - lo) / ((hi - lo) or 1) > 0.6
                    ax.text(j, i, format(A[i, j], fmt), ha="center", va="center", fontsize=7.5,
                            color="white" if dark else INK)
        ax.set_xticks(range(len(P.columns)), [str(c) for c in P.columns])
        ax.set_yticks(range(len(P.index)), list(P.index))
        ax.set_xlabel("train"); ax.grid(False)
        ax.set_title(title, fontsize=10)
        _mark_layers(ax, P, layers)
    for ax in axs.ravel()[len(specs):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_trajectories(m, specs, time_label, layers=None, ncols=3, relative=False):
    """Small multiples: each metric against t, one line per run (fixed run colours, legend on the first
    panel). Open black rings: layer trains. relative=True divides by each run's first train."""
    import matplotlib.pyplot as plt
    cols = run_colors(m["run"].unique())
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
    for k, (ax, (col, title, _)) in enumerate(zip(axs.ravel(), specs)):
        for run, g in m.sort_values("t").groupby("run"):
            y = g[col] / g[col].iloc[0] if relative else g[col]
            ax.plot(g["t"], y, "-o", color=cols[run], ms=5, mec="white", mew=1, lw=1.6, label=run)
            if layers:
                sel = [(run, t) in layers for t in g["train"]]
                ax.plot(g["t"][sel], y[sel], "o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=4)
        if relative:
            ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_title(title + (" (÷ first train)" if relative else ""), fontsize=10)
        ax.set_xlabel(time_label)
        if m["t"].nunique() <= 12:
            ax.set_xticks(sorted(m["t"].unique()))
        if k == 0:
            ax.legend(fontsize=8, ncol=2)
    for ax in axs.ravel()[len(specs):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_size_heatmaps(bubbles, m, time_label, ncols=2, var="r_eq_um"):
    """bubble_stats.plot_size_heatmap for every run, on shared size bins."""
    import matplotlib.pyplot as plt
    runs = sorted(m["run"].unique())
    bins = bst.log_bins(bubbles[var], n=30)
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.4 * nrows), squeeze=False)
    for ax, run in zip(axs.ravel(), runs):
        g = m[m["run"] == run].sort_values("t")
        bst.plot_size_heatmap(bubbles[bubbles["run"] == run], g, time_label, var=var, bins=bins, ax=ax)
        ax.set_title(f"{run}: size distribution per train", fontsize=10)
    for ax in axs.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_cdfs_by_train(bubbles, m, var="r_eq_um", xlabel="equivalent radius [µm]", ncols=4):
    """Small multiples, one per train: cumulative size distribution of every run."""
    import matplotlib.pyplot as plt
    cols = run_colors(m["run"].unique())
    trains = sorted(m["train"].unique())
    nrows = int(np.ceil(len(trains) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows), sharex=True, sharey=True, squeeze=False)
    lo, hi = bubbles[var].min() * 0.8, bubbles[var].max() * 1.2
    for k, (ax, t) in enumerate(zip(axs.ravel(), trains)):
        for r in m[m["train"] == t].itertuples():
            v = np.sort(bubbles.loc[bubbles["frame"] == r.frame, var].to_numpy())
            ax.step(v, np.arange(1, len(v) + 1) / max(len(v), 1), where="post", color=cols[r.run], lw=1.6, label=r.run)
        ax.set_xscale("log")
        bst._log_ticks(ax, [v for v in (2, 5, 10, 20, 50, 100, 200) if lo <= v <= hi])
        ax.set_title(f"train {t}", fontsize=10)
        if k == 0:
            ax.legend(fontsize=7)
    for ax in axs.ravel()[len(trains):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("fraction ≤ size")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle("Cumulative size distribution: runs compared at each train", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_scaled(centers, curves, layers=None, cv=None):
    """All r/<r> curves on one axes: grey = ordinary trains, orange = layer / outlier trains."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for key, h in curves.items():
        hot = layers and key in layers
        ax.plot(centers, h, "-", color=C[1] if hot else C[0], lw=1.8 if hot else 1.0, alpha=0.95 if hot else 0.35,
                zorder=3 if hot else 2)
    ax.set_xscale("log")
    bst._log_ticks(ax, [0.2, 0.3, 0.5, 1, 2, 3, 5])
    ax.set_xlabel("r / ⟨r⟩"); ax.set_ylabel("probability density")
    title = "Scaled size distributions, every run and train"
    if cv is not None:
        title += f"  (spread {cv:.2f}; 0 = perfect collapse)"
    ax.set_title(title, fontsize=10)
    if layers:
        ax.plot([], [], color=C[1], lw=1.8, label="layer / outlier trains")
        ax.plot([], [], color=C[0], lw=1.0, alpha=0.6, label="other trains")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pooled(b, x, w, fits, table, rmin=5.0, var="r_eq_um", n_show=3):
    """Pooled size distribution with the spread between runs and the best n_show fits.
    Left: linear radius axis; middle: log-log (tail); right: r / <r> per frame (shape only, scale removed)."""
    import matplotlib.pyplot as plt
    bins = np.geomspace(rmin, np.quantile(x, 0.999) * 1.05, 36)
    centers = np.sqrt(bins[1:] * bins[:-1])
    hist = np.histogram(x, bins=bins, weights=w, density=True)[0]
    H = run_histograms(b, bins, var)
    sbins = np.geomspace(0.15, 8, 30)
    S = run_histograms(b, sbins, var, scaled=True)
    sc = np.sqrt(sbins[1:] * sbins[:-1])
    rr = np.geomspace(rmin, bins[-1], 400)
    names = table["model"].tolist()[:n_show]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, logy in zip(axs[:2], (False, True)):
        if len(H) > 1:
            ax.fill_between(centers, H.quantile(0.16), H.quantile(0.84), color=C[0], alpha=0.18, lw=0,
                            label=f"16–84 % of runs (n = {len(H)})")
        ax.plot(centers, hist, "o", color=C[0], ms=4, label=f"pooled ({len(x)} bubbles)")
        for i, n in enumerate(names):
            ax.plot(rr, fits[n][0](rr), "-", color=[INK, C[1], C[2], C[3]][i % 4], lw=1.6 if i == 0 else 1.1,
                    label=f"{n} (ΔAIC {table.loc[table['model'] == n, 'dAIC'].iloc[0]:.0f})")
        ax.set_xlabel("equivalent radius [µm]"); ax.set_ylabel("probability density [1/µm]")
        if logy:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_ylim(max(hist[hist > 0].min() * 0.3, 1e-7), hist.max() * 3)
            ax.set_title("Log-log (tail)", fontsize=10)
        else:
            ax.set_xlim(0, np.quantile(x, 0.99))
            ax.set_title(f"Pooled size distribution (r ≥ {rmin:g} µm)", fontsize=10)
            ax.legend(fontsize=8)
    ax = axs[2]
    if len(S) > 1:
        for run, h in S.iterrows():
            ax.plot(sc, h, "-", color=C[0], lw=0.8, alpha=0.3)
        ax.plot(sc, S.median(), "-", color=INK, lw=2, label="median of runs")
        ax.legend(fontsize=8)
    ax.set_xscale("log"); bst._log_ticks(ax, [0.2, 0.3, 0.5, 1, 2, 3, 5])
    ax.set_xlabel("r / ⟨r⟩ of its frame"); ax.set_ylabel("probability density")
    ax.set_title("Shape only: one line per run", fontsize=10)
    fig.tight_layout()
    return fig


def plot_variance_partition(vp, labels=None):
    import matplotlib.pyplot as plt
    parts = [("run_only", "run", C[0]), ("train_only", "train", C[2]), ("shared", "shared", C[3]),
             ("interaction_noise", "run × train + noise", "#c9c8c4")]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(vp) + 1.4))
    left = np.zeros(len(vp))
    names = [labels.get(i, i) if labels else i for i in vp.index]
    for col, lab, c in parts:
        v = vp[col].clip(lower=0).to_numpy()
        ax.barh(names, v, left=left, color=c, edgecolor="white", linewidth=2, label=lab)
        left += v
    ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.set_xlabel("share of the variation across all frames")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


STEP_COLORS = {"new bubbles": C[1], "coalescence": C[0], "growth": C[2], "dissolution": "#d6457a",
               "little change": "#9a9893", "mixed": C[3]}


def plot_steps(st):
    """Each step between consecutive trains: count ratio vs volume-proxy ratio (log axes), coloured by class."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for kind, g in st.groupby("kind"):
        ax.scatter(g["count_ratio"], g["volume_ratio"], s=55, color=STEP_COLORS.get(kind, INK2), edgecolor="white",
                   lw=1, label=f"{kind} ({len(g)})", zorder=3)
        for r in g.itertuples():
            ax.annotate(f"{r.run} {r.train_from}→{r.train_to}", (r.count_ratio, r.volume_ratio), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4]
    for axis, col in (("x", "count_ratio"), ("y", "volume_ratio")):
        lo, hi = st[col].min(), st[col].max()
        bst._log_ticks(ax, [t for t in ticks if lo / 1.1 <= t <= hi * 1.1] or [1], axis=axis)
    ax.axvline(1, color=INK2, lw=0.8, ls=":"); ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("count ratio (next / previous train)")
    ax.set_ylabel("gas volume proxy ratio")
    ax.set_title("What happens between trains", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_w1(D):
    import matplotlib.pyplot as plt
    n = len(D)
    fig, ax = plt.subplots(figsize=(min(0.32 * n + 2.5, 16), min(0.3 * n + 1.8, 15)))
    im = ax.imshow(D.to_numpy(), cmap="Blues")
    ax.set_xticks(range(n), D.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(n), D.index, fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Wasserstein distance of ln r")
    ax.set_title("How different are the size distributions? (blocks along the diagonal = runs)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_drift_runs(m, native_hw, um_per_px=3.2):
    """Area-weighted centroid path of every run, relative to its first train."""
    import matplotlib.pyplot as plt
    cols = run_colors(m["run"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for run, g in m.sort_values("t").groupby("run"):
        x, y = g["cx_um"] - g["cx_um"].iloc[0], g["cy_um"] - g["cy_um"].iloc[0]
        ax.plot(x, y, "-o", color=cols[run], ms=5, mec="white", mew=1, label=run)
        ax.annotate(f"{g['train'].iloc[-1]}", (x.iloc[-1], y.iloc[-1]), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("Δx [µm]"); ax.set_ylabel("Δy [µm] (image down = +)")
    ax.legend(fontsize=8)
    ax.set_title("Population centroid drift from the first train (label: last train)", fontsize=10)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Findings text box
# ----------------------------------------------------------------------------
class Findings:
    """Collects one-line findings per section; text() renders the box printed at the end of the notebook."""

    def __init__(self):
        self.items = []

    def add(self, section, text):
        self.items.append((section, text))

    def clear(self, section=None):
        self.items = [] if section is None else [i for i in self.items if i[0] != section]

    def text(self, header=""):
        import textwrap
        out, last = [], None
        if header:
            out += [header]
        for sec, t in self.items:
            if sec != last:
                out += ["", f"[{sec}]"] if out else [f"[{sec}]"]
                last = sec
            out += textwrap.wrap(t, 108, initial_indent="- ", subsequent_indent="  ")
        w = max(len(l) for l in out) if out else 0
        bar = "=" * (w + 4)
        return "\n".join([bar] + [f"| {l.ljust(w)} |" for l in out] + [bar])


def fmt_p(p):
    return "p < 1e-6" if p < 1e-6 else f"p = {p:.2g}"
