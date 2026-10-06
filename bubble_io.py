"""
Frame selection, results folders, and measurement of reviewed bubble shapes.

Used by bubble_inference.ipynb (predict -> review in napari -> export) and
bubble_analysis.ipynb (load the exported results).

Results folder layout (one folder per inference run):

    results/<run_name>/
        run_info.json            model, settings and creation time of the run
        shapes/<frame>.tif       the frame (float32, native resolution)
        shapes/<frame>.json      bubble outlines: same format as training annotations, plus
                                 "reviewed", "model", "drafted_at", "frame_path"
        bubbles.csv              one row per bubble (written by export_results)
        frames.csv               one row per frame: train, frame number, counts, reviewed flag

The JSON shapes (napari ellipses / polygons, native coordinates, overlaps allowed) are the
editable source of truth; measurements are recomputed from them by export_results, and
masks can be regenerated at any resolution with shapes_to_masks.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import bubble_rcnn as br
import bubble_seg as bs

_TF_RE = re.compile(r"_tid(\d+)_(\d+)\.[^.]+$")


# ----------------------------------------------------------------------------
# Frame indexing and selection
# ----------------------------------------------------------------------------
def train_frame(path: str):
    """(train, frame number) from a file name like ..._tid3_120.jpg, or (None, None)."""
    m = _TF_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def index_frames(frames) -> dict:
    """{(train, frame): path} for a glob pattern or a list of paths."""
    paths = sorted(glob.glob(frames)) if isinstance(frames, str) else list(frames)
    idx = {}
    for p in paths:
        t, f = train_frame(p)
        if t is not None:
            idx[(t, f)] = p
    return idx


def _frame_numbers(spec, available):
    """Frame spec -> sorted list of frame numbers present in `available`."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("all", "*", "")):
        return sorted(available)
    if isinstance(spec, (int, np.integer)):
        nums = [int(spec)]
    elif isinstance(spec, range):
        nums = list(spec)
    elif isinstance(spec, str):
        nums = []
        for part in spec.split(","):
            part = part.strip()
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
            if m:
                nums += list(range(int(m.group(1)), int(m.group(2)) + 1))
            elif part.isdigit():
                nums.append(int(part))
            elif part:
                raise ValueError(f"cannot read frame spec {part!r} (use 10, '10-20', '10,15,20' or 'all')")
    else:
        nums = [int(x) for x in spec]
    return [n for n in sorted(set(nums)) if n in available]


def select_frames(selection: Iterable, index: dict, verbose=True) -> list:
    """Resolve a selection list to frame paths (sorted by train, frame; duplicates removed).

    Each item can be:
        (0, 10)                 train 0, frame 10
        (0, [10, 20, 30])       several frames of train 0
        (0, range(10, 20))      a range (end exclusive, like Python)
        (0, "10-20")            a range (end inclusive), also "10-20,30,40-45"
        (3, "all") or (3, None) every frame of train 3
        "train0frame010", "tid0_010"   single frames as strings
        "train3", "tid3"        a whole train
        "all"                   everything
    """
    by_train = {}
    for (t, f) in index:
        by_train.setdefault(t, set()).add(f)
    keys = []
    for item in selection:
        if isinstance(item, str):
            s = item.strip()
            if s.lower() == "all":
                keys += sorted(index)
                continue
            m = re.fullmatch(r"(?:train|tid)\s*(\d+)\s*[_,\s]*(?:frame|f)?\s*(\d+)", s, re.I)
            if m:
                item = (int(m.group(1)), int(m.group(2)))
            else:
                m = re.fullmatch(r"(?:train|tid)\s*(\d+)", s, re.I)
                if not m:
                    raise ValueError(f"cannot read selection {item!r}")
                item = (int(m.group(1)), None)
        t, spec = item
        t = int(t)
        if t not in by_train:
            print(f"train {t} not found")
            continue
        nums = _frame_numbers(spec, by_train[t])
        if not nums:
            print(f"no frames found for train {t}, frames {spec!r}")
        keys += [(t, n) for n in nums]
    keys = sorted(set(keys))
    paths = [index[k] for k in keys]
    if verbose:
        per_train = {}
        for t, _ in keys:
            per_train[t] = per_train.get(t, 0) + 1
        print(f"{len(paths)} frames selected: " + ", ".join(f"train {t}: {n}" for t, n in sorted(per_train.items())))
    return paths


def preview_frames(paths, max_n=8):
    """Thumbnails on a fixed grey scale (over/under-exposure stays visible)."""
    import matplotlib.pyplot as plt
    paths = list(paths)
    if not paths:
        return
    if len(paths) > max_n:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_n).astype(int)]
    fig, axs = plt.subplots(1, len(paths), figsize=(4.5 * len(paths), 3.2), squeeze=False)
    for ax, f in zip(axs[0], paths):
        im = bs.load_image(f)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        clipped = 100 * np.mean((im < 0.01) | (im > 0.99))
        ax.set_title(f"{os.path.basename(f)}\nmean {im.mean():.2f}, clipped {clipped:.1f}%", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Results folder
# ----------------------------------------------------------------------------
def shapes_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "shapes")


def _json_for(run_dir, frame_path):
    return os.path.join(shapes_dir(run_dir), os.path.splitext(os.path.basename(frame_path))[0] + ".json")


def write_run_info(run_dir: str, model_path: str, cfg: br.RCNNConfig, extra: Optional[dict] = None):
    os.makedirs(run_dir, exist_ok=True)
    p = os.path.join(run_dir, "run_info.json")
    info = {}
    if os.path.exists(p):
        with open(p) as f:
            info = json.load(f)
    info.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    info.update(model_path=os.path.abspath(model_path), updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=asdict(cfg), **(extra or {}))
    with open(p, "w") as f:
        json.dump(info, f, indent=2, default=str)


def candidates_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "candidates")


def draft_frames(model, cfg: br.RCNNConfig, frame_paths, run_dir: str, model_path: str = "",
                 kind: str = "polygon", overwrite_unreviewed: bool = False, show_every: int = 0,
                 candidate_thresh: Optional[float] = None):
    """Predict each frame and write <frame>.tif + <frame>.json into run_dir/shapes/.

    Existing JSONs are never touched unless overwrite_unreviewed=True AND they are not reviewed.
    kind: 'polygon' keeps the predicted outline (best for non-spherical bubbles; edge bubbles are
    still measured edge-aware at export), 'ellipse' gives shapes that are quicker to adjust.
    candidate_thresh: also save the model's low-score detections (score in [candidate_thresh,
    cfg.score_thresh)) to run_dir/candidates/<frame>.json, used by bubble_track to bridge frames in which a
    bubble was missed. Frames that already have a draft but no candidates are predicted again for the
    candidates only (the draft is left as it is).
    Returns a DataFrame with one row per frame (status: drafted / candidates / skipped).
    """
    import tifffile
    d = shapes_dir(run_dir)
    os.makedirs(d, exist_ok=True)
    if candidate_thresh is not None:
        os.makedirs(candidates_dir(run_dir), exist_ok=True)
    rows, t0 = [], time.time()
    for k, f in enumerate(frame_paths):
        js = _json_for(run_dir, f)
        name = os.path.splitext(os.path.basename(f))[0]
        cj = os.path.join(candidates_dir(run_dir), name + ".json")
        need_shapes, n_old = True, 0
        if os.path.exists(js):
            with open(js) as fh:
                old = json.load(fh)
            need_shapes = not old.get("reviewed") and overwrite_unreviewed
            n_old = len(old.get("bubbles", []))
            reason = "skipped (exists" + (", reviewed)" if old.get("reviewed") else ")")
        need_cands = candidate_thresh is not None and (need_shapes or not os.path.exists(cj))
        if not need_shapes and not need_cands:
            rows.append(dict(frame=name, status=reason, n_bubbles=n_old))
            continue
        img = bs.load_image(f)
        tif = os.path.join(d, name + ".tif")
        if not os.path.exists(tif):
            tifffile.imwrite(tif, img)
        x = br.prepare_image(img, cfg)
        lo = min(cfg.score_thresh, candidate_thresh) if need_cands else cfg.score_thresh
        pred = br.predict(model, x, cfg, score_thresh=lo)    # same high-score result as with cfg.score_thresh
        if need_shapes:
            shapes = br.masks_to_shapes(pred, cfg, kind=kind, min_score=cfg.score_thresh)
            with open(js, "w") as fh:
                json.dump(dict(image=name + ".tif", bubbles=shapes, rois=[], reviewed=False,
                               model=os.path.basename(model_path), drafted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                               frame_path=os.path.abspath(f), score_thresh=cfg.score_thresh), fh)
        if need_cands:
            cands = br.masks_to_shapes(pred, cfg, kind="polygon", min_score=candidate_thresh,
                                       max_score=cfg.score_thresh, with_score=True)
            with open(cj, "w") as fh:
                json.dump(dict(image=name + ".tif", bubbles=cands, score_range=[candidate_thresh, cfg.score_thresh],
                               model=os.path.basename(model_path), drafted_at=time.strftime("%Y-%m-%d %H:%M:%S")), fh)
        n = len(shapes) if need_shapes else n_old
        rows.append(dict(frame=name, status="drafted" if need_shapes else "candidates", n_bubbles=n,
                         n_candidates=len(cands) if need_cands else np.nan))
        if show_every and (k % show_every == 0 or k == len(frame_paths) - 1):
            extra = f", {len(cands)} low-score" if need_cands else ""
            print(f"{k + 1}/{len(frame_paths)}  {name}: {n} bubbles{extra}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def list_results(run_dir: str) -> pd.DataFrame:
    """One row per frame JSON in the run: train, frame, n_bubbles, reviewed, path."""
    rows = []
    for js in sorted(glob.glob(os.path.join(shapes_dir(run_dir), "*.json"))):
        with open(js) as f:
            a = json.load(f)
        t, fr = train_frame(a.get("frame_path", js))
        if t is None:
            t, fr = train_frame(js)
        rows.append(dict(frame=os.path.splitext(os.path.basename(js))[0], train=t, frame_idx=fr,
                         n_bubbles=len(a.get("bubbles", [])), reviewed=bool(a.get("reviewed", False)),
                         model=a.get("model", ""), json_path=js))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Measuring shapes (the reviewed result)
# ----------------------------------------------------------------------------
def _ellipse_from_shape(shape):
    """napari ellipse (4 bbox corners, row/col) -> centre (x, y), semi-axes (a >= b), orientation_deg."""
    d = np.asarray(shape["data"], float)
    c = d.mean(axis=0)
    e1, e2 = (d[1] - d[0]) / 2.0, (d[3] - d[0]) / 2.0
    l1, l2 = np.hypot(*e1), np.hypot(*e2)
    major = e1 if l1 >= l2 else e2
    a, b = max(l1, l2), min(l1, l2)
    orient = np.degrees(np.arctan2(major[1], major[0]))      # angle from the row axis (regionprops convention)
    orient = (orient + 90) % 180 - 90
    return float(c[1]), float(c[0]), float(a), float(b), float(orient)


def measure_shapes(shapes: list, native_hw, cfg: br.RCNNConfig, frame: str = "") -> pd.DataFrame:
    """Measure annotation shapes with the same rules as model predictions.

    Polygons: rasterised at cfg.model_scale and measured like a predicted mask (edge-aware fit for
    bubbles cut by the image edge). Ellipses: their exact parameters are used (a drawn ellipse may
    extend beyond the image; its centre and size are then still exact).
    """
    s = cfg.model_scale
    Hn, Wn = native_hw
    hw = (int(round(Hn * s)), int(round(Wn * s)))
    rows = []
    for i, sh in enumerate(shapes):
        m = br.rasterize(br.shape_outline(sh), hw, s)
        row = br.measure_mask(m, cfg)
        if row is None:            # entirely outside the image
            continue
        if sh["type"] == "ellipse":
            x, y, a, b, orient = _ellipse_from_shape(sh)
            outline = br.shape_outline(sh)                        # (row, col), exact ellipse
            inside = ((outline[:, 1] >= -0.5) & (outline[:, 1] <= Wn - 0.5)
                      & (outline[:, 0] >= -0.5) & (outline[:, 0] <= Hn - 0.5))
            row.update(x=x, y=y, area_px2=np.pi * a * b, r_eq=np.sqrt(a * b), major_axis=2 * a,
                       minor_axis=2 * b, orientation_deg=orient if a > b * 1.0001 else np.nan,
                       eccentricity=np.sqrt(max(0.0, 1 - (b / a) ** 2)), solidity=1.0,
                       edge_truncated=bool(inside.mean() < 1), fit_kind="drawn_ellipse",
                       arc_deg=360.0 * float(inside.mean()), outline_visible_frac=float(inside.mean()),
                       fit_reliable=True)
        rows.append(dict(frame=frame, bubble_id=i, shape_type=sh["type"], **row))
    return br.finish_measurements(pd.DataFrame(rows), (Hn, Wn), cfg)


def export_results(run_dir: str, cfg: br.RCNNConfig, include_unreviewed: bool = False, verbose=True):
    """Measure every frame of the run and write bubbles.csv + frames.csv. Returns (bubbles, frames)."""
    lst = list_results(run_dir)
    if lst.empty:
        raise FileNotFoundError(f"no frames in {shapes_dir(run_dir)}")
    use = lst if include_unreviewed else lst[lst["reviewed"]]
    if verbose and len(use) < len(lst):
        print(f"{len(lst) - len(use)} unreviewed frame(s) left out (include_unreviewed=False)")
    tables, frows = [], []
    for _, r in use.iterrows():
        with open(r.json_path) as f:
            a = json.load(f)
        img_path = os.path.join(os.path.dirname(r.json_path), a["image"])
        hw = bs.load_image(img_path).shape
        df = measure_shapes(a.get("bubbles", []), hw, cfg, frame=r.frame)
        if len(df):
            df.insert(1, "train", r.train)
            df.insert(2, "frame_idx", r.frame_idx)
            df.insert(3, "reviewed", r.reviewed)
            tables.append(df)
        n_in = int(df["in_analysis"].sum()) if len(df) else 0
        frows.append(dict(frame=r.frame, train=r.train, frame_idx=r.frame_idx, reviewed=r.reviewed,
                          n_bubbles=len(df), n_in_analysis=n_in, image_height_px=hw[0], image_width_px=hw[1],
                          model=r.model))
    bubbles = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    frames = pd.DataFrame(frows)
    bubbles.to_csv(os.path.join(run_dir, "bubbles.csv"), index=False)
    frames.to_csv(os.path.join(run_dir, "frames.csv"), index=False)
    with open(os.path.join(run_dir, "export_settings.json"), "w") as f:
        json.dump(dict(exported=time.strftime("%Y-%m-%d %H:%M:%S"), include_unreviewed=include_unreviewed,
                       config=asdict(cfg)), f, indent=2, default=str)
    if verbose:
        print(f"exported {len(frames)} frames, {len(bubbles)} bubbles -> {run_dir}/bubbles.csv, frames.csv")
    return bubbles, frames


def load_results(run_dir: str, reviewed_only: bool = True, in_analysis_only: bool = False):
    """Load bubbles.csv and frames.csv of a run (written by export_results)."""
    bubbles = pd.read_csv(os.path.join(run_dir, "bubbles.csv"))
    frames = pd.read_csv(os.path.join(run_dir, "frames.csv"))
    if reviewed_only and len(bubbles):
        bubbles = bubbles[bubbles["reviewed"]]
        frames = frames[frames["reviewed"]]
    if in_analysis_only and len(bubbles):
        bubbles = bubbles[bubbles["in_analysis"]]
    return bubbles.reset_index(drop=True), frames.reset_index(drop=True)


def load_frame(run_dir: str, frame: str):
    """(image, shapes) of one frame of a run."""
    js = os.path.join(shapes_dir(run_dir), frame + ".json")
    with open(js) as f:
        a = json.load(f)
    return bs.load_image(os.path.join(shapes_dir(run_dir), a["image"])), a.get("bubbles", [])


def shapes_to_masks(shapes: list, native_hw, scale: float = 1.0) -> np.ndarray:
    """One boolean mask per bubble, shape (n, H*scale, W*scale). Masks may overlap."""
    hw = (int(round(native_hw[0] * scale)), int(round(native_hw[1] * scale)))
    if not shapes:
        return np.zeros((0,) + hw, bool)
    return np.array([br.rasterize(br.shape_outline(s), hw, scale) for s in shapes])


def plot_shapes(img, shapes, ax=None, title=None, color=None, view_pad=0):
    """Overlay annotation shapes (native coords) on an image."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(img, cmap="gray")
    cmap = plt.get_cmap("tab20")
    for i, sh in enumerate(shapes):
        o = br.shape_outline(sh)
        o = np.vstack([o, o[:1]])
        ax.plot(o[:, 1], o[:, 0], lw=0.8, color=color or cmap(i % 20))
    ax.set_xlim(-0.5 - view_pad, img.shape[1] - 0.5 + view_pad)
    ax.set_ylim(img.shape[0] - 0.5 + view_pad, -0.5 - view_pad)
    ax.set_title(title or f"{len(shapes)} bubbles")
    ax.axis("off")
    return ax


def _reviewed_frames(results_root: str) -> pd.DataFrame:
    """All reviewed frame JSONs under results_root (any run or train folder; tracking output excluded)."""
    rows = []
    for js in sorted(glob.glob(os.path.join(results_root, "**", "shapes", "*.json"), recursive=True)):
        if os.sep + "tracked" + os.sep in js:
            continue
        with open(js) as f:
            a = json.load(f)
        if a.get("reviewed") and a.get("bubbles"):
            rows.append(dict(frame=os.path.splitext(os.path.basename(js))[0], json_path=js,
                             mtime=os.path.getmtime(js), n_bubbles=len(a["bubbles"])))
    return pd.DataFrame(rows, columns=["frame", "json_path", "mtime", "n_bubbles"])


def sync_reviewed_to_training(results_root: str, annotation_dir: str, run_dir: Optional[str] = None,
                              frames: Optional[list] = None, verbose=True) -> pd.DataFrame:
    """Copy every reviewed frame (tif + json) from the results folders into the training annotations.

    The whole frame counts as fully annotated (no ROI). Run it before training; re-running is safe:
    * new reviewed frames are copied;
    * frames copied earlier are updated when you have corrected them again in the results folder since;
    * annotations you made in the training folder yourself, or edited there after copying, are never overwritten;
    * a frame reviewed in several results folders: the most recently saved version is used.
    run_dir: only this results folder (default: everything under results_root). frames: only these frame names.
    Returns one row per reviewed frame with the action taken.
    """
    import shutil
    src = _reviewed_frames(run_dir if run_dir is not None else results_root)
    if frames is not None:
        src = src[src["frame"].isin(frames)]
    os.makedirs(annotation_dir, exist_ok=True)
    rows = []
    for frame, g in src.sort_values("mtime").groupby("frame", sort=False):
        r = g.iloc[-1]                                   # newest version of this frame
        dst_js = os.path.join(annotation_dir, frame + ".json")
        action = "copied"
        if os.path.exists(dst_js):
            with open(dst_js) as f:
                old = json.load(f)
            imp = old.get("imported_from")
            if imp is None:
                action = "skipped (own training annotation)"
            elif os.path.getmtime(dst_js) > old.get("imported_at", 0) + 5:
                action = "skipped (edited in the training folder)"
            elif os.path.abspath(r.json_path) == imp.get("path") and r.mtime <= imp.get("mtime", 0) + 1e-3:
                action = "up to date"
            else:
                action = "updated"
        if action in ("copied", "updated"):
            with open(r.json_path) as f:
                a = json.load(f)
            src_img = os.path.join(os.path.dirname(r.json_path), a["image"])
            if not os.path.exists(src_img):
                rows.append(dict(frame=frame, action="skipped (frame image missing)", source=r.json_path,
                                 n_bubbles=r.n_bubbles))
                continue
            dst_img = os.path.join(annotation_dir, a["image"])
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            now = time.time()
            with open(dst_js, "w") as f:                 # no ROI = whole frame is complete
                json.dump(dict(image=a["image"], bubbles=a["bubbles"], rois=[], frame_path=a.get("frame_path", ""),
                               imported_from=dict(path=os.path.abspath(r.json_path), mtime=r.mtime),
                               imported_at=now), f)
            os.utime(dst_js, (now, now))
        rows.append(dict(frame=frame, action=action, source=r.json_path, n_bubbles=r.n_bubbles,
                         n_versions=len(g)))
    out = pd.DataFrame(rows, columns=["frame", "action", "source", "n_bubbles", "n_versions"])
    if verbose:
        counts = out["action"].value_counts().to_dict() if len(out) else {}
        print(f"{len(out)} reviewed frame(s) in {run_dir or results_root}: {counts}")
    return out


def copy_to_training(run_dir: str, annotation_dir: str, frames: Optional[list] = None) -> list:
    """Copy (or update) the reviewed frames of one results folder in the training annotations
    (see sync_reviewed_to_training). frames: only these frame names."""
    out = sync_reviewed_to_training(run_dir, annotation_dir, run_dir=run_dir, frames=frames)
    return out.loc[out["action"].isin(["copied", "updated"]), "frame"].tolist()
