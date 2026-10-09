"""
Frame selection, results folders, and measurement of reviewed bubble shapes.

Used by bubble_training.ipynb (picking frames to annotate), bubble_inference.ipynb (predict -> review in napari
-> export), bubble_train_analysis.ipynb and bubble_analysis.ipynb (load the exported results).

Frames are identified by (run, train, frame), e.g. ('r571', 3, 10) for raw/jpg_r571_svd_normalised_tr050/
r571_svd_normalised_tr050_tid3_010.jpg, so frames of several runs can be selected and analysed together
(see select_frames).

Results folder layout (one folder per inference run; it may hold frames of several runs):

    results/<run_name>/
        run_info.json            model, settings and creation time of the run
        shapes/<frame>.tif       the frame (float32, native resolution)
        shapes/<frame>.json      bubble outlines: same format as training annotations (circles / ellipses), plus
                                 "reviewed", "model", "drafted_at", "frame_path"
        bubbles.csv              one row per bubble (written by export_results)
        frames.csv               one row per frame: run, train, frame number, counts, reviewed flag

The JSON shapes (napari circles / ellipses, native coordinates, overlaps allowed) are the editable source of
truth; measurements are recomputed from them by export_results, and masks can be regenerated at any
resolution with shapes_to_masks.
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

_TF_RE = re.compile(r"_tid(\d+)_(\d+)(?:\.[A-Za-z0-9]+)?$")
_RUN_RE = re.compile(r"r(\d+)", re.I)
_ALL = ("all", "*", "")
IMAGE_EXT = ("jpg", "jpeg", "png", "tif", "tiff")


# ----------------------------------------------------------------------------
# Frame indexing and selection
# ----------------------------------------------------------------------------
def run_id(name) -> str:
    """Short run name: 'r563' for 'r563_svd_normalised_tr050_tid0_010.jpg' (any path), 'r563', '563' or 563."""
    if isinstance(name, (int, np.integer)):
        return f"r{int(name):03d}"
    s = os.path.basename(str(name).strip())
    m = re.fullmatch(r"r?(\d+)", s, re.I) or _RUN_RE.match(s)
    if m:
        return f"r{int(m.group(1)):03d}"
    return s.split("_tid")[0]


def train_frame(path: str):
    """(train, frame number) from a file name like ..._tid3_120.jpg (or ..._tid3_120), or (None, None)."""
    m = _TF_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def frame_key(path: str):
    """(run, train, frame) of a frame file name, e.g. ('r563', 0, 10); None if it is not a frame name."""
    t, f = train_frame(path)
    return None if t is None else (run_id(path), t, f)


def _is_all(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in _ALL)


def find_frames(source="raw", runs=None) -> list:
    """Paths of the clean frames (never the _annotated copies with scale bar / time stamp).

    source: the raw folder holding the jpg_<run>/ folders (default "raw": every run in it), one jpg_<run>
            folder, a glob pattern such as "raw/jpg_r563_svd_normalised_tr050/*.jpg", or a list of these.
    runs:   only these runs, e.g. ["r571", "r572"] (default: all).
    """
    sources = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    paths = []
    for src in map(str, sources):
        if any(ch in src for ch in "*?["):
            cand = glob.glob(src)
        elif os.path.isdir(src):
            subs = sorted(d for d in glob.glob(os.path.join(src, "jpg_*")) if os.path.isdir(d))
            cand = [p for d in (subs or [src]) for e in IMAGE_EXT for p in glob.glob(os.path.join(d, "*." + e))]
        elif os.path.isfile(src):
            cand = [src]
        else:
            print(f"not found: {src}")
            cand = []
        paths += [p for p in cand if not os.path.basename(os.path.dirname(os.path.abspath(p))).endswith("_annotated")]
    if runs is not None:
        want = {run_id(r) for r in ([runs] if isinstance(runs, (str, int, np.integer)) else runs)}
        paths = [p for p in paths if run_id(p) in want]
    return sorted(set(paths))


def index_frames(source="raw", runs=None) -> dict:
    """{(run, train, frame): path} of every frame found (see find_frames for source / runs)."""
    idx, dup = {}, []
    for p in find_frames(source, runs):
        k = frame_key(p)
        if k is None:
            continue
        if k in idx and os.path.abspath(idx[k]) != os.path.abspath(p):
            dup.append((idx[k], p))
            continue
        idx[k] = p
    if dup:
        a, b = dup[0]
        raise ValueError(f"{len(dup)} frame(s) found twice for the same run / train / frame, e.g.\n  {a}\n  {b}\n"
                         "Restrict `runs`, or pass only the folder you want as source "
                         "(e.g. 'raw/jpg_r563_svd_normalised_tr050').")
    return dict(sorted(idx.items()))


def describe_index(index: dict):
    """Print the runs, their trains and the number of frames in an index."""
    tree = {}
    for r, t, _ in index:
        tree.setdefault(r, {}).setdefault(t, 0)
        tree[r][t] += 1
    print(f"{len(index)} frames in {len(tree)} run(s)")
    for r, d in sorted(tree.items()):
        print(f"  {r}: trains {sorted(d)} ({sum(d.values())} frames)")


def _numbers(spec, available, what, missing=None):
    """'all' / None / int / list / range / '10-20,30' -> sorted list of the numbers present in `available`
    (requested numbers that are not available are added to `missing`)."""
    if _is_all(spec):
        return sorted(available)
    if isinstance(spec, (int, np.integer)):
        nums = [int(spec)]
    elif isinstance(spec, (str, range)):
        nums = _expand(spec, what)
    else:
        nums = []
        for v in spec:
            if isinstance(v, (str, range)):
                nums += [] if _is_all(v) else _expand(v, what)
            else:
                nums.append(int(v))
    nums = sorted(set(nums))
    if missing is not None:
        missing += [n for n in nums if n not in available]
    return [n for n in nums if n in available]


def _expand(spec, what):
    """'10-20,30' or range -> list of ints (no availability check)."""
    if isinstance(spec, range):
        return list(spec)
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            out += list(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            out.append(int(part))
        elif part:
            raise ValueError(f"cannot read {what} spec {part!r} (use 10, '10-20', '10,15,20' or 'all')")
    return out


def _ranges(nums):
    """[1, 2, 3, 7] -> '1-3, 7'"""
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def _run_list(spec, available):
    if _is_all(spec):
        return sorted(available)
    items = [spec] if isinstance(spec, (str, int, np.integer)) else list(spec)
    out = []
    for r in items:
        rid = run_id(r)
        if rid in available:
            out.append(rid)
        else:
            print(f"run {r!r} not found (indexed runs: {', '.join(sorted(available)) or 'none'})")
    return out


def _parse_selection_string(s):
    """'r571' / 'r571_tid3' / 'r571_tid3_010' / 'tid3_010' / 'train3frame010' / 'tid3' / 'all' -> (run, train, frame)."""
    s = s.strip()
    if s.lower() in _ALL:
        return ("all", "all", "all")
    run = re.match(r"r(\d+)(?=$|[_\s,])", s, re.I)
    tf = re.search(r"(?:^|[_\s,])(?:train|tid)\s*(\d+)(?:[_\s,]*(?:frame|f)?\s*(\d+))?\s*$", s, re.I)
    has_train = re.search(r"(?:^|[_\s,])(?:train|tid)\s*\d", s, re.I)
    if (tf is None and (has_train or not (run and re.fullmatch(r"r\d+(?:_[a-z]\w*)?", s, re.I)))) or \
            (tf is not None and not run and not re.match(r"(?:train|tid)", s, re.I)):
        raise ValueError(f"cannot read selection {s!r} (e.g. 'r571', 'r571_tid3', 'r571_tid3_010', 'tid3_010', 'all'; "
                         "runs as 'r571'; for several frames use a tuple, e.g. ('r571', 3, '10-20'))")
    return (run_id(run.group(0)) if run else "all",
            int(tf.group(1)) if tf else "all",
            int(tf.group(2)) if tf and tf.group(2) is not None else "all")


def select_frames(selection: Iterable, index: dict, verbose=True) -> list:
    """Resolve a selection to frame paths (sorted by run, train, frame; duplicates removed).

    `index` comes from index_frames. Each item of `selection` is (run, train, frames) or a string:
        ("r571", 3, 10)                 run r571, train 3, frame 10
        ("r571", 3, [10, 20, 30])       several frames
        ("r571", 3, "10-20")            a range (end inclusive), also "10-20,30,40-45"; range(10, 20) is end exclusive
        ("r571", 3, "all")              every frame of train 3
        ("r571", "all", 10)             frame 10 of every train of r571
        (["r571", "r572"], "all", 10)   frame 10 of every train of both runs
        ("all", [0, 2], "all")          trains 0 and 2 of every indexed run
        ("r571", "0-3", [0, 100])       trains 0-3, frames 0 and 100
        (3, 10)                         train 3, frame 10 of every indexed run (no run given)
        "r571", "r571_tid3", "r571_tid3_010"   a whole run / train / one frame
        "tid3_010", "train3frame010", "tid3"   the same in every indexed run
        "all"                           everything
    Runs can be written "r571", "571" or 571.
    """
    tree = {}
    for (r, t, f) in index:
        tree.setdefault(r, {}).setdefault(t, set()).add(f)
    if isinstance(selection, str) or (isinstance(selection, tuple)
                                       and not all(isinstance(x, (tuple, list)) for x in selection)):
        selection = [selection]                  # a single item rather than a list / tuple of items
    keys = []
    for item in selection:
        if isinstance(item, str):
            item = _parse_selection_string(item)
        item = tuple(item)
        if len(item) == 2:
            item = ("all",) + item
        if len(item) != 3:
            raise ValueError(f"cannot read selection {item!r}: use (run, train, frames) or (train, frames)")
        rspec, tspec, fspec = item
        n0, misses = len(keys), []
        for r in _run_list(rspec, tree):
            no_train = []
            for t in _numbers(tspec, tree[r], "train", no_train):
                no_frame = []
                keys += [(r, t, f) for f in _numbers(fspec, tree[r][t], "frame", no_frame)]
                if no_frame:
                    misses.append(f"{r} train {t} has no frame {_ranges(no_frame)}")
            if no_train:
                misses.append(f"{r} has no train {_ranges(no_train)}")
        if len(keys) == n0:
            print(f"no frames found for {item!r}")
        for msg in misses[:6]:
            print(f"  {item!r}: {msg}")
        if len(misses) > 6:
            print(f"  {item!r}: ... and {len(misses) - 6} more")
    keys = sorted(set(keys))
    paths = [index[k] for k in keys]
    if verbose:
        per = {}
        for r, t, _ in keys:
            per.setdefault(r, {}).setdefault(t, 0)
            per[r][t] += 1
        print(f"{len(paths)} frames selected" + "".join(
            f"\n  {r}: " + ", ".join(f"train {t}: {n}" for t, n in sorted(d.items())) for r, d in sorted(per.items())))
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


def write_run_info(run_dir: str, model_path: str, cfg: br.RCNNConfig, extra: Optional[dict] = None,
                   selection: Optional[Iterable] = None):
    """Record the model and settings of a results folder (run_info.json); `selection` (the frames asked for) is
    added to a history, so a folder holding frames of several runs can be traced back."""
    os.makedirs(run_dir, exist_ok=True)
    p = os.path.join(run_dir, "run_info.json")
    info = {}
    if os.path.exists(p):
        with open(p) as f:
            info = json.load(f)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    info.setdefault("created", now)
    info.update(model_path=os.path.abspath(model_path), updated=now, config=asdict(cfg), **(extra or {}))
    if selection is not None:
        items = [selection] if isinstance(selection, str) or (
            isinstance(selection, tuple) and not all(isinstance(x, (tuple, list)) for x in selection)) else list(selection)
        info.setdefault("selections", []).append(dict(at=now, model=os.path.basename(model_path),
                                                      items=[repr(x) for x in items]))
    with open(p, "w") as f:
        json.dump(info, f, indent=2, default=str)


def _frame_image(json_path: str, a: dict, verbose=True) -> str:
    """Image of a results / annotation JSON: the .tif next to it, else the original frame (frame_path, or raw/)."""
    return br.annotation_image(dict(a, image_path=os.path.join(os.path.dirname(json_path), a["image"])), verbose)


def draft_frames(model, cfg: br.RCNNConfig, frame_paths, run_dir: str, model_path: str = "",
                 kind: str = "auto", overwrite_unreviewed: bool = False, show_every: int = 0,
                 reuse_reviewed: bool = True, results_root: str = "results"):
    """Predict each frame and write <frame>.tif + <frame>.json (circles / ellipses) into run_dir/shapes/.

    Existing JSONs are never touched unless overwrite_unreviewed=True AND they are neither reviewed nor
    corrected in napari (annotate_bubbles writes "edited_at" when you change a frame).
    reuse_reviewed: a frame you already reviewed in another results folder under results_root is copied from
    there (reviewed, with "copied_from") instead of being predicted and reviewed again.
    kind: 'auto' (default) saves a circle where the bubble is round, else an ellipse; 'ellipse' / 'circle'
    always that (bubble_rcnn.fit_shape).
    Returns a DataFrame with one row per frame (status: drafted / copied / skipped).
    """
    import shutil
    import tifffile
    if kind not in br.SHAPE_KINDS:
        raise ValueError(f"kind must be one of {br.SHAPE_KINDS}, not {kind!r}")
    d = shapes_dir(run_dir)
    os.makedirs(d, exist_ok=True)
    elsewhere = {}
    if reuse_reviewed and os.path.isdir(results_root):
        rv = _reviewed_frames(results_root)
        rv = rv.loc[np.array([os.path.abspath(os.path.dirname(p)) != os.path.abspath(d) for p in rv["json_path"]],
                             dtype=bool)]
        elsewhere = {fr: g.sort_values("mtime")["json_path"].iloc[-1] for fr, g in rv.groupby("frame")}
    model_saved = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(model_path)))
                   if model_path and os.path.exists(model_path) else "")
    rows, t0 = [], time.time()
    for k, f in enumerate(frame_paths):
        js = _json_for(run_dir, f)
        name = os.path.splitext(os.path.basename(f))[0]
        tif = os.path.join(d, name + ".tif")
        if os.path.exists(js):
            with open(js) as fh:
                old = json.load(fh)
            if old.get("reviewed") or old.get("edited_at") or not overwrite_unreviewed:
                why = ", reviewed" if old.get("reviewed") else (", corrected in napari" if old.get("edited_at") else "")
                rows.append(dict(frame=name, status=f"skipped (exists{why})", n_bubbles=len(old.get("bubbles", []))))
                if not os.path.exists(tif):              # .tif files are not in git: recreate them
                    tifffile.imwrite(tif, bs.load_image(f))
                continue
        if name in elsewhere:                            # already reviewed in another results folder
            with open(elsewhere[name]) as fh:
                a = json.load(fh)
            src_tif = os.path.join(os.path.dirname(elsewhere[name]), a["image"])
            if not os.path.exists(tif):
                if os.path.exists(src_tif):
                    shutil.copy2(src_tif, tif)
                else:
                    tifffile.imwrite(tif, bs.load_image(f))
            bub = [b for b in br.as_ellipses(a.get("bubbles", [])) if b.get("type") == "ellipse"]
            keep = {kk: v for kk, v in a.items()
                    if kk not in ("image", "bubbles", "rois", "reviewed", "frame_path", "copied_from")}
            br.save_annotation(js, name + ".tif", bub, a.get("rois", []), **keep, reviewed=True,
                               frame_path=os.path.abspath(f), copied_from=os.path.relpath(elsewhere[name]))
            rows.append(dict(frame=name, status="copied (reviewed elsewhere)", n_bubbles=len(bub)))
            continue
        img = bs.load_image(f)
        if not os.path.exists(tif):
            tifffile.imwrite(tif, img)
        pred = br.predict(model, br.prepare_image(img, cfg), cfg)
        shapes = br.masks_to_shapes(pred, cfg, kind=kind)
        br.save_annotation(js, name + ".tif", shapes, [], reviewed=False, model=os.path.basename(model_path),
                           model_saved=model_saved, drafted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                           frame_path=os.path.abspath(f), score_thresh=cfg.score_thresh)
        rows.append(dict(frame=name, status="drafted", n_bubbles=len(shapes)))
        if show_every and (k % show_every == 0 or k == len(frame_paths) - 1):
            print(f"{k + 1}/{len(frame_paths)}  {name}: {len(shapes)} bubbles  ({time.time() - t0:.0f}s)")
    n_copied = sum(r["status"].startswith("copied") for r in rows)
    if n_copied:
        print(f"{n_copied} frame(s) were already reviewed in another results folder: copied, not predicted again")
    return pd.DataFrame(rows, columns=["frame", "status", "n_bubbles"])


def list_results(run_dir: str) -> pd.DataFrame:
    """One row per frame JSON in the run: run, train, frame, n_bubbles, reviewed, path."""
    rows = []
    for js in sorted(glob.glob(os.path.join(shapes_dir(run_dir), "*.json"))):
        with open(js) as f:
            a = json.load(f)
        key = frame_key(js) or frame_key(a.get("frame_path", "")) or (None, None, None)
        rows.append(dict(frame=os.path.splitext(os.path.basename(js))[0], run=key[0], train=key[1], frame_idx=key[2],
                         n_bubbles=len(a.get("bubbles", [])), reviewed=bool(a.get("reviewed", False)),
                         model=a.get("model", ""), json_path=js))
    return pd.DataFrame(rows, columns=["frame", "run", "train", "frame_idx", "n_bubbles", "reviewed", "model",
                                      "json_path"])


# ----------------------------------------------------------------------------
# Measuring shapes (the reviewed result)
# ----------------------------------------------------------------------------
measure_shapes = br.measure_shapes          # circles / ellipses -> one row per bubble (see bubble_rcnn)


def export_results(run_dir: str, cfg: br.RCNNConfig, include_unreviewed: bool = False, verbose=True):
    """Measure every frame of the run and write bubbles.csv + frames.csv. Returns (bubbles, frames)."""
    lst = list_results(run_dir)
    if lst.empty:
        raise FileNotFoundError(f"no frames in {shapes_dir(run_dir)}")
    use = lst if include_unreviewed else lst[lst["reviewed"]]
    if verbose and len(use) < len(lst):
        print(f"{len(lst) - len(use)} unreviewed frame(s) left out (include_unreviewed=False)")
    if use.empty:
        print("no frames to export (tick 'Reviewed' in napari, or use include_unreviewed=True); "
              "bubbles.csv / frames.csv left as they were")
        return pd.DataFrame(), pd.DataFrame()
    tables, frows = [], []
    for _, r in use.iterrows():
        with open(r.json_path) as f:
            a = json.load(f)
        hw = bs.load_image(_frame_image(r.json_path, a, verbose=False)).shape
        df = measure_shapes(a.get("bubbles", []), hw, cfg, frame=r.frame)
        if len(df):
            df.insert(1, "run", r.run)
            df.insert(2, "train", r.train)
            df.insert(3, "frame_idx", r.frame_idx)
            df.insert(4, "reviewed", r.reviewed)
            tables.append(df)
        n_in = int(df["in_analysis"].sum()) if len(df) else 0
        frows.append(dict(frame=r.frame, run=r.run, train=r.train, frame_idx=r.frame_idx, reviewed=r.reviewed,
                          n_bubbles=len(df), n_in_analysis=n_in, image_height_px=hw[0], image_width_px=hw[1],
                          model=r.model, shapes_key=br.content_key(a.get("bubbles", []), [])))
    bubbles = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["frame", "run", "train", "frame_idx", "reviewed", "bubble_id", "in_analysis"])
    frames = pd.DataFrame(frows, columns=["frame", "run", "train", "frame_idx", "reviewed", "n_bubbles",
                                          "n_in_analysis", "image_height_px", "image_width_px", "model", "shapes_key"])
    bubbles.to_csv(os.path.join(run_dir, "bubbles.csv"), index=False)
    frames.to_csv(os.path.join(run_dir, "frames.csv"), index=False)
    with open(os.path.join(run_dir, "export_settings.json"), "w") as f:
        json.dump(dict(exported=time.strftime("%Y-%m-%d %H:%M:%S"), include_unreviewed=include_unreviewed,
                       config=asdict(cfg)), f, indent=2, default=str)
    if verbose:
        print(f"exported {len(frames)} frames, {len(bubbles)} bubbles -> {run_dir}/bubbles.csv, frames.csv")
    return bubbles, frames


def _add_run_column(df):
    """Older exports have no run column: take it from the frame name."""
    if len(df.columns) and "run" not in df.columns and "frame" in df.columns:
        df.insert(df.columns.get_loc("frame") + 1, "run", [run_id(f) for f in df["frame"]])
    return df


def _read_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:            # written by an older export that found no bubbles
        return pd.DataFrame()


def load_results(run_dir: str, reviewed_only: bool = True, in_analysis_only: bool = False):
    """Load bubbles.csv and frames.csv of a run (written by export_results)."""
    bubbles = _add_run_column(_read_csv(os.path.join(run_dir, "bubbles.csv")))
    frames = _add_run_column(_read_csv(os.path.join(run_dir, "frames.csv"), dtype={"shapes_key": str}))
    if reviewed_only:
        if len(bubbles):
            bubbles = bubbles[bubbles["reviewed"].astype(bool)]
        if len(frames):
            frames = frames[frames["reviewed"].astype(bool)]
    if in_analysis_only and len(bubbles):
        bubbles = bubbles[bubbles["in_analysis"].astype(bool)]
    if not len(frames):
        print(f"no frames in {run_dir}/frames.csv" + (" (reviewed_only=True: none reviewed?)" if reviewed_only else ""))
    elif "shapes_key" in frames:            # bubble_id points into the JSON shapes: warn if they changed since
        changed = []
        for fr, key in zip(frames["frame"], frames["shapes_key"]):
            js = os.path.join(shapes_dir(run_dir), fr + ".json")
            if os.path.exists(js):
                with open(js) as f:
                    if br.content_key(json.load(f).get("bubbles", []), []) != key:
                        changed.append(fr)
        if changed:
            print(f"WARNING: {len(changed)} frame(s) were changed after the export (e.g. {changed[0]}): "
                  "run the export again before analysing")
    return bubbles.reset_index(drop=True), frames.reset_index(drop=True)


def load_frame(run_dir: str, frame: str):
    """(image, shapes) of one frame of a run."""
    js = os.path.join(shapes_dir(run_dir), frame + ".json")
    with open(js) as f:
        a = json.load(f)
    return bs.load_image(_frame_image(js, a, verbose=False)), br.as_ellipses(a.get("bubbles", []))


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


def _reviewed_frames(results_root: str, recursive: bool = True) -> pd.DataFrame:
    """All reviewed frame JSONs in results_root/**/shapes/ (or only results_root/shapes/ if not recursive)."""
    pattern = os.path.join(results_root, "**", "shapes", "*.json") if recursive else \
        os.path.join(results_root, "shapes", "*.json")
    rows = []
    for js in sorted(glob.glob(pattern, recursive=recursive)):
        if "tracked" in os.path.normpath(js).split(os.sep):   # output of the old bubble tracking: never training data
            continue
        with open(js) as f:
            a = json.load(f)
        if a.get("tracked"):
            continue
        if a.get("reviewed") and a.get("bubbles"):
            rows.append(dict(frame=os.path.splitext(os.path.basename(js))[0], json_path=js,
                             mtime=os.path.getmtime(js), n_bubbles=len(a["bubbles"])))
    return pd.DataFrame(rows, columns=["frame", "json_path", "mtime", "n_bubbles"])


def sync_reviewed_to_training(results_root: str, annotation_dir: str, run_dir: Optional[str] = None,
                              frames: Optional[list] = None, verbose=True) -> pd.DataFrame:
    """Copy every reviewed frame (tif + json) from the results folders into the training annotations.

    The whole frame counts as fully annotated, unless you drew 'fully annotated' rectangles while reviewing
    (then only those regions). Run it before training; re-running is safe:
    * new reviewed frames are copied;
    * frames copied earlier are updated when you have corrected them again in the results folder since;
    * annotations you made in the training folder yourself, or changed there after copying, are never overwritten
      (only a real change of the shapes counts, not opening and saving the file); an uncorrected model draft staged
      there (bubble_training.ipynb section 8) is replaced by the reviewed frame;
    * frames you moved to <annotation_dir>/excluded/ are not copied again;
    * a frame reviewed in several results folders: the most recently saved version is used.
    run_dir: only this results folder (not its sub-folders; default: everything under results_root).
    frames: only these frame names. Returns one row per reviewed frame with the action taken.
    """
    import shutil
    src = _reviewed_frames(run_dir, recursive=False) if run_dir is not None else _reviewed_frames(results_root)
    if frames is not None:
        src = src[src["frame"].isin(frames)]
    os.makedirs(annotation_dir, exist_ok=True)
    rows = []
    for frame, g in src.sort_values("mtime").groupby("frame", sort=False):
        r = g.iloc[-1]                                   # newest version of this frame
        dst_js = os.path.join(annotation_dir, frame + ".json")
        with open(r.json_path) as f:
            a = json.load(f)
        bub = [b for b in br.as_ellipses(a["bubbles"]) if b.get("type") == "ellipse"]
        rois = a.get("rois", [])
        src_key = br.content_key(bub, rois)
        action = "copied"
        if os.path.exists(os.path.join(annotation_dir, "excluded", frame + ".json")):
            action = "skipped (in excluded/)"
        elif os.path.exists(dst_js):
            with open(dst_js) as f:
                old = json.load(f)
            imp = old.get("imported_from")
            if "imported_key" in old:      # edited here = the shapes differ from what was imported
                edited = br.content_key(old.get("bubbles", []), old.get("rois", [])) != old["imported_key"]
            else:                          # copies made by an older version: judge by modification time
                edited = os.path.getmtime(dst_js) > old.get("imported_at", 0) + 5
            if imp is None and not old.get("bubbles") and not old.get("rois"):
                action = "updated (replaces an empty staged frame)"
            elif imp is None and old.get("draft") and not (old.get("reviewed") or old.get("rois")):
                action = "updated (replaces an uncorrected draft)"
            elif imp is None:
                action = "skipped (own training annotation)"
            elif edited:
                action = "skipped (edited in the training folder)"
            elif old.get("imported_key") == src_key or (
                    "imported_key" not in old and os.path.abspath(imp.get("path", "")) == os.path.abspath(r.json_path)
                    and r.mtime <= imp.get("mtime", 0) + 1e-3):
                action = "up to date"
            else:
                action = "updated"
        if action in ("copied", "updated") or action.startswith("updated"):
            try:
                src_img = _frame_image(r.json_path, a, verbose=False)
            except FileNotFoundError:
                rows.append(dict(frame=frame, action="skipped (frame image missing)", source=r.json_path,
                                 n_bubbles=r.n_bubbles, n_versions=len(g)))
                continue
            dst_img = os.path.join(annotation_dir, a["image"])
            if not os.path.exists(dst_img):
                if src_img.lower().endswith((".tif", ".tiff")):
                    shutil.copy2(src_img, dst_img)
                else:
                    import tifffile
                    tifffile.imwrite(dst_img, bs.load_image(src_img))
            now = time.time()
            br.save_annotation(dst_js, a["image"], bub, rois, frame_path=a.get("frame_path", ""),
                               imported_from=dict(path=os.path.relpath(r.json_path), mtime=r.mtime),
                               imported_at=now, imported_key=src_key)
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
    return out.loc[out["action"].str.startswith(("copied", "updated")), "frame"].tolist()


def polygons_to_ellipses(folder: str, dry_run: bool = False) -> pd.DataFrame:
    """Rewrite the polygons in every JSON of a folder (results/<run>/shapes, annotations, ...) as ellipses with the
    same area, centre and orientation (bubble_rcnn.as_ellipse). Older versions saved polygons; they are read as
    ellipses anyway, this makes the files themselves circles / ellipses only. Unreviewed drafts are better
    predicted again (OVERWRITE_UNREVIEWED = True). dry_run: only count. Returns one row per file with polygons."""
    rows = []
    for js in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(js) as f:
            a = json.load(f)
        bub = a.get("bubbles")
        if not isinstance(bub, list):
            continue
        n = sum(b.get("type") in ("polygon", "path") for b in bub)
        if not n:
            continue
        new = [b for b in br.as_ellipses(bub) if b.get("type") == "ellipse"]
        rows.append(dict(file=os.path.basename(js), polygons=n, dropped=len(bub) - len(new), reviewed=a.get("reviewed")))
        if not dry_run:
            st = os.stat(js)
            a["bubbles"] = new
            tmp = js + ".tmp"
            with open(tmp, "w") as f:
                json.dump(a, f)
            os.replace(tmp, js)
            os.utime(js, ns=(st.st_atime_ns, st.st_mtime_ns))   # a format change is not a new review: keep the time
    out = pd.DataFrame(rows, columns=["file", "polygons", "dropped", "reviewed"])
    print(f"{folder}: {int(out['polygons'].sum()) if len(out) else 0} polygon(s) in {len(out)} file(s)"
          + (" (dry run, nothing changed)" if dry_run else " rewritten as ellipses"))
    return out
