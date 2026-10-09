"""
napari annotation tool for overlapping bubbles.

    python annotate_bubbles.py                      # list frames in ./annotations, open the first unannotated one
    python annotate_bubbles.py annotations          # same, for a given folder
    python annotate_bubbles.py annotations/r563_svd_normalised_tr050_tid0_000.json   # a specific frame

Layers
------
* image          : the frame (native resolution)
* rim map        : dark-ridge map from bubble_seg (hidden; toggle the eye icon to help see faint rims)
* bubbles        : one CIRCLE or ELLIPSE per bubble. Draw an ellipse (key E); select shapes (key S) and press
                   C to turn them into exact circles (same centre and area). Overlapping shapes are fine.
                   Draw the FULL outline of every bubble, including the part hidden behind
                   another bubble or a fibre (continue the visible arc).
                   Only circles and ellipses are saved: a polygon drawn by mistake is saved as the ellipse
                   with the same area and orientation, lines are ignored. Polygons in older files are shown
                   (and saved) as ellipses.
* fully annotated: RECTANGLES (key R) around regions where EVERY bubble has been drawn.
                   Only these regions are used for training, so you can annotate part of a frame.
                   No rectangle at all = the whole frame counts as fully annotated.

Saving: "Save" button (right dock) or Shift-S. Also autosaves every 2 minutes.
Reviewed: tick the checkbox when a frame is fully checked (used by bubble_inference.ipynb to
decide which frames are final). Works for result folders too:
    python annotate_bubbles.py results/<run_name>/shapes
Run on Maxwell inside a FastX desktop session (max-display), or locally after copying the
annotations folder (then copy the .json files back).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bubble_rcnn as br   # noqa: E402  (circle / ellipse helpers)


def rect_corners(r0, c0, r1, c1):
    return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]], float)


MIN_SEMI_AXIS_PX = 0.5      # ellipses thinner than this (stray clicks, collapsed resizes) are not saved
MIN_ROI_PX = 4.0            # 'fully annotated' rectangles smaller than this (stray clicks) are not saved


def _shape_id(data):
    return tuple(np.round(np.asarray(data, float)[:, -2:], 3).ravel())


def layer_to_bubbles(layer, notes=None, extras=None):
    """Bubbles layer -> list of napari ellipses (circles are ellipses with equal axes).
    Polygons / paths are saved as their ellipse, lines and degenerate ellipses are left out (counted in notes).
    extras: {shape id: extra keys} of the loaded shapes; shapes that were not changed keep them (e.g. fit_kind)."""
    out, converted, skipped, tiny = [], 0, 0, 0
    for d, t in zip(layer.data, layer.shape_type):
        if t == "rectangle":     # treated as ROIs (see move_rectangles_to_rois)
            continue
        sh = br.as_ellipse(dict(type=t, data=np.asarray(d)[:, -2:].tolist()))
        if sh is None:
            skipped += 1
            continue
        if br.ellipse_params(sh)[3] < MIN_SEMI_AXIS_PX:
            tiny += 1
            continue
        converted += t != "ellipse"
        out.append({**(extras or {}).get(_shape_id(sh["data"]), {}), "type": "ellipse", "data": sh["data"]})
    if notes is not None:
        if converted:
            notes.append(f"{converted} polygon(s) saved as ellipses")
        if skipped:
            notes.append(f"{skipped} line(s) ignored")
        if tiny:
            notes.append(f"{tiny} ellipse(s) thinner than {MIN_SEMI_AXIS_PX} px left out")
    return out


def circle_data(shape_type, d):
    """A napari ellipse (or polygon) -> the circle with the same centre and area, as 4 bounding-box corners."""
    sh = br.as_ellipse(dict(type=shape_type, data=np.asarray(d)[:, -2:].tolist()))
    if sh is None:
        return None
    x, y, a, b, th = br.ellipse_params(sh)
    r = float(np.sqrt(a * b))
    return np.asarray(br.ellipse_shape(x, y, r, r, th)["data"])


def make_circles(layer):
    """Replace the selected ellipses (and polygons) of a shapes layer by circles. Returns the number changed."""
    idx = sorted(i for i in layer.selected_data if layer.shape_type[i] in ("ellipse", "polygon", "path"))
    new = [c for c in (circle_data(layer.shape_type[i], layer.data[i]) for i in idx) if c is not None]
    if not new:
        return 0
    layer.selected_data = set(idx)
    layer.remove_selected()
    layer.add(new, shape_type="ellipse", edge_color="lime", face_color=[0, 0, 0, 0], edge_width=0.4)
    n = len(layer.data)
    layer.selected_data = set(range(n - len(new), n))
    return len(new)


def rectangles_in(layer):
    """Rectangles accidentally drawn on the bubbles layer (indices, corner arrays)."""
    idx = [i for i, t in enumerate(layer.shape_type) if t == "rectangle"]
    return idx, [np.asarray(layer.data[i])[:, -2:] for i in idx]


def move_rectangles_to_rois(bubbles, rois):
    """Move rectangles from the bubbles layer to the 'fully annotated' layer (visual fix)."""
    idx, rects = rectangles_in(bubbles)
    if not idx:
        return 0
    try:
        rois.add(rects, shape_type="rectangle")
        bubbles.selected_data = set(idx)
        bubbles.remove_selected()
    except Exception as e:   # e.g. window already closed: the save below still counts them as ROIs
        print("  (could not move rectangles between layers:", e, ")")
        return -len(idx)
    print(f"  moved {len(idx)} rectangle(s) from 'bubbles' to 'fully annotated'")
    return len(idx)


def shapes_not_rectangles(layer):
    """Ellipses accidentally drawn on the 'fully annotated' layer (indices, types, vertices). Polygons there are
    rectangles edited with the vertex tools (napari turns them into polygons): they stay ROIs."""
    idx = [i for i, t in enumerate(layer.shape_type) if t == "ellipse"]
    return idx, [layer.shape_type[i] for i in idx], [np.asarray(layer.data[i])[:, -2:] for i in idx]


def move_bubbles_from_rois(rois, bubbles):
    """Move ellipses drawn on the 'fully annotated' layer to the bubbles layer (they are bubbles, not ROIs)."""
    idx, types, data = shapes_not_rectangles(rois)
    if not idx:
        return 0
    try:
        for t, d in zip(types, data):
            bubbles.add([d], shape_type=t, edge_color="lime", face_color=[0, 0, 0, 0], edge_width=0.4)
        rois.selected_data = set(idx)
        rois.remove_selected()
    except Exception as e:   # window already closed: the save below adds them to the bubbles directly
        print("  (could not move shapes between layers:", e, ")")
        return -len(idx)
    print(f"  moved {len(idx)} shape(s) from 'fully annotated' to 'bubbles'")
    return len(idx)


def layer_to_rois(layer, notes=None):
    """Rectangles (and rectangles edited into polygons) of the 'fully annotated' layer -> bounding boxes
    [r0, c0, r1, c1]; stray tiny rectangles are left out."""
    out, tiny = [], 0
    for d, t in zip(layer.data, layer.shape_type):
        if t not in ("rectangle", "polygon"):
            continue
        d = np.asarray(d)[:, -2:]
        r = [float(d[:, 0].min()), float(d[:, 1].min()), float(d[:, 0].max()), float(d[:, 1].max())]
        if r[2] - r[0] < MIN_ROI_PX or r[3] - r[1] < MIN_ROI_PX:
            tiny += 1
            continue
        out.append(r)
    if tiny and notes is not None:
        notes.append(f"{tiny} tiny rectangle(s) left out")
    return out


def build_payload(ann, bubbles, rois_list, reviewed):
    """JSON to save: keeps every extra field of the original file (model, frame_path, ...)."""
    data = {k: v for k, v in ann.items() if k not in ("bubbles", "rois")}
    data.update(bubbles=bubbles, rois=rois_list)
    if reviewed is not None:
        data["reviewed"] = bool(reviewed)
    return data


def resolve_target(arg):
    """Accept a JSON file or a folder; for a folder, list its annotation files and pick the first one still to do
    (results: first not reviewed; training: first without bubbles or an uncorrected draft), or ask."""
    path = os.path.abspath(arg)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, arg)
        if os.path.exists(alt):
            return resolve_target(alt)
        sys.exit(f"not found: {path}\n(run section 2 of bubble_training.ipynb first, or check the path / current folder)")
    if os.path.isdir(os.path.join(path, "shapes")):      # a results folder: its frames are in shapes/
        path = os.path.join(path, "shapes")
    files, anns = [], []
    for f in sorted(f for f in os.listdir(path) if f.endswith(".json")):
        try:
            with open(os.path.join(path, f)) as fh:
                a = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(a, dict) and "image" in a:            # skip run_info.json etc.
            files.append(f)
            anns.append(a)
    if not files:
        sys.exit(f"no annotation .json files in {path} (run section 2 of bubble_training.ipynb to stage frames)")
    results_mode = any("drafted_at" in a for a in anns)  # frames drafted by bubble_inference / draft_frames
    print("annotation files:")
    for i, (f, a) in enumerate(zip(files, anns)):
        flags = [", reviewed" if a.get("reviewed") else "", ", draft to correct" if a.get("draft") and not
                 (a.get("reviewed") or a.get("rois")) else ""]
        print(f"  [{i}] {f}  ({len(a.get('bubbles', []))} bubbles{''.join(flags)})")
    if results_mode:
        default = next((i for i, a in enumerate(anns) if not a.get("reviewed")), 0)
    else:
        default = next((i for i, a in enumerate(anns) if not a.get("bubbles") or
                        (a.get("draft") and not (a.get("reviewed") or a.get("rois")))), 0)
    while True:
        ans = input(f"open which? [enter = {default}] ").strip() if sys.stdin.isatty() else ""
        if not ans:
            return os.path.join(path, files[default])
        if ans.isdigit() and int(ans) < len(files):
            return os.path.join(path, files[int(ans)])
        print(f"  enter a number 0-{len(files) - 1}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default="annotations",
                    help="annotation JSON, or a folder of them (default: ./annotations)")
    ap.add_argument("--no-rim", action="store_true", help="do not compute the rim-map helper layer")
    args = ap.parse_args()
    args.json = resolve_target(args.json)
    print("opening", args.json)

    import napari
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

    import bubble_seg as bs

    with open(args.json) as f:
        ann = json.load(f)
    img = bs.load_image(br.annotation_image(dict(ann, image_path=os.path.join(os.path.dirname(args.json), ann["image"]))))

    viewer = napari.Viewer(title=f"bubbles: {os.path.basename(args.json)}")
    lo, hi = np.percentile(img, [0.5, 99.5])
    viewer.add_image(img, name="image", colormap="gray", contrast_limits=(lo, hi))
    if not args.no_rim:
        cfg = bs.Config(work_scale=1.0, exclude_boxes=[])
        pre = bs.preprocess(img, bs.build_valid_mask(img.shape, cfg), cfg)
        viewer.add_image(pre["ridge_z"], name="rim map", colormap="magma", contrast_limits=(0, 10),
                         visible=False, blending="additive")

    rois = viewer.add_shapes([rect_corners(*r) for r in ann.get("rois", [])], shape_type="rectangle",
                             name="fully annotated", edge_color="yellow", face_color=[0, 0, 0, 0], edge_width=0.6)
    rois.current_edge_color = "yellow"
    rois.current_face_color = [0, 0, 0, 0]
    rois.current_edge_width = 0.6
    bub = [b for b in br.as_ellipses(ann.get("bubbles", []), warn=os.path.basename(args.json))
           if b.get("type") == "ellipse"]
    bubbles = viewer.add_shapes([np.asarray(b["data"]) for b in bub], shape_type="ellipse",
                                name="bubbles", edge_color="lime", face_color=[0, 0, 0, 0], edge_width=0.4)
    bubbles.current_edge_color = "lime"
    bubbles.current_face_color = [0, 0, 0, 0]
    bubbles.current_edge_width = 0.4
    viewer.layers.selection.active = bubbles
    extras = {_shape_id(b["data"]): {k: v for k, v in b.items() if k not in ("type", "data")} for b in bub}

    status = QLabel("")
    chk = QCheckBox("Reviewed (frame fully checked)")
    chk.setChecked(bool(ann.get("reviewed", False)))
    # what the file holds, as shown in napari (older polygons as ellipses): only changes from this are edits
    state = dict(reviewed=chk.isChecked(), saved=(br.content_key(layer_to_bubbles(bubbles, extras=extras),
                                                                 layer_to_rois(rois)),
                                                  ann.get("reviewed")))

    def show(msg):
        print(msg)
        try:
            status.setText(msg)
        except RuntimeError:     # the window is already closed (final save)
            pass

    def save(*_):
        notes = []
        moved = move_rectangles_to_rois(bubbles, rois)
        moved_b = move_bubbles_from_rois(rois, bubbles)
        roi_list = layer_to_rois(rois, notes)
        if moved < 0:   # could not move them in the viewer: include them directly
            roi_list += [[float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 0].max()), float(r[:, 1].max())]
                         for r in rectangles_in(bubbles)[1]]
        bub = layer_to_bubbles(bubbles, notes, extras)
        if moved_b < 0:  # shapes still on the ROI layer: they are bubbles
            _, types, data = shapes_not_rectangles(rois)
            bub += [dict(type="ellipse", data=e["data"]) for e in
                    (br.as_ellipse(dict(type=t, data=d.tolist())) for t, d in zip(types, data)) if e]
        # "reviewed" is only written for result files (or once ticked), so training files stay unchanged
        rv = state["reviewed"] if ("reviewed" in ann or state["reviewed"]) else None
        key = (br.content_key(bub, roi_list), rv)
        if key == state["saved"]:          # nothing changed: leave the file (and its modification time) alone
            show("no changes" + (" (" + "; ".join(notes) + ")" if notes else ""))
            return
        data = build_payload(ann, bub, roi_list, rv)
        if key[0] != state["saved"][0]:
            data["edited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        elif "edited_at" in ann:
            data["edited_at"] = ann["edited_at"]
        tmp = args.json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, args.json)
        ann.clear()
        ann.update(data)
        state["saved"] = key
        show(f"saved {len(data['bubbles'])} bubbles, {len(data['rois'])} ROIs" + (", reviewed" if data.get("reviewed") else "")
             + ("; " + "; ".join(notes) if notes else "") + f" -> {os.path.basename(args.json)}")

    def on_reviewed(*_):
        state["reviewed"] = chk.isChecked()
        save()

    def circles(*_):
        n = make_circles(bubbles)
        show(f"{n} shape(s) made circles" if n else "select ellipses first (select tool S), then press C")

    viewer.bind_key("Shift-S", save, overwrite=True)
    bubbles.bind_key("c", circles, overwrite=True)
    w = QWidget()
    lay = QVBoxLayout(w)
    btn = QPushButton("Save annotations (Shift-S)")
    btn.clicked.connect(save)
    lay.addWidget(btn)
    btn_c = QPushButton("Make selected bubbles circles (C)")
    btn_c.clicked.connect(circles)
    lay.addWidget(btn_c)
    chk.stateChanged.connect(on_reviewed)
    lay.addWidget(chk)
    lay.addWidget(QLabel("bubbles layer: E = ellipse\n"
                         "select (S) a shape to move / rotate / resize it\n"
                         "C = make the selected shapes circles\n"
                         "fully annotated layer: R = rectangle\n"
                         "(rectangles drawn on the bubbles layer are moved\n to 'fully annotated' on save, and ellipses\n"
                         " drawn on 'fully annotated' to 'bubbles')"))
    lay.addWidget(status)
    lay.addStretch()
    viewer.window.add_dock_widget(w, name="save", area="right")

    timer = QTimer()
    timer.timeout.connect(save)
    timer.start(120_000)
    napari.run()
    save()  # final save when the window closes


if __name__ == "__main__":
    main()
