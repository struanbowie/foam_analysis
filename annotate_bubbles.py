"""
napari annotation tool for overlapping bubbles.

    python annotate_bubbles.py                      # list frames in ./annotations, open the first unannotated one
    python annotate_bubbles.py annotations          # same, for a given folder
    python annotate_bubbles.py annotations/r563_svd_normalised_tr050_tid0_000.json   # a specific frame

Layers
------
* image          : the frame (native resolution)
* rim map        : dark-ridge map from bubble_seg (hidden; toggle the eye icon to help see faint rims)
* bubbles        : one ELLIPSE (key E) or POLYGON (key P) per bubble. Overlapping shapes are fine.
                   Draw the FULL outline of every bubble, including the part hidden behind
                   another bubble or a fibre (continue the visible arc).
* fully annotated: RECTANGLES (key R) around regions where EVERY bubble has been drawn.
                   Only these regions are used for training, so you can annotate part of a frame.
                   No rectangle at all = the whole frame counts as fully annotated.

Saving: "Save" button (right dock) or Shift-S. Also autosaves every 2 minutes.
Run on Maxwell inside a FastX desktop session (max-display), or locally after copying the
annotations folder (then copy the .json files back).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rect_corners(r0, c0, r1, c1):
    return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]], float)


def layer_to_bubbles(layer):
    out = []
    for d, t in zip(layer.data, layer.shape_type):
        if t in ("ellipse", "polygon"):
            out.append(dict(type=t, data=np.asarray(d)[:, -2:].tolist()))
        elif t != "rectangle":   # rectangles are treated as ROIs (see move_rectangles_to_rois)
            print(f"  skipping unsupported shape type '{t}' in bubbles layer (use ellipse or polygon)")
    return out


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


def layer_to_rois(layer):
    out = []
    for d in layer.data:
        d = np.asarray(d)[:, -2:]
        out.append([float(d[:, 0].min()), float(d[:, 1].min()), float(d[:, 0].max()), float(d[:, 1].max())])
    return out


def resolve_target(arg):
    """Accept a JSON file or a folder; for a folder, list its annotation files and pick the first
    one without bubbles (or ask)."""
    path = os.path.abspath(arg)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, arg)
        if os.path.exists(alt):
            return resolve_target(alt)
        sys.exit(f"not found: {path}\n(run section 2 of bubble_training.ipynb first, or check the path / current folder)")
    files = sorted(f for f in os.listdir(path) if f.endswith(".json"))
    if not files:
        sys.exit(f"no .json files in {path} (run section 2 of bubble_training.ipynb to stage frames)")
    counts = []
    for f in files:
        with open(os.path.join(path, f)) as fh:
            counts.append(len(json.load(fh).get("bubbles", [])))
    print("annotation files:")
    for i, (f, n) in enumerate(zip(files, counts)):
        print(f"  [{i}] {f}  ({n} bubbles)")
    default = next((i for i, n in enumerate(counts) if n == 0), 0)
    ans = input(f"open which? [enter = {default}] ").strip() if sys.stdin.isatty() else ""
    return os.path.join(path, files[int(ans) if ans else default])


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
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

    import bubble_seg as bs

    with open(args.json) as f:
        ann = json.load(f)
    img_path = os.path.join(os.path.dirname(args.json), ann["image"])
    img = bs.load_image(img_path)

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
    bub = ann.get("bubbles", [])
    bubbles = viewer.add_shapes([np.asarray(b["data"]) for b in bub], shape_type=[b["type"] for b in bub],
                                name="bubbles", edge_color="lime", face_color=[0, 0, 0, 0], edge_width=0.4)
    bubbles.current_edge_color = "lime"
    bubbles.current_face_color = [0, 0, 0, 0]
    bubbles.current_edge_width = 0.4
    viewer.layers.selection.active = bubbles

    status = QLabel("")

    def save(*_):
        moved = move_rectangles_to_rois(bubbles, rois)
        roi_list = layer_to_rois(rois)
        if moved < 0:   # could not move them in the viewer: include them directly
            roi_list += [[float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 0].max()), float(r[:, 1].max())]
                         for r in rectangles_in(bubbles)[1]]
        data = dict(image=ann["image"], bubbles=layer_to_bubbles(bubbles), rois=roi_list)
        tmp = args.json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, args.json)
        msg = f"saved {len(data['bubbles'])} bubbles, {len(data['rois'])} ROIs"
        status.setText(msg)
        print(msg, "->", args.json)

    viewer.bind_key("Shift-S", save, overwrite=True)
    w = QWidget()
    lay = QVBoxLayout(w)
    btn = QPushButton("Save annotations (Shift-S)")
    btn.clicked.connect(save)
    lay.addWidget(btn)
    lay.addWidget(QLabel("bubbles layer: E = ellipse, P = polygon\n"
                         "select (S) a shape to move / rotate / resize it\n"
                         "fully annotated layer: R = rectangle\n"
                         "(rectangles drawn on the bubbles layer are\n moved to 'fully annotated' on save)"))
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
