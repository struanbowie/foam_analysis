"""
napari annotation tool for overlapping bubbles.

    python annotate_bubbles.py annotations/<frame>.json

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
        else:
            print(f"  skipping unsupported shape type '{t}' in bubbles layer (use ellipse or polygon)")
    return out


def layer_to_rois(layer):
    out = []
    for d in layer.data:
        d = np.asarray(d)[:, -2:]
        out.append([float(d[:, 0].min()), float(d[:, 1].min()), float(d[:, 0].max()), float(d[:, 1].max())])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="annotation JSON (created by the training notebook)")
    ap.add_argument("--no-rim", action="store_true", help="do not compute the rim-map helper layer")
    args = ap.parse_args()

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
    bub = ann.get("bubbles", [])
    bubbles = viewer.add_shapes([np.asarray(b["data"]) for b in bub], shape_type=[b["type"] for b in bub],
                                name="bubbles", edge_color="lime", face_color=[0, 0, 0, 0], edge_width=0.4)
    bubbles.current_edge_color = "lime"
    bubbles.current_face_color = [0, 0, 0, 0]
    bubbles.current_edge_width = 0.4
    viewer.layers.selection.active = bubbles

    status = QLabel("")

    def save(*_):
        data = dict(image=ann["image"], bubbles=layer_to_bubbles(bubbles), rois=layer_to_rois(rois))
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
                         "fully annotated layer: R = rectangle"))
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
