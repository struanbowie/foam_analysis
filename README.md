# foam_analysis

Bubble segmentation and analysis for high-speed (HPV-X2) foam image series: hand-annotate bubbles, train a
Mask R-CNN that handles overlapping, nested and non-spherical bubbles, review its predictions, analyse the results.

## Pipeline

| step | notebook | output |
|---|---|---|
| 1. Normalise a run | `preprocessing/normalise_runs_TKM_fast_2.ipynb` | `raw/jpg_<run>/` clean 400×250 frames (analysis), `raw/jpg_<run>_annotated/` with scale bar + time stamp (figures), optional `raw/<run>.npy` |
| 1b. GIFs of the runs (optional) | `preprocessing/make_run_gifs.ipynb` | `raw/gifs/<run>_annotated.gif` from the annotated frames of every run that has no GIF yet |
| 2. Train the model | `bubble_training.ipynb` | `annotations/` (your napari circles / ellipses on frames picked from any runs and trains), `models/` (trained models) |
| 3. Predict + review | `bubble_inference.ipynb` | `results/<run_name>/shapes/` (reviewed circles / ellipses), `bubbles.csv`, `frames.csv`; any selection of frames, trains or runs |
| 4. Analyse | `bubble_analysis.ipynb` | your analysis of `results/<run_name>/` |
| 4c. Compare runs | `bubble_compare_runs.ipynb` | count, size, area, ... per train with several runs on one plot, aligned on the laser-scan train (bubble-count jump), also relative to before the scan; figures in `results/compare/` |
| 4b. Analyse one train, every frame | `bubble_train_analysis.ipynb` | predicts all frames of train `N` into the run's folder, `results/<run>/tid<N>/`, time-series plots and an overlay GIF in `figures/` |

Annotation and review happen in napari: `python annotate_bubbles.py <folder>` (FastX desktop session on Maxwell).
Bubbles are circles and ellipses only: draw an ellipse (E), select shapes and press C to make them circles.

Frames are picked by `(run, train, frames)`, e.g. `("r571", 3, "10-20")`, `("r571", "all", 10)` (frame 10 of every
train) or `(["r571", "r572"], "all", 10)` (several runs); see `bubble_io.select_frames`.

## Code

* `bubble_rcnn.py`: Mask R-CNN model, training, prediction, edge-aware measurement (bubbles cut by the image edge
  are fitted from their visible arc), annotation I/O, measurement of circles / ellipses. Predicted masks are saved
  as circles (round bubbles) or ellipses (`fit_shape`); polygons in older files are read as ellipses.
* `bubble_io.py`: frame indexing and selection across runs (frames / ranges / whole trains / runs), results folders,
  export and loading of results (with `run`, `train`, `frame_idx` columns), syncing reviewed frames into the training
  set. `polygons_to_ellipses(folder)` rewrites polygons saved by older versions as ellipses (they are read as
  ellipses anyway; unreviewed drafts are better predicted again with `OVERWRITE_UNREVIEWED = True`).
* `bubble_stats.py`: per-frame metrics (count, density, mean/median/Sauter radius, coverage, volume proxy, shape,
  centroid) and the analysis plots.
* `bubble_seg.py`: image preprocessing (background flattening, denoising, CLAHE, rim map), the model's input channels.
* `annotate_bubbles.py`: napari tool (one circle / ellipse per bubble, overlaps allowed; "fully annotated" rectangles
  for training; **Reviewed** checkbox for results).
* `preprocessing/svd_on_datasets.py`: SVD flat-field model used by the normalisation notebook.

## Folders

| folder | in git? | content |
|---|---|---|
| `annotations/` | `.json` yes, `.tif` no | training annotations (+ frame copies) |
| `results/` | `.json`/`.csv` yes, `.tif` no | reviewed predictions and measurements per inference run (may hold frames of several runs); `results/<run>/tid<N>/` per-train runs inside their run folder |
| `raw/` | no | normalised frames from step 1 |
| `models/` | no | trained models (~180 MB each) |

## Setup (DESY Maxwell)

The bubble notebooks use the `foam` environment (conda env on GPFS, see below). The normalisation notebook needs
`extra_data` (European XFEL); run it with the kernel that provides it (e.g. the default Python 3 kernel on max-jhub).

```bash
SB=/gpfs/exfel/u/usr/SPB/202501/p007699/Shared/sbowie
ENV=$SB/venvs/foam
export PIP_CACHE_DIR=$SB/.cache/pip TORCH_HOME=$SB/.cache/torch
mamba create -y -p $ENV python=3.11 pip && mamba activate $ENV
pip install -r requirements.txt
pip install "napari[all]"                        # where you annotate (FastX)
python -m ipykernel install --user --name foam --display-name "foam (bubbles)" --env TORCH_HOME $TORCH_HOME
# COCO-pretrained Mask R-CNN weights, once (needs internet, e.g. a login node):
python -c "from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as m; m(weights='DEFAULT')"
```
Check the GPU on a GPU node:
`python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"`.
If the capability (e.g. `(7, 0)` = V100, `(6, 0)` = P100) is missing from the arch list you get
"CUDA error: no kernel image is available for execution on the device". Either use an A100/H100 node, or install a
PyTorch build that still supports older GPUs: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

Run the notebooks from the repository root (paths are relative to it) on a GPU JupyterHub session with the
`foam (bubbles)` kernel.
