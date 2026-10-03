# foam_analysis

Bubble segmentation for high-speed foam image series.

* `bubble_segmentation.ipynb`: step-by-step pipeline on one frame, with every tunable parameter in a single CONFIG cell.
* `preprocessing/normalise_runs_TKM_fast_2.ipynb`: raw run → SVD-normalised frames + GIF (no annotations, native resolution).
* `bubble_seg.py`: the pipeline functions (preprocessing, Cellpose-SAM, circle fitting, Hough completion, QC, batch driver).

Input: the clean native 400×250 frames (3.2 µm/px, no timestamp/scale bar) exported by
`preprocessing/normalise_runs_TKM_fast_2.ipynb` (SVD flat-field normalisation of the raw HPV-X2 runs).

Pipeline: optional exclusion regions → masked background flattening, denoising, CLAHE and a
noise-normalised rim (dark-ridge) map → static-structure mask (fibres) → Cellpose-SAM (multi-diameter passes) →
robust circle fit per mask refined on the rim map (recovers partly hidden bubbles) → optional ridge-Hough completion
for large bubbles → quality scores + duplicate suppression → CSV of x, y, r (px and µm) per bubble.

## Setup (DESY Maxwell)

```bash
module load maxwell python/3.11
python -m venv ~/venvs/foam && source ~/venvs/foam/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name foam --display-name "foam (cellpose)"
# download the Cellpose-SAM weights once (~1.2 GB) on a node with internet, e.g. a login node.
# gpu=False only because login nodes have no GPU; it is just a download. The notebook itself
# runs on the GPU (cp_gpu=True in CONFIG).
python -c "from cellpose import models; models.CellposeModel(gpu=False)"
```
On a GPU node (e.g. A100), check that PyTorch sees the card before running the notebook:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # expect: True NVIDIA A100...
```
If this prints `False`, the installed torch wheel does not match the node's CUDA driver: reinstall torch with the
CUDA build from https://pytorch.org/get-started/locally/ (e.g. `--index-url https://download.pytorch.org/whl/cu124`).

Open the notebook from the repository root (paths are relative to it) on a GPU JupyterHub session with the `foam (cellpose)` kernel.
