# foam_analysis

Bubble segmentation for high-speed foam image series.

* `bubble_segmentation.ipynb`: step-by-step pipeline on one frame, with every tunable parameter in a single CONFIG cell.
* `bubble_seg.py`: the pipeline functions (preprocessing, Cellpose-SAM, circle fitting, Hough completion, QC, batch driver).

Pipeline: overlay masking + scale-bar calibration → masked background flattening, denoising, CLAHE and a
noise-normalised rim (dark-ridge) map → static-structure mask (fibres) → Cellpose-SAM (multi-diameter passes) →
robust circle fit per mask refined on the rim map (recovers partly hidden bubbles) → optional ridge-Hough completion
for large bubbles → quality scores + duplicate suppression → CSV of x, y, r (px and µm) per bubble.

## Setup (DESY Maxwell)

```bash
module load maxwell python/3.11
python -m venv ~/venvs/foam && source ~/venvs/foam/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name foam --display-name "foam (cellpose)"
python -c "from cellpose import models; models.CellposeModel(gpu=False)"   # downloads Cellpose-SAM weights once
```
Open the notebook from the repository root (paths are relative to it) on a GPU JupyterHub session with the `foam (cellpose)` kernel.
