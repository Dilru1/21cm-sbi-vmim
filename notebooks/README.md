# Notebooks

Interactive **diagnostics and reporting** for the pipeline. The heavy compute
(stages 1–3) runs as SLURM jobs on the cluster; these notebooks only read the
outputs and draw figures.

## How to run

Run them **on `tycho`** (via the VS Code tunnel) so they can see scratch data:

1. Open `diagnostics.ipynb` in VS Code connected to the cluster.
2. Edit the **Configuration** cell (arm paths, config, family/scope).
3. *Run All*. Each section calls the same script you'd run on the command line
   (`tools/plot_training_compressor.py`, `tools/plot_training_nle.py`,
   `eval.py`) and shows the result inline.

Generated figures are written under `notebooks/_figs/` (git-ignored).

## Committing notebooks

- **`diagnostics.ipynb` is a report** — commit it **with its outputs** so the
  figures appear in the published Jupyter Book.
- For any future *scratch / exploration* notebooks where outputs are just noise,
  strip them before committing to keep git clean:
  ```bash
  pip install nbstripout
  nbstripout notebooks/scratch/*.ipynb      # strip on demand
  ```
  (Don't run nbstripout on the report notebooks — it would erase their figures.)
