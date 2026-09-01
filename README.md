# 21cm-sbi-vmim

Simulation-based inference (SBI) of reionization-era astrophysical parameters from
the 21 cm signal, using **learned summary statistics** and **neural likelihood
estimation (NLE)**.

The project asks a focused methodological question: *does compressing 21 cm
lightcones with a **VMIM**-trained (Variational Mutual Information Maximization)
neural compressor give better, better-calibrated posteriors than the usual
MSE-trained compressor or a fixed hand-made summary (power spectrum / PDF)?* —
and how robust that answer is to instrumental **thermal noise** and to
**dequantization jitter** during training.


---

## What the pipeline does

Four stages, each a standalone script, sharing one config file per "arm"
(a compressor + inference variant):

| Stage | Script | Role |
|-------|--------|------|
| 1 | `stage1_compress.py` / `stage1_raw.py` | Train the compressor `t = F(x)` on 21 cm cubes and export summaries. `stage1_raw.py` is the no-compression baseline. |
| 2 | `stage2_nle.py` | Train one or more neural likelihood families `q(t \| θ)` (GMM / MAF / NSF) on the exported summaries. |
| 3 | `stage3_mcmc.py` | Run SBC MCMC chains — one posterior per held-out simulation. |
| 4 | `eval.py` | Aggregate into SBC rank histograms, coverage/GV metrics, corner plots, and LaTeX tables. |

The design goal is a **fair comparison**: every arm shares the *identical* stage-2/3/4
machinery, so any difference in the final posteriors is attributable to the summary
statistic alone.

### Compressor objectives compared
- **VMIM** — compressor trained to maximize mutual information with the parameters via a variational density head.
- **MSE** — compressor trained to regress the parameters directly.
- **Baseline** — fixed summaries (power spectrum, PDF, or both) with no learned compression.

### Axes of the study
- **Noise ladder** `n1…n4` — increasing thermal-noise levels (100 h → 6.25 h integration).
- **Jitter vs no-jitter** — dequantization noise added to parameters during compressor training.
- **Seeds** — multiple `init_seed` runs per config for variance estimates.

---

## Repository layout

```
sbi_project/
├── stage1_compress.py       # Stage 1: train + export learned compressor
├── stage1_raw.py            # Stage 1: no-compression baseline
├── stage2_nle.py            # Stage 2: neural likelihood estimation
├── stage3_mcmc.py           # Stage 3: SBC MCMC sampling
├── eval.py                  # Stage 4: metrics, SBC, corner plots, LaTeX tables
├── sbi/                     # Core library
│   ├── config.py            #   YAML config loading + arm directory layout
│   ├── data.py              #   data / summary loading, sim-level splits
│   ├── cubes.py             #   21 cm cube handling
│   ├── nle.py               #   neural likelihood models (GMM/MAF/NSF via nflows)
│   ├── mcmc.py              #   MCMC sampler
│   ├── train_comp_update.py #   compressor training loop (VMIM/MSE)
│   ├── seeding.py           #   reproducible seeding utilities
│   └── compressors/         #   CNN / MLP compressor architectures + density heads
├── configs_seeds/           # Experiment configs (noise ladder, seeds, mse/vmim/mlp)
├── slurm/                   # SLURM batch scripts for each stage
├── tools/                   # Plotting + diagnostics (training curves, latent space, SBC)
├── submit_nle_grid.sh       # Convenience launcher for stage-2 family/scope grids
├── tests/                   # torch-free pytest suite (imports, config, seeding)
├── workflow/Snakefile       # optional Snakemake DAG (compress→nle→mcmc→eval)
├── docs/COMMANDS.md         # Lab-notebook of exact commands used on the cluster
├── docs/MLOPS.md            # Phased MLOps plan (tracking, CI, reproducibility)
├── requirements.txt         # Python dependencies
├── pyproject.toml           # ruff / black / pytest config
└── environment.yml          # Conda environment (GPU)
```

Heavy artifacts (raw cubes, `.npy`/`.pt`/`.h5` summaries, checkpoints, chains) live
on cluster scratch (`scratch_root` in each config) and are **not** tracked in git.

---

## Installation

The code targets **Python 3.10** with a CUDA-capable GPU for stages 1–2.

Conda (matches the cluster `torch_gpu` env):

```bash
conda env create -f environment.yml
conda activate torch_gpu
```

or pip into an existing environment:

```bash
pip install -r requirements.txt
```

> `torch` should be installed with the CUDA build matching your driver
> (see https://pytorch.org). `requirements.txt` lists a plain `torch` for
> portability; on the cluster it is provided by the module system + conda env.

