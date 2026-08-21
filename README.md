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

> **Status:** research code, actively developed. The pipeline runs end-to-end on a
> SLURM cluster; several analyses are still in progress.

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

---

## Quick start

A single arm, end-to-end (config-driven; edit paths in the YAML first):

```bash
# 1. train the compressor and export summaries
python stage1_compress.py configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=0

# 2. train a neural likelihood family (nsf, standardized t)
python stage2_nle.py configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o nle.model=nsf

# 3. SBC MCMC chains (sharded across array tasks on the cluster)
python stage3_mcmc.py configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml --task-num 0 --task-nb 10

# 4. evaluate + build the report
python eval.py --out eval_reports/n1 \
    --item "configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml|nsf|std|VMIM n1"
```

On the cluster, use the batch scripts in `slurm/` and the `submit_nle_grid.sh`
launcher instead of calling the stages by hand. See **[`docs/COMMANDS.md`](docs/COMMANDS.md)**
for the full set of commands used to reproduce the figures.

### Config overrides
Any config key can be overridden on the command line without editing the YAML:

```bash
python stage1_compress.py <config>.yaml -o compressor.init_seed=3 -o compressor.num_workers=0
```

---

## Configuration

Each experiment is one YAML in `configs_seeds/`. Key fields:

- `arm_name`, `arm_type` (`cnn` / `mlp`) — identity and compressor family of the arm.
- `scratch_root` — where heavy outputs are written (**cluster path, edit for your system**).
- `data.*` — paths to signal cubes, noise cubes, and parameter files.
- `compressor.*` — architecture, objective (`vmim_head` / MSE), noise scale, jitter (`dequant`), seeds.
- `nle.*` — density family and hyperparameters (shared across arms for fairness).
- `mcmc.*` — sampler settings.

---

## Reproducibility & MLOps

Every run records its provenance (git commit, config hash, SLURM ids) via
`sbi/tracking.py`, and optionally streams metrics to
[Weights & Biases](https://wandb.ai) when `wandb` is installed. Code quality is
kept with `ruff` + `pre-commit`, a torch-free `pytest` suite runs in GitHub
Actions CI, and common tasks are wrapped in a `Makefile` (`make setup`,
`make test`, `make fmt`). See **[`docs/MLOPS.md`](docs/MLOPS.md)** for the full,
phased plan and how to enable each piece.

## Citing

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" button from it).

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
