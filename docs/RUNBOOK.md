---
title: End-to-end runbook
---

# End-to-end runbook: from experiment to published, tested result

This is the whole loop in one place, using a concrete example: **run 2 seeds of
one arm, track them, check W&B, commit, and confirm with tests.** It ties
together every tool in the repo.

## First, the mental model (4 tools, 4 jobs)

Keep these separate — most confusion comes from expecting one tool to do
another's job:

```{mermaid}
flowchart LR
  subgraph tycho["tycho (login node + SLURM)"]
    NB[run_experiments.ipynb] -->|submits| J[stage1 to 3 jobs]
  end
  J -->|metrics, live| WB[(Weights & Biases<br/>dashboard = TRACK)]
  NB -->|writes| MAN[experiments/manifest.csv]
  J -->|eval.py| MET[metrics.csv + figures]
  MAN -->|git commit + push| GH[(GitHub = RECORD)]
  MET -->|git commit + push| GH
  GH -->|on push| CI[GitHub Actions<br/>= TEST + PUBLISH]
  CI --> SITE[Jupyter Book site]
```

| Tool | Its one job | Not its job |
|---|---|---|
| **SLURM / tycho** | run the heavy compute | — |
| **Weights & Biases** | *track* & compare runs (metrics, curves) | running jobs |
| **git / GitHub** | *record* small results (manifest, metrics) — versioned | storing data/checkpoints |
| **GitHub Actions (CI)** | *test* code on push + *publish* the site | running experiments or "tracking results" |

**The correction to remember:** CI does **not** track your experiment results —
it tests your *code* and publishes your *site*. Tracking lives in W&B (live) and
in the small files you commit (permanent).

---

## Part 0 — one-time setup (do once)

A checklist; each maps to a kit you've already applied.

1. **Dev tools + hooks**
   ```bash
   make setup        # installs deps + ruff/pytest/pre-commit
   make hooks        # turns on pre-commit (format/hygiene on every commit)
   ```
2. **Weights & Biases account** (free academic tier), logged in on the login node:
   ```bash
   pip install wandb
   wandb login                       # once
   export WANDB_PROJECT=21cm-sbi-vmim
   ```
3. **Wire tracking into the stage scripts** (only needed once, if not done). In
   `stage1_compress.py` and `stage2_nle.py`, add three lines:
   ```python
   from sbi.tracking import init_run
   run = init_run(cfg, stage="stage1_compress", tags=[cfg["arm_type"]])
   # ... in the training loop:  run.log({"epoch": e, "val_loss": v})
   # ... at the end:            run.finish()
   ```
   Without this, sweeps still run and the manifest still records — you just won't
   see runs on the W&B dashboard.
4. **Confirm CI + Pages are on**: the repo has `.github/workflows/ci.yml`
   (tests) and `deploy-book.yml` (site); GitHub → Settings → Pages → Source =
   "GitHub Actions".

---

## Part 1 — the run loop (2 seeds), step by step

### 1. Start clean
```bash
cd ~/21cm/sbi_project
git pull origin main
```

### 2. Test *before* you spend GPU time
Catch broken code locally before launching a 6-hour job:
```bash
make test
```

### 3. Launch the 2-seed sweep (on tycho's login node)
Open **`notebooks/run_experiments.ipynb`**, set the config cell to:
```python
CONFIG = "configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml"
SEEDS  = [0, 1]      # <-- two seeds
FAMILY = "nsf"; SCOPE = "std"; STAGES = "1"; PUSH = False
```
Run All. (Equivalent one-liner: `python experiments/run_sweep.py <config>
--seeds 0 1 --family nsf --scope std --stages 1`.) This submits the SLURM jobs
and appends two rows to `experiments/manifest.csv`.

> **Cluster + W&B gotcha:** compute nodes usually have no internet. In your
> SLURM scripts set `export WANDB_MODE=offline` so runs log to disk during the
> job; you'll sync them in step 5.

### 4. Monitor
```bash
squeue -u $USER          # watch the jobs; or the notebook's last cell
```

### 5. When jobs finish — sync + check the W&B dashboard
From the **login node** (which has internet):
```bash
wandb sync <offline-run-dirs>     # uploads the offline logs
```
Open wandb.ai → project `21cm-sbi-vmim`. Compare the two seeds: loss curves,
validation R², etc. This is your "track the effects" view.

### 6. Evaluate — produce the metrics + figures
Open **`notebooks/diagnostics.ipynb`**, point it at the arm, Run All. It calls
your `eval.py` and writes `metrics.csv`, `metrics.tex`, and the SBC/corner PDFs,
and shows them inline.

### 7. Commit the record (submissions + outcomes)
```bash
git add experiments/manifest.csv <arm_out>/metrics.csv
git commit -m "n4 no-jitter nsf/std: seeds 0-1 (metrics + manifest)"
git push origin main
```
`pre-commit` runs here (auto-format/hygiene). If it tweaks a file, `git add -u`
and commit again — that's normal.

### 8. CI takes over automatically (this is the "GitHub CI workflow")
On that push, GitHub Actions:
- **runs your tests** (`ci.yml`) → a green ✓ confirms your pipeline code still
  imports, config overrides work, seeding is deterministic;
- **rebuilds the docs site** (`deploy-book.yml`) → your published Jupyter Book
  reflects the latest committed results.

### 9. Verify on GitHub
Open the repo's **Actions** tab: both workflows should show green ✓. A red ✗
means a test failed or the book build broke — click in to see which. Your site
updates at `https://Dilru1.github.io/21cm-sbi-vmim/`.

---

## Part 2 — where "testing like real MLOps" fits

Testing happens at three moments, each catching a different failure:

1. **Before commit (you):** `make test` — fast, torch-free checks (imports,
   config overrides, seeding determinism). Run before launching or committing.
2. **At commit (automatic):** `pre-commit` — formatting + hygiene + a guard that
   blocks accidental large-file commits.
3. **On push (automatic, the CI gate):** GitHub Actions re-runs the tests on a
   clean machine, so a green check on the PR/commit means "the code is still
   sound on a fresh checkout, not just on my machine."

Two honest clarifications:
- CI tests your **code**, not your **science**. A green check means the pipeline
  didn't break — *not* that VMIM beats MSE. The scientific "tests" are your
  **SBC calibration** and **seeding determinism** checks.
- CI can't run the training (no GPU, no data). That's by design; SLURM does that.

---

## Part 3 — what you might be missing

Ordered by value:

1. **Close the loop by committing *outcomes*, not just submissions.** The
   manifest logs what you launched; you also need the small `metrics.csv` from
   `eval.py` committed, so GitHub records the *results*. (Heavy data/checkpoints
   stay on scratch — never commit those.)
2. **Make W&B-offline-then-sync a habit** on tycho, or you'll have empty
   dashboards. Put `export WANDB_MODE=offline` in the SLURM scripts.
3. **Wire `tracking.py` into every stage** you want on the dashboard (step 0.3).
4. **Input provenance.** One-off, record the cube/param checksums so a result is
   tied to verifiable inputs:
   ```bash
   sha256sum /data/.../clean_cubes* /data/.../astro_params* > docs/data_manifest.sha256
   git add docs/data_manifest.sha256 && git commit -m "Record input data checksums"
   ```
5. **Tag paper milestones** so any figure is reproducible from a fixed commit:
   ```bash
   git tag -a v0.1-thesis-draft -m "state for draft" && git push --tags
   ```
6. **A "Results" page in the book** that reads `manifest.csv` / `metrics.csv` and
   plots the seed/noise comparison — turns the site into a living results view.
7. **(Optional) papermill CI smoke-test** of notebooks on tiny synthetic data,
   so they don't silently rot.
8. **(Optional) nbstripout** for scratch/exploration notebooks (keep outputs on
   report notebooks so the book shows figures).

You are *not* missing: model serving, Kubernetes, Airflow, a feature store, or
GPU CI. Those belong to product ML, not a research campaign — skipping them is
correct.

---

## The loop in one line each

`git pull` → `make test` → launch sweep (notebook, on tycho) → `squeue` →
`wandb sync` + check dashboard → run eval → commit manifest + metrics → `git
push` → CI tests + republishes → verify green on the Actions tab.
