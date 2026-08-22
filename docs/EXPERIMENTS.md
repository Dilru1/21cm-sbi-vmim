# Running & tracking experiments

This page explains how to automate seed/noise sweeps and keep a durable history
of their results — and, importantly, **which tool does which job**. There's a
common mix-up worth clearing up first.

## Three different jobs — don't conflate them

When you ask "can I track my experiment history on GitHub via CI?", three
separate things are hiding in that sentence:

| You want to… | Right tool | What it is |
|---|---|---|
| Run the heavy jobs | **SLURM on tycho** | GPUs + your data live here. Nothing else runs experiments. |
| Compare seeds/noise effects live | **Weights & Biases** | A dashboard: each run logs its seed, noise level, and metrics; you compare dozens of runs with plots and tables. |
| Keep a permanent, versioned record | **git / GitHub** | Commit *small* result files (a manifest, a metrics CSV). GitHub then stores the history; you can `git diff` how numbers changed over time. |
| Auto-publish/refresh results | **GitHub Actions (CI)** | Reacts to a commit — e.g. rebuilds the docs site so the published results are always current. It does **not** run experiments. |

The key correction: **CI is not where experiments run or where results are
tracked.** CI runners have no GPU and can't see your cluster data. CI's role
here is only to *react* to results you commit (for example, rebuild the Jupyter
Book so the site shows your latest metrics table). The actual "history of
effects" lives in **W&B** (interactive) and in **committed result files**
(permanent, diffable). Both are better tools for that than CI.

## The end-to-end flow

```
  you (login node)                 SLURM / tycho                 GitHub
  ────────────────                 ─────────────                 ──────
  run_sweep.py  ──submit──▶  stage1→2→3 jobs run  ──logs──▶  W&B dashboard
       │                           │                          (live compare)
       │ writes                    │ each run writes
       ▼                           ▼ run_meta.json (provenance)
  experiments/manifest.csv   scratch outputs
       │
       │ git commit + push
       ▼
  GitHub stores the versioned experiment log
       │
       ▼ (optional) Actions rebuilds the docs site with the latest results
```

Two complementary records come out of this:

- **`manifest.csv`** — *what you launched*: one row per submission with the
  timestamp, git commit, config, seed, family/scope, and the SLURM job ids.
- **W&B + `run_meta.json`** — *what came out*: the metrics of each run, plus the
  exact code+config that produced them.

## Automating your seed sweep

Your five manual lines:

```bash
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=0 "nsf" "std" 1
bash submit_nle_grid.sh ... -o compressor.init_seed=1 "nsf" "std" 1
...  (seeds 2, 3, 4)
```

become one command (run on the login node):

```bash
python experiments/run_sweep.py \
    configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml \
    --seeds 0 1 2 3 4 --family nsf --scope std --stages 1 --push
```

or, interactively, open **`notebooks/run_experiments.ipynb`**, edit the config
cell, and *Run All*.

`run_sweep.py` loops the seeds, submits each with your existing
`submit_nle_grid.sh` (so the SLURM logic is unchanged), captures the job ids,
appends a row per seed to `experiments/manifest.csv`, and — with `--push` —
commits and pushes that manifest. Preview first with `--dry-run` to see the
exact `sbatch` commands without submitting.

Sweeping several **noise versions** too is just a loop over configs:

```bash
for cfg in configs_seeds/noise/arm_cnn_vmim_no_jitter_n{1,2,3,4}.yaml; do
    python experiments/run_sweep.py "$cfg" --seeds 0 1 2 3 4 \
        --family nsf --scope std --stages 1
done
git add experiments/manifest.csv && git commit -m "Log n1-n4 sweeps" && git push
```

## Where the results (metrics) come from

The manifest logs *submissions*. To log *outcomes*, once jobs finish run your
`eval.py` for each arm and keep the small `metrics.csv` it writes under version
control (they're tiny and already allow-listed in `.gitignore`). Committing
those is what turns GitHub into a diffable record of how, say, VMIM vs MSE
behaves as noise increases. If you've enabled `sbi/tracking.py`, the same
metrics also stream to W&B for interactive comparison — no extra work.

## The (optional) CI piece

Your docs site already rebuilds on every push to `main`
(`.github/workflows/deploy-book.yml`). So the moment you commit updated results,
the published Jupyter Book can show them — *that's* the legitimate role of CI
here: publishing, not computing. (A small "Results" page that reads
`experiments/manifest.csv` / `metrics.csv` and plots them is a nice next step —
ask when you're ready.)
