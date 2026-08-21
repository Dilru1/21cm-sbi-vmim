# MLOps for this project — a phased plan

You want an "industry-level but *suitable*" MLOps stack, and you're learning as
you go. The single most useful idea to start with:

> Most of "industry MLOps" — model serving, Kubernetes, feature stores,
> autoscaling inference, production monitoring — **does not apply here**. Your
> "product" is a paper and a set of reproducible figures, not a deployed
> service. The parts of MLOps that *do* apply are **reproducibility,
> experiment tracking, and disciplined data/pipeline management**. Adopt those;
> skip the rest until you actually have a service to run.

This document is the map. It's ordered so each phase pays off before the next
one adds complexity — resist the urge to install everything at once.

---

## The suitable stack (and what to skip)

| Concern | Tool | Verdict for this project |
|---|---|---|
| Code versioning | **Git + GitHub** | ✅ done |
| Environment pinning | **requirements.txt / environment.yml** | ✅ done |
| Reproducible seeds | **`sbi/seeding.py`** | ✅ you already have this |
| **Experiment tracking** | **Weights & Biases** (free academic tier) | ⭐ start here |
| Run provenance | **`sbi/tracking.py`** (git SHA + config hash + SLURM ids) | ✅ added, opt-in |
| Code quality | **ruff + black + pre-commit** | ✅ added |
| Testing | **pytest** (formalize your `check_*.py`) | ✅ starter added |
| Continuous integration | **GitHub Actions** | ✅ added |
| Task running | **Makefile** | ✅ added |
| Pipeline / DAG on SLURM | **Snakemake** | ◻ phase 3, skeleton added |
| Data / artifact provenance | **manifest + checksums** (DVC/DataLad optional) | ◻ phase 3 |
| Container reproducibility | **Apptainer/Singularity** | ◻ phase 4, optional |
| Hyperparameter search | **W&B Sweeps** | ◻ phase 4, optional |

**Deliberately skipped** (not suitable for a research campaign): model serving
(FastAPI/TorchServe/KServe), Kubernetes/Kubeflow, feature stores (Feast),
Airflow, production monitoring (Prometheus/Grafana), A/B testing. If this ever
becomes a public inference service, revisit — not before.

**On Gemini's list:** GitHub ✅, SLURM ✅, and W&B ✅ are spot-on. On **DVC**,
be aware of one thing generic advice omits: **DVC has no SLURM integration** —
its `dvc repro` runs stages locally, so it cannot drive your cluster jobs. For
you, DVC's only real niche is versioning *derived artifacts*, which W&B
artifacts + a data manifest already cover with less friction. So DVC is
"learn the concept, optional to adopt," not a priority. See Phase 3.

---

## Phase 0 — foundation ✅ (already in place)

Git/GitHub, pinned environment, and deterministic seeding are the bedrock of
reproducibility, and you already have all three. Nothing to do.

---

## Phase 1 — experiment tracking + provenance ⭐ (do this first)

This is the highest-value addition by far. You run a *large grid* (noise ladder
n1–n4 × jitter/no-jitter × seeds × families × scopes). Right now the record of
"which config produced which loss curve / which SBC result" lives in filenames
and your memory. Experiment tracking replaces that with a searchable dashboard.

**Why W&B here:** best-in-class UI and run comparison, a **free academic Pro
tier**, native **offline mode** for compute nodes without internet, and built-in
hyperparameter **Sweeps** for later. (MLflow is the strong open-source,
self-hosted alternative if you ever want zero vendor lock-in; Aim is a good
lightweight local-only option. Avoid Neptune — its hosted service is being shut
down.)

### How to implement (already wired, opt-in)

`sbi/tracking.py` is added and safe: if `wandb` isn't installed it does nothing
but still write a `run_meta.json` provenance file. To turn it on:

```bash
pip install wandb
wandb login                      # once, on a login node
export WANDB_PROJECT=21cm-sbi-vmim
```

Then add **three lines** to a stage script (e.g. `stage1_compress.py`):

```python
from sbi.tracking import init_run
run = init_run(cfg, stage="stage1_compress", tags=[cfg["arm_type"]])
# ... inside the epoch loop:
run.log({"epoch": epoch, "val_loss": val_loss, "rf_r2": r2})
# ... at the end:
run.log_artifact(dirs["nle"] / "best.pt", type="model")   # optional
run.finish()
```

That's it. Every run then appears in your browser with its loss curves, its
**git commit**, its **config hash**, and its **SLURM job id** attached — so any
figure is traceable back to exact code + config.

### Cluster gotcha (important)

Compute nodes on `tycho` may have no outbound internet. In your SLURM script:

```bash
export WANDB_MODE=offline        # log to disk during the job
python stage1_compress.py <cfg>
```

then, later, from a login node:

```bash
wandb sync <the-offline-run-dir>
```

The provenance `run_meta.json` is written regardless, online or off.

### Kill switch

`export SBI_TRACK=0` disables all tracking (provenance still written). Nothing
about tracking can break a training run — every call is wrapped.

---

## Phase 2 — code quality, tests, CI (cheap, great habits)

These are low-effort and teach the core "engineering discipline" half of MLOps.

1. **Formatting + linting** — `ruff` (fast, replaces flake8/isort/pyupgrade)
   and `black`-compatible formatting, configured in `pyproject.toml`.
   ```bash
   make setup      # installs ruff, pytest, pre-commit
   make fmt        # auto-format + autofix the whole repo (do this once)
   make lint       # see what's left
   ```

2. **pre-commit hooks** — run those checks automatically on every commit, plus
   a guard that blocks accidental >500 kB commits (protects you from ever
   committing a cube again):
   ```bash
   make hooks      # = pre-commit install
   ```

3. **Tests** — `tests/` has a starter suite that runs anywhere (no torch/GPU):
   import smoke test, config-override coercion, and seeding determinism (a
   pytest version of what `tools/check_seeding.py` checks by hand).
   ```bash
   make test
   ```
   Grow it by turning your other `check_*.py` diagnostics into `tests/`, marking
   the heavy ones `@pytest.mark.needs_torch` / `needs_data` so CI skips them.

4. **CI** — `.github/workflows/ci.yml` runs the torch-free tests on every push
   (the **blocking** gate) and reports lint issues **advisorily** (so a green
   badge doesn't require cleaning all 180 legacy nits first). Once you've run
   `make fmt` and the count is near zero, flip `continue-on-error` to `false`
   in the lint step to make it a hard gate.

---

## Phase 3 — reproducible pipeline + data provenance

Adopt these once Phase 1–2 are habit and the grid gets unwieldy.

### Pipeline orchestration — Snakemake (not DVC, not Airflow)

Your DAG is `compress → nle → mcmc → eval` and it runs on SLURM. **Snakemake**
is the research-standard workflow tool that (a) submits to SLURM via an
executor plugin and (b) reruns only stages whose inputs changed.
`workflow/Snakefile` is a commented skeleton — start with `snakemake -n`
(dry-run) to see the plan, then `snakemake --executor slurm -j 20`. You don't
have to migrate; your `slurm/*.sbatch` + `submit_nle_grid.sh` already encode
this imperatively. Move to Snakemake when "which jobs do I need to rerun after
this change?" starts costing you real time.

### Data / artifact provenance — start with a manifest

Your raw 21 cm cubes are large, static, and generated once — full data
versioning is overkill. What you actually need is to **record exactly which
inputs produced a result**. Cheapest high-value step: a committed manifest.

```bash
# one-off: record checksums + sizes of the inputs a campaign used
sha256sum /data/.../clean_cubes_z=*.dat /data/.../astro_params_*.npy \
  > docs/data_manifest.sha256
```

Commit that file. Now any result is tied to verifiable input hashes, with zero
new tooling. **DVC / DataLad** give you true rollback-able data versioning if
you later need it — worth *learning* — but note again DVC won't run your SLURM
jobs, so its role for you is storage/versioning of artifacts, layered under
Snakemake, not instead of it.

---

## Phase 4 — advanced / optional

- **Apptainer/Singularity** — a container that freezes CUDA + Python + libs for
  bit-level environment reproducibility on HPC (clusters run Apptainer, not
  Docker). The gold standard for "runs identically in 3 years," but a real
  time investment. Do it when you're preparing the paper's reproducibility
  package.
- **W&B Sweeps** — you already sweep seeds/noise/families by hand; Sweeps
  formalize that into a searchable, resumable hyperparameter study.
- **Config framework (Hydra)** — *don't rush this.* Your YAML + `-o` dotted
  override system is already good. Adopt Hydra only if config composition
  starts hurting; it's a nontrivial refactor.

---

## Suggested order of attack

1. `pip install wandb`, wire the 3 tracking lines into `stage1_compress.py`,
   run one job, watch it appear in the dashboard. **(biggest payoff)**
2. `make setup && make hooks && make fmt`, commit the formatting pass.
3. `make test` locally; push and watch CI go green.
4. Convert one more `check_*.py` into a `tests/` test.
5. Add `docs/data_manifest.sha256`.
6. Only then: try the Snakemake dry-run.
7. Later, when writing up: Apptainer image + W&B Sweeps.

## Learning resources

- W&B quickstart: https://docs.wandb.ai/quickstart
- ruff: https://docs.astral.sh/ruff/ · pre-commit: https://pre-commit.com
- pytest: https://docs.pytest.org
- Snakemake + SLURM: https://snakemake.readthedocs.io and the
  `snakemake-executor-plugin-slurm` docs
- "The Turing Way" (open, excellent on reproducible research):
  https://the-turing-way.netlify.app
