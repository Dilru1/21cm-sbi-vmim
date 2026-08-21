#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# apply_mlops.sh — add the Phase 1–3 MLOps scaffold to the repo.
#
# Run from the ROOT of your repo, AFTER extracting mlops-additions.zip there:
#   cd /path/to/sbi_project
#   unzip /path/to/mlops-additions.zip -d .
#   bash apply_mlops.sh
#
# Adds: sbi/tracking.py, tests/, pyproject.toml, .pre-commit-config.yaml,
#       .github/workflows/ci.yml, Makefile, .env.example, workflow/Snakefile,
#       docs/MLOPS.md, and an updated README.md. Makes ONE commit. Does NOT push.
#
# Independent of apply_updates.sh — safe to run whether or not you ran that one.
# ---------------------------------------------------------------------------
set -euo pipefail

[ -d .git ] || { echo "ERROR: run from your repo root (no .git here)." >&2; exit 1; }
[ -f sbi/tracking.py ] && [ -f docs/MLOPS.md ] || {
  echo "ERROR: additions missing. Extract mlops-additions.zip into this folder first." >&2
  exit 1; }

echo ">> repo:   $(git config --get remote.origin.url || echo '(no origin)')"
echo ">> branch: $(git rev-parse --abbrev-ref HEAD)"

# Optional but recommended: do the MLOps work on a feature branch.
if [ "${MLOPS_BRANCH:-1}" = "1" ]; then
  git switch -c mlops-scaffold 2>/dev/null || git switch mlops-scaffold
  echo ">> on branch mlops-scaffold (merge to main when happy; set MLOPS_BRANCH=0 to skip)"
fi

git add \
  sbi/tracking.py \
  tests/__init__.py tests/test_smoke.py tests/test_config.py tests/test_seeding.py \
  pyproject.toml .pre-commit-config.yaml .env.example Makefile \
  .github/workflows/ci.yml workflow/Snakefile \
  docs/MLOPS.md README.md

git commit --quiet -m "Add MLOps scaffold: tracking, tests, CI, tooling, docs

- sbi/tracking.py: opt-in W&B experiment tracking + always-on run provenance
  (git SHA, config hash, SLURM ids); safe no-op without wandb, offline-friendly
- tests/: torch-free pytest suite (imports, config overrides, seeding determinism)
- pyproject.toml: ruff/black/pytest config; .pre-commit-config.yaml: git hooks
- .github/workflows/ci.yml: tests as blocking gate, lint advisory
- Makefile: make setup/test/fmt/lint/hooks; .env.example: W&B + kill-switch
- workflow/Snakefile: optional SLURM-aware DAG skeleton
- docs/MLOPS.md: phased plan; README: reproducibility section" \
  && echo ">> committed MLOps scaffold"

echo ""
echo "Next:"
echo "  make setup && make test        # verify locally (installs ruff/pytest)"
echo "  git push -u origin $(git rev-parse --abbrev-ref HEAD)"
echo "  # then, to start tracking:  pip install wandb && wandb login"
echo "  # see docs/MLOPS.md for the 3-line integration and the ordered plan."
