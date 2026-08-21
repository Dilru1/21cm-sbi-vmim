#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# apply_updates.sh — complete the 21cm-sbi-vmim repo.
#
# Run this from the ROOT of your existing repo (the folder that contains .git,
# i.e. your sbi_project checkout on the cluster), AFTER extracting
# repo-additions.zip into that same folder.
#
#   cd /path/to/sbi_project
#   unzip /path/to/repo-additions.zip -d .      # drops README.md, LICENSE, docs/, etc.
#   bash apply_updates.sh
#
# It does NOT push. It stages + commits in two logical commits, then prints the
# exact push command. Review with `git log` / `git diff` before pushing.
# ---------------------------------------------------------------------------
set -euo pipefail

# 0. sanity checks -----------------------------------------------------------
if [ ! -d .git ]; then
  echo "ERROR: no .git here. cd into your repo root first." >&2; exit 1
fi
if [ ! -f README.md ] || [ ! -f LICENSE ] || [ ! -f docs/COMMANDS.md ]; then
  echo "ERROR: additions not found. Extract repo-additions.zip into this folder first." >&2; exit 1
fi
echo ">> repo: $(git config --get remote.origin.url || echo '(no origin)')"
echo ">> branch: $(git rev-parse --abbrev-ref HEAD)"

# 1. stop tracking churny / generated files (kept on disk) --------------------
if git ls-files --error-unmatch logs >/dev/null 2>&1 || git ls-files 'logs/*' | grep -q .; then
  git rm -r --cached --quiet logs 2>/dev/null || true
  echo ">> untracked logs/"
fi
# all __pycache__ dirs anywhere in the tree
git ls-files | grep '__pycache__/' | sed 's/[^/]*$//' | sort -u | while read -r d; do
  [ -n "$d" ] && git rm -r --cached --quiet "$d" 2>/dev/null || true
done
echo ">> untracked __pycache__/"

# 2. retire the old command-scratchpad readme (content now in docs/COMMANDS.md)
if git ls-files --error-unmatch readme.md >/dev/null 2>&1; then
  git rm --quiet readme.md
  echo ">> retired old readme.md -> docs/COMMANDS.md"
fi

# 3. commit A: hygiene -------------------------------------------------------
git add .gitignore .gitattributes
git commit --quiet -m "Repo hygiene: stop tracking logs/ and __pycache__, expand .gitignore

- git rm --cached the SLURM logs/ tree (churns every run) and all
  __pycache__ dirs (were committed before .gitignore existed)
- rewrite .gitignore with Python/editor/cluster-log/data rules plus a
  curated allow-list for small tracked outputs (SBC_OUT, metrics.*)
- add .gitattributes for consistent LF line endings and binary handling" \
  && echo ">> commit A (hygiene) done"

# 4. commit B: docs + packaging ---------------------------------------------
git add README.md docs/COMMANDS.md LICENSE CITATION.cff requirements.txt environment.yml
git commit --quiet -m "Add README, license, citation, and dependency manifests

- replace command-scratchpad readme.md with a proper project README
- preserve the original scratchpad as docs/COMMANDS.md
- add MIT LICENSE, CITATION.cff, requirements.txt, environment.yml" \
  && echo ">> commit B (docs) done"

# 5. done --------------------------------------------------------------------
echo ""
echo "Local commits ready:"
git log --oneline -4
echo ""
echo "Review, then push with:"
echo "    git push origin $(git rev-parse --abbrev-ref HEAD)"
echo ""
echo "NOTE: your in-progress change to submit_nle_grid.sh (if any) was left"
echo "      untouched and uncommitted — commit it separately when ready."
