#!/usr/bin/env python3
"""Submit a seed sweep of the pipeline and record what was launched.

Run this ON the cluster login node -- it calls `sbatch` (via
`submit_nle_grid.sh`), so it needs SLURM. It does NOT run on your laptop, Colab,
or GitHub. What it does:

  1. loops over the seeds you give it,
  2. submits each one with `submit_nle_grid.sh` (capturing the SLURM job ids),
  3. appends one row per submission to `experiments/manifest.csv`,
  4. optionally commits + pushes that manifest so GitHub holds a versioned
     history of every experiment you launched.

The jobs themselves run asynchronously in SLURM; this script returns as soon as
they're submitted (it does not wait for training to finish).

Examples
--------
Replace your five manual lines with one command:

    python experiments/run_sweep.py \
        configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml \
        --seeds 0 1 2 3 4 --family nsf --scope std --stages 1 --push

Dry run (print what it *would* submit, touch nothing):

    python experiments/run_sweep.py <config> --seeds 0 1 2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
MANIFEST = Path(REPO) / "experiments" / "manifest.csv"
FIELDS = [
    "timestamp",
    "git_sha",
    "config",
    "arm_name",
    "seed",
    "family",
    "scope",
    "stages",
    "job_ids",
    "user",
    "note",
]


def git_sha() -> str:
    sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
    )
    dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO)
    return sha + ("-dirty" if dirty else "")


def arm_name(config: str) -> str:
    for line in Path(config).read_text().splitlines():
        if line.strip().startswith("arm_name:"):
            return line.split(":", 1)[1].strip()
    return ""


def submit_one(config, seed, family, scope, stages, dry_run=False):
    cmd = [
        "bash",
        "submit_nle_grid.sh",
        config,
        "-o",
        f"compressor.init_seed={seed}",
        family,
        scope,
        str(stages),
    ]
    print(">>", " ".join(cmd), flush=True)
    if dry_run:
        return []
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit(f"submit failed for seed {seed}")
    # SLURM job ids are the 5+ digit numbers submit_nle_grid.sh printed.
    return re.findall(r"\b(\d{5,})\b", out.stdout)


def append_row(row: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit a seed sweep and log it.")
    ap.add_argument("config", help="path to the arm's config YAML")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="e.g. --seeds 0 1 2 3 4")
    ap.add_argument("--family", default="nsf", help="gmm | maf | nsf")
    ap.add_argument("--scope", default="std", help="std | raw")
    ap.add_argument("--stages", default="123", help="e.g. 1, 23, 123")
    ap.add_argument("--note", default="", help="free-text note stored in the manifest")
    ap.add_argument(
        "--push", action="store_true", help="git commit + push the manifest after submitting"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the sbatch commands without submitting"
    )
    a = ap.parse_args()

    sha = git_sha()
    arm = arm_name(a.config)
    user = os.environ.get("USER", "")
    print(f"config={a.config}  arm={arm}  git={sha}")
    print(f"seeds={a.seeds}  family={a.family}  scope={a.scope}  stages={a.stages}\n")

    for seed in a.seeds:
        ids = submit_one(a.config, seed, a.family, a.scope, a.stages, a.dry_run)
        if a.dry_run:
            continue
        append_row(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "git_sha": sha,
                "config": a.config,
                "arm_name": arm,
                "seed": seed,
                "family": a.family,
                "scope": a.scope,
                "stages": a.stages,
                "job_ids": " ".join(ids),
                "user": user,
                "note": a.note,
            }
        )

    if a.dry_run:
        print("\ndry run: nothing submitted, manifest untouched.")
        return

    print(f"\nrecorded {len(a.seeds)} submissions -> {MANIFEST}")

    if a.push:
        subprocess.run(["git", "add", str(MANIFEST)], cwd=REPO, check=True)
        msg = f"Log sweep: {arm} {a.family}/{a.scope} seeds {a.seeds} stages={a.stages} @ {sha}"
        subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=True)
        if subprocess.call(["git", "push"], cwd=REPO) == 0:
            print("pushed manifest to GitHub")
        else:
            print(
                "commit made, but `git push` failed -- push it yourself once "
                "your remote/auth is set up:  git push"
            )
    else:
        print("(not pushed; re-run with --push, or commit experiments/manifest.csv yourself)")


if __name__ == "__main__":
    main()
