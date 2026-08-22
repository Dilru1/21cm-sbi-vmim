#!/usr/bin/env python3
"""Validate and summarize the experiment log (experiments/manifest.csv).
This is the concrete answer to "how does CI track completed experiments?" CI
cannot watch your training (no GPU, no cluster data). What it CAN do is react to
the *record* you commit: when you push an updated manifest, this script runs in
GitHub Actions to (1) VALIDATE the log is well-formed -- a real "test" of your
experiment metadata -- and (2) print a SUMMARY you can read in the Actions tab.
It also runs locally:  python experiments/summarize.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

MANIFEST = Path("experiments/manifest.csv")
REQUIRED = [
    "timestamp",
    "git_sha",
    "config",
    "arm_name",
    "seed",
    "family",
    "scope",
    "stages",
    "job_ids",
]


def emit(md: str) -> None:
    """Print, and also append to the GitHub Actions run summary if present."""
    print(md)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a") as f:
            f.write(md + "\n")


def main() -> int:
    if not MANIFEST.exists():
        emit("No experiments/manifest.csv yet — nothing to summarize.")
        return 0

    df = pd.read_csv(MANIFEST)

    # ---- VALIDATION (the "test": fail red if the log is malformed) ----
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    else:
        if df["seed"].isna().any():
            problems.append("some rows have an empty seed")
        if df["job_ids"].astype(str).str.strip().eq("").any():
            problems.append("some rows have no SLURM job id (submission may have failed)")
        dups = df["job_ids"].astype(str)
        if dups.duplicated().any():
            problems.append("duplicate job_ids found (same job logged twice?)")

    # ---- SUMMARY ----
    emit("## Experiment log summary\n")
    emit(f"- **Total submissions:** {len(df)}")
    if not missing:
        emit(f"- **Distinct arms:** {df['arm_name'].nunique()}")
        emit(f"- **Latest submission:** {df['timestamp'].max()}\n")

        # seed coverage per (config, family, scope)
        emit("### Runs per config × family × scope\n")
        emit("| config | family | scope | seeds | # runs |")
        emit("|---|---|---|---|---|")
        grp = df.groupby(["config", "family", "scope"])
        for (cfg, fam, sc), g in grp:
            seeds = ",".join(str(s) for s in sorted(g["seed"].unique()))
            cfg_short = Path(str(cfg)).name
            emit(f"| {cfg_short} | {fam} | {sc} | {seeds} | {len(g)} |")

    # ---- verdict ----
    if problems:
        emit("\n### ❌ Validation failed\n")
        for p in problems:
            emit(f"- {p}")
        return 1
    emit("\n### ✅ Manifest looks well-formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
