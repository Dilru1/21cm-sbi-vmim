"""Lightweight, opt-in experiment tracking + run provenance.

Design goals
------------
* **Additive and safe.** Import never fails and never pulls in torch. If
  ``wandb`` is not installed, or tracking is disabled, every call is a no-op
  that returns a dummy handle -- so existing runs keep working untouched.
* **Provenance always.** Even with no wandb and no internet (compute nodes),
  ``init_run`` writes a ``run_meta.json`` into the arm's output dir capturing
  the git commit, a config hash, hostname, and SLURM ids. That file alone lets
  you answer "which code + config produced this result?" months later.
* **Cluster-friendly.** On an HPC compute node with no outbound network, set
  ``WANDB_MODE=offline`` (see docs/MLOPS.md); runs log to disk and you
  ``wandb sync`` them later from a login node. This helper respects that.

Usage (add ~3 lines to a stage script)
--------------------------------------
    from sbi.tracking import init_run

    run = init_run(cfg, stage="stage1_compress", tags=["cnn", "vmim"])
    ...
    run.log({"epoch": e, "val_loss": v})     # inside the training loop
    ...
    run.log_artifact(dirs["nle"] / "best.pt", type="model")   # optional
    run.finish()

Disable entirely with ``SBI_TRACK=0`` in the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

__all__ = ["init_run", "RunHandle"]


def _git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        sha = out.decode().strip()
        # mark a dirty working tree so a result is never silently attributed
        # to a clean commit it wasn't built from
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return default


def _config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _slurm_ids() -> dict:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_NNODES",
        "SLURM_GPUS_ON_NODE",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}


def _tracking_enabled() -> bool:
    return os.environ.get("SBI_TRACK", "1") not in ("0", "false", "False", "")


class RunHandle:
    """Uniform handle whether or not wandb is active. All methods are safe."""

    def __init__(self, wandb_run=None, meta_path: Path | None = None):
        self._run = wandb_run
        self.meta_path = meta_path

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, data: dict, step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(data, step=step)

    def log_artifact(self, path, name: str | None = None, type: str = "artifact") -> None:
        if self._run is None:
            return
        try:
            import wandb

            art = wandb.Artifact(name or Path(path).stem, type=type)
            art.add_file(str(path))
            self._run.log_artifact(art)
        except Exception as e:  # never let logging crash a training run
            print(f"[track] artifact log skipped: {e}", flush=True)

    def summary(self, data: dict) -> None:
        if self._run is not None:
            self._run.summary.update(data)

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass


def init_run(cfg: dict, stage: str, tags=None, notes: str = "") -> RunHandle:
    """Start a tracked run and write provenance. Returns a safe RunHandle.

    Always writes ``{scratch_root}/{arm_name}/run_meta.json``. Starts a wandb
    run in addition, iff wandb is importable and SBI_TRACK != 0.
    """
    meta = {
        "stage": stage,
        "arm_name": cfg.get("arm_name"),
        "git_sha": _git_sha(),
        "config_hash": _config_hash(cfg),
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "slurm": _slurm_ids(),
        "config": cfg,
    }

    # --- provenance file (always, even offline / no wandb) ---
    meta_path = None
    try:
        root = Path(cfg["scratch_root"]) / cfg["arm_name"]
        root.mkdir(parents=True, exist_ok=True)
        meta_path = root / "run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    except Exception as e:
        print(f"[track] could not write run_meta.json: {e}", flush=True)

    # --- optional wandb run ---
    if not _tracking_enabled():
        return RunHandle(None, meta_path)
    try:
        import wandb
    except Exception:
        print(
            "[track] wandb not installed -> provenance only "
            "(pip install wandb to enable dashboards)",
            flush=True,
        )
        return RunHandle(None, meta_path)

    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "21cm-sbi-vmim"),
            group=str(cfg.get("arm_name", "")).split("/")[0] or None,
            name=f"{cfg.get('arm_name')}::{stage}",
            job_type=stage,
            tags=tags or [],
            notes=notes,
            config={
                **cfg,
                "git_sha": meta["git_sha"],
                "config_hash": meta["config_hash"],
                "slurm": meta["slurm"],
            },
        )
        return RunHandle(run, meta_path)
    except Exception as e:
        print(f"[track] wandb.init failed ({e}); provenance only", flush=True)
        return RunHandle(None, meta_path)
