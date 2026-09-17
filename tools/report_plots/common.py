"""Shared loaders for Chapter 3 figures.

Two data representations of the same observation (the 'two paths' of the
report):
  Path A  hand-crafted summaries : noised power spectrum + PDF linear moments
          (text files, one row per (sim, noise) pair, sim-major order)
  Path B  raw clean cubes        : 3 x 32^3 binary .dat cubes + noise bank

All loaders return raw physical units; normalisation belongs to the pipeline,
not to descriptive statistics.
"""
from __future__ import annotations

import numpy as np

N = 32
VOX = N ** 3
NB_SIMU_TOTAL = 9827          # cubes kept by the pipeline (cubes.py [:9827])
REDSHIFTS = [8.18, 10.32, 12.06]

# power spectrum binning of Semelin+25: 8 log bins, sqrt(2) steps, 0.03-0.5 h/cMpc
PS_EDGES = 0.03 * np.sqrt(2.0) ** np.arange(9)          # 0.03 ... ~0.48
PS_KCENTERS = np.sqrt(PS_EDGES[:-1] * PS_EDGES[1:])     # geometric centres
PS_NBINS = 8
PS_KEPT = slice(2, 7)          # inference keeps 5 bins: drop 2 lowest + highest
LMOM_NCOLS = 6                 # l1..l6 stored; inference uses l2..l6
LMOM_KEPT = slice(1, 6)


# --------------------------------------------------------------- text tables
def read_summary_table(path, ncols, n_noise=1000, max_sims=None, rng=None,
                       noise_stride=1):
    """Read one lmom_/noised_ps_ text file into (n_sims, n_noise_kept, ncols).

    Files store one whitespace-separated row per (sim, noise) pair in
    sim-major order (legacy view_*_distrib.py convention). np.fromfile with
    sep=' ' parses ~10^7 rows far faster than loadtxt.

    noise_stride : keep every k-th noise realisation (memory control)
    max_sims     : randomly sub-sample simulations AFTER reshaping
    """
    flat = np.fromfile(path, dtype=np.float64, sep=" ")
    if flat.size % ncols:
        raise ValueError(f"{path}: {flat.size} values not divisible by {ncols}")
    rows = flat.reshape(-1, ncols)
    if rows.shape[0] % n_noise:
        raise ValueError(f"{path}: {rows.shape[0]} rows not divisible by "
                         f"n_noise={n_noise}")
    arr = rows.reshape(-1, n_noise, ncols)[:, ::noise_stride, :]
    n_sims_file = arr.shape[0]
    sel = np.arange(n_sims_file)
    if max_sims and max_sims < n_sims_file:
        sel = np.sort(rng.choice(n_sims_file, max_sims, replace=False))
        arr = arr[sel]
    return arr.astype(np.float32), sel, n_sims_file


def load_params(params_path, simids_path):
    """Masked raw params (n,5) + the original sim ids they belong to."""
    params = np.load(params_path).astype(np.float64)
    sim_ids = np.load(simids_path).astype(int)
    assert len(params) == len(sim_ids)
    return params, sim_ids


def align_params_to_table(params, sim_ids, table_sel, n_sims_in_file):
    """Map masked params onto rows of a summary table indexed 0..n_file-1
    by ORIGINAL sim id. Rows of unavailable sims get NaN params.

    Returns params_per_row (len(table_sel), 5) and a validity mask.
    """
    full = np.full((n_sims_in_file, params.shape[1]), np.nan)
    ok = sim_ids < n_sims_in_file
    full[sim_ids[ok]] = params[ok]
    out = full[table_sel]
    return out, ~np.isnan(out[:, 0])


# ------------------------------------------------------------- binary cubes
def load_cubes(clean_tpl, noise_tpl, params_path, simids_path,
               max_sims, noise_scale, rng, n_noise_keep=200):
    """Memmap the cube .dat files; pull only a subsample of sims into RAM."""
    params, sim_ids = load_params(params_path, simids_path)
    n_avail = len(sim_ids)
    keep = (np.sort(rng.choice(n_avail, max_sims, replace=False))
            if max_sims and max_sims < n_avail else np.arange(n_avail))
    params = params[keep]
    sel_ids = sim_ids[keep]

    cubes, noise = [], []
    for z in REDSHIFTS:
        cm = np.memmap(clean_tpl.format(z=z), dtype=np.float64, mode="r")
        cm = cm.reshape(-1, N, N, N)[:NB_SIMU_TOTAL]
        cubes.append(np.asarray(cm[sel_ids], dtype=np.float32))
        nm = np.memmap(noise_tpl.format(z=z), dtype=np.float64,
                       mode="r").reshape(-1, N, N, N)
        m = min(n_noise_keep, nm.shape[0])
        noise.append(np.asarray(nm[:m], dtype=np.float32) * noise_scale)
    return {"params": params, "cubes": cubes, "noise": noise}


# ------------------------------------------------------------------- demo
def demo_params(n, rng):
    """Gridded, censored design mimicking Loreli II (style-preview only)."""
    fX = np.geomspace(0.1, 10.0, 10)
    tau = np.geomspace(738.9, 10504.7, 10)
    th2 = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    mmin = np.linspace(8.0, 9.6, 8)
    fesc = np.array([0.05, 0.2, 0.5])
    rows = []
    while len(rows) < n:
        p = [rng.choice(fX), rng.choice(tau), rng.choice(th2),
             rng.choice(mmin), rng.choice(fesc)]
        if p[1] > 4000.0 and p[3] > 8.9:       # censored corner
            continue
        rows.append(p)
    return np.array(rows)


def demo_summaries(n_sims, n_noise, rng):
    """Synthetic (params, ps, lmom) with fX/Mmin-driven trends + noise scatter."""
    p = demo_params(n_sims, rng)
    ps = np.empty((n_sims, n_noise, PS_NBINS), np.float32)
    lm = np.empty((n_sims, n_noise, LMOM_NCOLS), np.float32)
    for i, (fx, tau, th2, mm, fe) in enumerate(p):
        amp = 30.0 * fx ** 0.8 * (mm / 8.8) ** -2
        shape = amp * (PS_KCENTERS / 0.1) ** (-0.6 + 0.15 * np.log10(fx))
        noise_floor = 8.0 * (PS_KCENTERS / 0.1) ** 1.5
        ps[i] = shape + noise_floor + rng.normal(
            0, 0.1 * (shape + noise_floor), (n_noise, PS_NBINS))
        base = np.array([0.0, 6 + 2 * np.log10(fx), -1.5 * np.log10(fx) - th2,
                         1.2 + 0.3 * np.log10(fx), -0.3 * th2, 0.15])
        base = base * (mm / 8.8) ** -1
        lm[i] = base + rng.normal(0, 0.35, (n_noise, LMOM_NCOLS))
    return p, ps, lm


def demo_cubes(n_sims, rng):
    """Lognormal GRF mock cubes (see previous deliverable) for style checks."""
    p = demo_params(n_sims, rng)
    kx = np.fft.fftfreq(N)[:, None, None]
    ky = np.fft.fftfreq(N)[None, :, None]
    kz = np.fft.fftfreq(N)[None, None, :]
    kk = np.sqrt(kx**2 + ky**2 + kz**2); kk[0, 0, 0] = 1e-6
    cubes, noise = [], []
    for iz in range(3):
        sig = np.empty((n_sims, N, N, N), np.float32)
        pk = kk ** (-2.4 + 0.15 * iz); pk[0, 0, 0] = 0.0
        for i, (fx, tau, th2, mm, fe) in enumerate(p):
            amp = 8.0 * fx ** 0.4 * (1 + 0.12 * iz) * (mm / 8.8) ** -1.5
            a = 0.55 + 0.3 * np.log10(fx) + 0.10 * iz
            g = np.fft.ifftn(np.fft.fftn(rng.standard_normal((N, N, N)))
                             * np.sqrt(pk)).real
            g = (g - g.mean()) / g.std()
            f = np.exp(np.clip(a * g, -20, 20)); f = f / f.mean() - 1.0
            sig[i] = (amp * f - 0.4 * amp * th2 * g).astype(np.float32)
        cubes.append(sig)
        pn = kk ** 0.5; pn[0, 0, 0] = 0.0
        nz = np.empty((60, N, N, N), np.float32)
        for j in range(60):
            g = np.fft.ifftn(np.fft.fftn(rng.standard_normal((N, N, N)))
                             * np.sqrt(pn)).real
            nz[j] = (4.0 * (1 + 0.9 * iz) * (g - g.mean()) / g.std())
        noise.append(nz)
    return {"params": p, "cubes": cubes, "noise": noise}
