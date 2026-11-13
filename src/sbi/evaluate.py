"""Stage-4 metrics on SBC chains.

Per chain (in this order):
  1. stuck-walker cut: drop walkers whose median log-posterior is more than `walker_cut`
     below the 90th percentile of the walker medians
  2. burn-in fraction, thinning
  3. dlogp cut: keep samples with log p > max(log p) - dlogp (if >= 20 remain)
Per run, over the simulations shared by every run being compared:
  SBC rank histogram (nbins) of each parameter, calibration ratio
  mean((h - 1)^2) / ((nbins - 1) / N)  (1 = consistent with uniform ranks),
  GV = median over sims of sqrt(det Cov) of the posterior samples,
  sigma = median over sims of the 16-84% half-width.
"""
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np

SIM_RE = re.compile(r"_sim(\d+)_row(\d+)")


def list_chains(d):
    return sorted(glob(os.path.join(d, "*.dat")))


def chain_sim(f):
    m = SIM_RE.search(os.path.basename(f))
    return int(m.group(1)) if m else None


def walker_mask(base, nrows, walker_cut):
    if walker_cut is None or walker_cut <= 0:
        return None
    if not (os.path.exists(base + "_meta.npy") and os.path.exists(base + "_logp.npy")):
        return None
    meta = np.load(base + "_meta.npy")
    W, S, B = int(meta[0]), int(meta[1]), int(meta[2])
    if W <= 0 or (S - B) * W != nrows:
        return None
    L = np.load(base + "_logp.npy")
    if L.shape[0] != nrows:
        return None
    L = L.reshape(S - B, W).astype(np.float64)
    L[~np.isfinite(L)] = -np.inf
    wm = np.median(L, axis=0)
    fin = wm[np.isfinite(wm)]
    if fin.size == 0:
        return None
    return np.tile(wm >= np.percentile(fin, 90.0) - walker_cut, S - B)


def read_chain(f, n_params, ev):
    """-> (samples f64, truth f64) after the cuts, or None."""
    base = f[:-4]
    if not os.path.exists(base + "_truth.npy"):
        return None
    truth = np.load(base + "_truth.npy").astype(np.float64)
    size = os.path.getsize(f) // 4
    if size == 0 or size % n_params:
        return None
    nrows = size // n_params
    chain = np.fromfile(f, dtype=np.float32, count=size).reshape(nrows, n_params)
    logp = np.load(base + "_logp.npy") if os.path.exists(base + "_logp.npy") else None
    if logp is not None and logp.shape[0] != nrows:
        logp = None
    wm = walker_mask(base, nrows, ev["walker_cut"])
    if wm is not None:
        chain, logp = chain[wm], (logp[wm] if logp is not None else None)
    b = int(ev["burnin_frac"] * chain.shape[0])
    samples = chain[b::ev["thin"]]
    if samples.shape[0] < 20:
        return None
    samples = samples.astype(np.float64)
    if logp is not None:
        lp = logp[b::ev["thin"]].astype(np.float64)
        if np.any(np.isfinite(lp)):
            keep = lp > np.max(lp[np.isfinite(lp)]) - ev["dlogp"]
            if keep.sum() >= 20:
                samples = samples[keep]
    return samples, truth


def half_width(col):
    a = np.quantile(col, [0.15865, 1.0 - 0.15865])
    return 0.5 * (a[1] - a[0])


def eval_run(d, n_params, params, ev, sims):
    files = [f for f in list_chains(d) if chain_sim(f) in sims]
    with ThreadPoolExecutor(int(os.environ.get("SBI_EVAL_WORKERS", "16"))) as ex:
        loaded = list(ex.map(lambda f: read_chain(f, n_params, ev), files))
    nb, P = ev["nbins"], len(params)
    hist = np.zeros((P, nb))
    sig = [[] for _ in params]
    gv = []
    used = 0
    for r in loaded:
        if r is None:
            continue
        s, truth = r
        sign, logdet = np.linalg.slogdet(np.cov(s[:, params], rowvar=False))
        if sign > 0 and np.isfinite(logdet):
            gv.append(np.exp(0.5 * logdet))
        for jj, j in enumerate(params):
            sig[jj].append(half_width(s[:, j]))
            hist[jj, min(int(np.mean(s[:, j] < truth[j]) * nb), nb - 1)] += 1.0
        used += 1
    if used == 0:
        return None
    hist *= nb / used
    calib = np.mean((hist - 1.0) ** 2, axis=1)
    floor = (nb - 1) / used
    return {"used": used, "gv": float(np.median(gv)), "sigma": [float(np.median(x)) for x in sig],
            "calib_ratio": (calib / floor).tolist(),
            "calib_nsigma": ((calib / floor - 1.0) / math.sqrt(2.0 / (nb - 1))).tolist(),
            "rank_hist": hist.tolist()}
