#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

# Import all three plotting modules safely!
from _sbc import plot_sbc
from _corner import plot_corner
from _latent import plot_latent

PATTERN = re.compile(r"_sim(\d+)_(?:row|offset)(\d+)")
COMMON_PARAMS = [0, 1, 2, 3]   
LABELS = [r"$\log_{10}(F_x)$", r"$\tau$", r"$r_{H/S}$", r"$\log_{10}(M_{min})$", r"$f_{esc}$"]

# ----------------------------- chain IO -----------------------------
def list_chains(d):
    return sorted(glob.glob(os.path.join(d, "*.dat")))

def chain_sim(fname):
    m = PATTERN.search(os.path.basename(fname))
    return int(m.group(1)) if m else None

def load_chain(fpath, n_params, burnin_frac, thin, dlogp):
    truth_path = fpath[:-4] + "_truth.npy"
    if not os.path.exists(truth_path):
        return None
    truth = np.load(truth_path).astype(np.float64)

    size = os.path.getsize(fpath) // np.dtype(np.float32).itemsize
    if size == 0 or size % n_params != 0:
        return None
    nrows = size // n_params
    chain = np.memmap(fpath, dtype=np.float32, mode="r", shape=(nrows, n_params))
    samples = np.asarray(chain, dtype=np.float64)

    b = int(burnin_frac * len(samples))
    samples = samples[b::thin]
    if samples.shape[0] < 20:
        return None

    if dlogp is not None:
        logp_path = fpath[:-4] + "_logp.npy"
        if os.path.exists(logp_path):
            logp = np.load(logp_path).astype(np.float64)
            if logp.shape[0] == nrows:
                logp = logp[b::thin]
                if np.any(np.isfinite(logp)):
                    keep = logp > (np.max(logp[np.isfinite(logp)]) - dlogp)
                    if keep.sum() >= 20:
                        samples = samples[keep]
    return samples, truth

# ----------------------------- statistics -----------------------------
def iqr_sigma(col):
    a = np.quantile(col, [0.15865, 1.0 - 0.15865])
    return 0.5 * (a[1] - a[0])

def sbc_rank(samples, truth, j, nbins):
    frac = np.mean(samples[:, j] < truth[j])
    return min(int(frac * nbins), nbins - 1)

def eval_arm(arm, burnin_frac, thin, dlogp, nbins, sims_keep):
    n_params = arm["n_params"]
    files = list_chains(arm["dir"])
    rank_hist = np.zeros((len(COMMON_PARAMS), nbins))
    sigmas = [[] for _ in COMMON_PARAMS]
    sqrtdets = []
    scatter = {j: {"truth": [], "mean": [], "std": []} for j in range(len(COMMON_PARAMS))}
    used = 0
    for f in files:
        sim = chain_sim(f)
        if sim is None: continue
        if sims_keep is not None and sim not in sims_keep: continue
        loaded = load_chain(f, n_params, burnin_frac, thin, dlogp)
        if loaded is None: continue
        samples, truth = loaded
        sub = samples[:, COMMON_PARAMS]
        cov = np.cov(sub, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov)
        if sign > 0 and np.isfinite(logdet):
            sqrtdets.append(np.exp(0.5 * logdet))
        for jj, j in enumerate(COMMON_PARAMS):
            sigmas[jj].append(iqr_sigma(samples[:, j]))
            rank_hist[jj, sbc_rank(samples, truth, j, nbins)] += 1.0
            scatter[jj]["truth"].append(truth[j])
            scatter[jj]["mean"].append(np.median(samples[:, j]))
            scatter[jj]["std"].append(iqr_sigma(samples[:, j]))
        used += 1
    if used == 0: return None
    rank_hist *= nbins / used
    calib = np.mean((rank_hist - 1.0) ** 2, axis=1)
    floor = (nbins - 1) / used
    calib_ratio = calib / floor
    tol_1sigma = math.sqrt(2.0 / (nbins - 1))
    calib_nsigma = (calib_ratio - 1.0) / tol_1sigma
    sqrtdets = np.asarray(sqrtdets)
    return {
        "name": arm["name"], "used": used,
        "dir": arm["dir"], "n_params": n_params,
        "sigma": [float(np.median(s)) for s in sigmas],
        "sigma_mean": [float(np.mean(s)) for s in sigmas],
        "gv": float(np.median(sqrtdets)),
        "gv_mean": float(np.mean(sqrtdets)),
        "calib": calib.tolist(),
        "calib_floor": float(floor),
        "calib_ratio": calib_ratio.tolist(),
        "calib_nsigma": calib_nsigma.tolist(),
        "rank_hist": rank_hist.tolist(),
        "scatter": {j: {k: list(map(float, v)) for k, v in scatter[j].items()} for j in scatter},
    }

def common_sims(arms):
    sets = []
    for a in arms:
        s = {chain_sim(f) for f in list_chains(a["dir"])}
        s.discard(None)
        sets.append(s)
    return set.intersection(*sets) if sets else set()

def _write_table(report, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "arm", "n_used", "GV_4x4", "GV_4x4_mean"]
                   + [f"sigma_{LABELS[j]}" for j in COMMON_PARAMS]
                   + [f"calibratio_{LABELS[j]}" for j in COMMON_PARAMS]
                   + [f"calibnsig_{LABELS[j]}" for j in COMMON_PARAMS]
                   + [f"calib_{LABELS[j]}" for j in COMMON_PARAMS])
        for mode, arms in report.items():
            for r in arms:
                w.writerow([mode, r["name"], r["used"],
                            f"{r['gv']:.6e}", f"{r['gv_mean']:.6e}"]
                           + [f"{s:.6e}" for s in r["sigma"]]
                           + [f"{c:.3f}" for c in r["calib_ratio"]]
                           + [f"{c:.2f}" for c in r["calib_nsigma"]]
                           + [f"{c:.4f}" for c in r["calib"]])
    print("\n=== GV (median sqrt|Sigma_4x4|) / sigma (median IQR) / calibration ===")
    for mode, arms in report.items():
        print(f"\n[{mode}]")
        for r in arms:
            print(f"  {r['name']:24s} N={r['used']:4d} GV={r['gv']:.4e} "
                  f"sigma={[round(s, 4) for s in r['sigma']]} "
                  f"calib_ratio={[round(c, 2) for c in r['calib_ratio']]} "
                  f"(1.0=perfect; floor={(r['calib_floor']):.3f})")

def _plot_scatter(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mode, arms in report.items():
        cmap = plt.get_cmap("viridis"); m = max(len(arms) - 1, 1)
        fig, axes = plt.subplots(1, len(COMMON_PARAMS), figsize=(4 * len(COMMON_PARAMS), 4))
        for jj, j in enumerate(COMMON_PARAMS):
            ax = axes[jj]
            lims = []
            for i, r in enumerate(arms):
                s = r["scatter"].get(jj, r["scatter"].get(str(jj)))
                t, md = np.array(s["truth"]), np.array(s["mean"])
                ax.scatter(t, md, s=8, alpha=0.4, color=cmap(0.12 + 0.76 * i / m),
                           label=(r["name"] if jj == 0 else None))
                lims += [t.min(), t.max(), md.min(), md.max()]
            lo, hi = min(lims), max(lims)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(LABELS[j]); ax.set_xlabel("truth"); ax.set_ylabel("posterior median")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"[{mode}]"); fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", mode)
        fig.savefig(os.path.join(out, f"scatter_overlay_{safe}.png"), dpi=150)
        plt.close(fig)

def _write_seed_avg(report, path):
    rows = []
    for mode, arms in report.items():
        groups = {}
        for r in arms:
            #base = re.sub(r"\s*s\d+$", "", r["name"]).strip()
            base = re.sub(r"\s*(?:s|seed\s*)\d+$", "", r["name"]).strip()
            groups.setdefault(base, []).append(r)
            
        for base, g in groups.items():
            n = len(g)
            
            # Geometric mean for scale quantities (GV, sigma)
            def geo_ms(vals):
                v = np.asarray(vals, float)
                lg = np.log10(np.clip(v, 1e-300, None))
                return float(10.0 ** lg.mean()), float(lg.std(ddof=1) if n > 1 else 0.0)
                
            # Arithmetic mean for calibration ratio
            def arith_ms(vals):
                v = np.asarray(vals, float)
                return float(v.mean()), float(v.std(ddof=1) if n > 1 else 0.0)

            # Apply correct averaging
            gv_m, gv_s = geo_ms([r["gv"] for r in g])
            row = {"mode": mode, "arm": base, "n_seeds": n, "GV_mean": gv_m, "GV_sem": gv_s}
            
            for jj, j in enumerate(COMMON_PARAMS):
                sm, ss = geo_ms([r["sigma"][jj] for r in g])
                cm, cs = arith_ms([r["calib_ratio"][jj] for r in g])
                row[f"sigma_{LABELS[j]}_mean"] = sm; row[f"sigma_{LABELS[j]}_sem"] = ss
                row[f"calibratio_{LABELS[j]}_mean"] = cm; row[f"calibratio_{LABELS[j]}_sem"] = cs
            rows.append(row)
            
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[seed-avg] wrote {path} ({len(rows)} groups)")
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of arms: name, dir, n_params")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--burnin-frac", type=float, default=0.0)
    ap.add_argument("--thin", type=int, default=11)
    ap.add_argument("--dlogp", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--no-intersect", action="store_true")
    ap.add_argument("--filtered-only", action="store_true", help="skip unfiltered")
    ap.add_argument("--full-table", action="store_true", help="keep GV_mean column")
    ap.add_argument("--n-corner", type=int, default=1)
    args = ap.parse_args()

    arms = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)

    sims_keep = None
    if not args.no_intersect:
        sims_keep = common_sims(arms)
        print(f"Shared sims across all arms: {len(sims_keep)}")
        if len(sims_keep) == 0:
            print("WARNING: no shared sims; falling back to per-arm sims.")
            sims_keep = None

    if args.filtered_only:
        modes = {f"filtered_dlogp{args.dlogp:g}": args.dlogp}
    else:
        modes = {"unfiltered": None, f"filtered_dlogp{args.dlogp:g}": args.dlogp}

    report = {}
    for mode, dlogp in modes.items():
        report[mode] = []
        for arm in arms:
            r = eval_arm(arm, args.burnin_frac, args.thin, dlogp, args.nbins, sims_keep)
            if r is not None: report[mode].append(r)

    json.dump(report, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    _write_table(report, os.path.join(args.out, "metrics.csv"))
    
    try:
        from latex_table import write_latex
        write_latex(report, os.path.join(args.out, "metrics.tex"), compact=not args.full_table)
    except Exception as e:
        pass

    _write_seed_avg(report, os.path.join(args.out, "metrics_seed_avg.csv"))
    _plot_scatter(report, args.out)
    
    # --- EXTERNAL PLOTS WITH EXPLICIT CONSTANTS PASSED IN ---
    plot_sbc(report, args.out, args.nbins, COMMON_PARAMS, LABELS)
    
    _corner_dlogp = args.dlogp if args.filtered_only else None
    plot_corner(report, args.out, args.burnin_frac, args.thin, _corner_dlogp, args.n_corner, COMMON_PARAMS, LABELS)
    
    #for arm in arms:
    #    plot_latent(arm, args.out, LABELS)

    print(f"\nWrote {args.out}/: metrics.csv, metrics_seed_avg.csv, metrics.tex, metrics.json, "
          f"scatter_overlay_*.png, sbc_overlay_*.png, corner_sim*_*.png")

if __name__ == "__main__":
    main()