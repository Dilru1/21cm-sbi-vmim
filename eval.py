#!/usr/bin/env python3
r"""Single evaluation tool for the compressor-comparison campaign.

Consolidates eval.py + eval_corner_global.py + stage4_eval.py into one file.
Produces, for a set of arms:
  * SBC rank histograms   -> sbc_<mode>.pdf       (one page, param rows, arms overlaid)
  * metrics table         -> metrics.csv
  * LaTeX table           -> metrics.tex          (\input-ready, booktabs)
  * overlay corner        -> corner_sim<ID>.pdf    (one shared simulation, arms overlaid)

Deliberately DROPPED from the old harness: scatter plots, PNG output, and the
two-pass filtered/unfiltered report.

FILTERING: by default EVERYTHING -- SBC ranks, GV, sigma, and the corner -- is
computed on chains cut at logp > max-dlogp (--dlogp, default 10), matching the
[filtered_dlogp10] tables and figures already in the report. --dlogp -1 disables
the cut entirely; --sbc-unfiltered keeps widths filtered but ranks on the full
chain. The mode is printed, recorded in metrics.csv, and put in the SBC filename
and panel titles, so a figure can never be mistaken for the other convention.

ARMS are specified as repeatable PACKED ITEMS, one per yaml:

    --item "CONFIG.yaml|FAMILY|SCOPE[|LABEL]"

  FAMILY  gmm | maf | nsf      (stage-2 density family)
  SCOPE   std | raw            (stage-2 t-coordinate)
  LABEL   optional legend text

Each item resolves to exactly ONE chains dir, built (not globbed) as
    {scratch_root}/{arm_name}/chains/{SCOPE}_{FAMILY}/
where arm_name already carries the _s<seed> suffix when the yaml sets
tag_arm_with_seed -- so seeded runs need no special handling, and every stage
(1/2/3/4) agrees on the path. A missing dir is an explicit warning naming it.

Different items may use different families/scopes, so one report can mix them.

  # noise ladder, same family/scope, one arm per yaml
  python eval_all.py --out eval_reports/noise \
      --item "configs/noise/n1.yaml|nsf|std|100 h (ns=1)" \
      --item "configs/noise/n2.yaml|nsf|std|25 h (ns=2)"  \
      --item "configs/noise/n3.yaml|nsf|std|11.1 h (ns=3)" \
      --corner-sim median

  # mixed families on one arm
  python eval_all.py --out eval_reports/family \
      --item "configs/arm_cnn.yaml|gmm|std" \
      --item "configs/arm_cnn.yaml|maf|std" \
      --item "configs/arm_cnn.yaml|nsf|std"

  # empty fields -> take nle.model / nle.standardize from the yaml
  python eval_all.py --item "configs/arm_cnn.yaml||" --out eval_report
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

# ---- conventions (formerly imported from eval.py; inlined so this file stands alone)
PATTERN = re.compile(r"_sim(\d+)_(?:row|offset)(\d+)")
COMMON_PARAMS = [0, 1, 2, 3]                    # Fx, tau, rH/S, Mmin (fesc dropped)
LABELS = [r"$\log_{10}(F_x)$", r"$\tau$", r"$r_{H/S}$",
          r"$\log_{10}(M_{\min})$", r"$f_{esc}$"]
PLABELS = [LABELS[j] for j in COMMON_PARAMS]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8B4513"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType: arXiv-safe
    "savefig.bbox": "tight",
})


# ============================================================ chain IO
def list_chains(d):
    return sorted(glob.glob(os.path.join(d, "*.dat")))


def chain_sim(fname):
    m = PATTERN.search(os.path.basename(fname))
    return int(m.group(1)) if m else None


def load_chain(fpath, n_params, burnin_frac, thin, dlogp):
    """(samples[N,n_params], truth[n_params]) after burn-in/thin/optional filter, or None.

    Stage-3 already discards burn-in and flattens walkers before writing, so the
    default burnin_frac=0 is correct; the option remains for foreign chains.
    dlogp=None -> UNFILTERED (what SBC must use).
    """
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
        lp = fpath[:-4] + "_logp.npy"
        if os.path.exists(lp):
            logp = np.load(lp).astype(np.float64)
            if logp.shape[0] == nrows:
                logp = logp[b::thin]
                if np.any(np.isfinite(logp)):
                    keep = logp > (np.max(logp[np.isfinite(logp)]) - dlogp)
                    if keep.sum() >= 20:
                        samples = samples[keep]
    return samples, truth


# ============================================================ statistics
def iqr_sigma(col):
    a = np.quantile(col, [0.15865, 0.84135])
    return 0.5 * (a[1] - a[0])


def eval_arm(arm, burnin_frac, thin, dlogp, nbins, sims_keep, sbc_unfiltered=False):
    """SBC + GV/sigma, both on the SAME chains.

    dlogp=None            -> everything on the full chain
    dlogp=X               -> everything on logp > max-X   (the default)
    sbc_unfiltered=True   -> widths filtered, SBC on the full chain

    Default is FILTERED THROUGHOUT, matching the [filtered_dlogp10] figures and
    tables already in the report. Note the tradeoff: the dlogp cut removes the
    low-posterior tail, so a truth sitting in that tail can no longer land in the
    extreme rank bins. This narrows the rank distribution slightly and biases
    chi^2 toward the under-confident (n-shaped) side. It is applied identically
    to every arm, so ARM-TO-ARM comparisons remain fair; only the absolute
    distance from the 0.033 sampling floor is affected.
    """
    n_params = arm["n_params"]
    rank_hist = np.zeros((len(COMMON_PARAMS), nbins))
    sigmas = [[] for _ in COMMON_PARAMS]
    sqrtdets = []
    used = 0
    for f in list_chains(arm["dir"]):
        sim = chain_sim(f)
        if sim is None or (sims_keep is not None and sim not in sims_keep):
            continue
        loaded = load_chain(f, n_params, burnin_frac, thin, dlogp)
        if loaded is None:
            continue
        samples, truth = loaded

        # ranks: on the filtered chain unless explicitly opted out
        rsrc = samples
        if sbc_unfiltered and dlogp is not None:
            raw = load_chain(f, n_params, burnin_frac, thin, None)
            if raw is not None:
                rsrc = raw[0]
        for jj, j in enumerate(COMMON_PARAMS):
            frac = np.mean(rsrc[:, j] < truth[j])
            rank_hist[jj, min(int(frac * nbins), nbins - 1)] += 1.0

        # widths: always on the filtered chain
        sub = samples[:, COMMON_PARAMS]
        cov = np.cov(sub, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov)
        if sign > 0 and np.isfinite(logdet):
            sqrtdets.append(np.exp(0.5 * logdet))
        for jj, j in enumerate(COMMON_PARAMS):
            sigmas[jj].append(iqr_sigma(samples[:, j]))
        used += 1

    if used == 0:
        return None
    rank_hist *= nbins / used
    calib = np.mean((rank_hist - 1.0) ** 2, axis=1)
    sqrtdets = np.asarray(sqrtdets)
    return {
        "name": arm["name"], "used": used,
        "sigma": [float(np.median(s)) for s in sigmas],
        "gv": float(np.median(sqrtdets)) if len(sqrtdets) else float("nan"),
        "calib": calib.tolist(),
        "rank_hist": rank_hist.tolist(),
    }


# ============================================================ arm discovery
SCOPE_ALIASES = {"std": "standard_t", "standard": "standard_t", "standard_t": "standard_t",
                 "raw": "raw_t", "raw_t": "raw_t"}


def parse_item(spec):
    """'cfg.yaml|family|scope[|label]' -> dict. Only cfg.yaml is required.

    family : gmm | maf | nsf   (stage-2 density family)
    scope  : std | raw         (stage-2 t-coordinate; std -> standard_t)
    label  : optional legend text; defaults to '<arm> [<scope>_<family>]'

    Empty fields are allowed: 'cfg.yaml||' keeps the yaml's own nle.model and
    nle.standardize. This is the SAME packing stage4_eval.py used, so existing
    command lines port over unchanged.
    """
    parts = [p.strip() for p in spec.split("|")]
    cf = parts[0]
    fam = parts[1].lower() if len(parts) > 1 and parts[1] else None
    sc = parts[2].lower() if len(parts) > 2 and parts[2] else None
    lab = parts[3] if len(parts) > 3 and parts[3] else None
    if sc is not None:
        if sc not in SCOPE_ALIASES:
            sys.exit(f"bad scope '{sc}' in item '{spec}' -- use std|raw")
        sc = SCOPE_ALIASES[sc]
    if fam is not None and fam not in ("gmm", "maf", "nsf"):
        sys.exit(f"bad family '{fam}' in item '{spec}' -- use gmm|maf|nsf")
    return {"config": cf, "family": fam, "scope": sc, "label": lab}


def arms_from_items(items, all_seeds=False, seeds_filter=None):
    """Resolve each packed item to one or more chains dirs.

    Default (all_seeds=False): each item resolves to exactly ONE chains dir,
        {scratch}/{arm_name}/chains/{scope}_{family}/
    with arm_name already carrying the _s<seed> suffix when tag_arm_with_seed is
    on (i.e. the single seed baked into the yaml). Path is CONSTRUCTED, not
    searched, so a typo/missing run is an explicit named error.

    all_seeds=True: each item is EXPANDED to every {arm_name}_s<seed> run found
    on disk for the requested {scope}_{family}, one arm per seed. This is how you
    evaluate a whole seed sweep from a single yaml. seeds_filter (a set of ints)
    optionally restricts which seeds are included. If no _s<seed> dirs are found
    (e.g. tagging was off), it falls back to the single-arm construction.
    """
    from sbi import load_config, arm_dirs
    try:
        from sbi.seeding import apply_arm_name
    except Exception:
        apply_arm_name = None

    arms = []
    for it in items:
        cfg = load_config(it["config"])
        raw_arm = cfg["arm_name"]                 # RAW name, before _s<seed> tagging
        nc = cfg.get("nle", {})
        fam = it["family"] or str(nc.get("model", "gmm")).lower()
        if it["scope"] is not None:
            scope = it["scope"]
        else:
            scope = "standard_t" if bool(nc.get("standardize", True)) else "raw_t"
        leaf = f"{scope}_{fam}"

        # ---- all-seeds: discover {scratch}/{raw_arm}_s<seed>/chains/{leaf} ----
        expanded = []
        if all_seeds:
            pat = os.path.join(cfg["scratch_root"], raw_arm + "_s*", "chains", leaf)
            for cdir in glob.glob(pat):
                armdirname = os.path.basename(os.path.dirname(os.path.dirname(cdir)))
                m = re.search(r"_s(\d+)$", armdirname)
                if not m:
                    continue
                seed = int(m.group(1))
                if seeds_filter and seed not in seeds_filter:
                    continue
                if next(iter(glob.glob(os.path.join(cdir, "*.dat"))), None) is None:
                    print(f"[warn] no .dat in {cdir} -- seed {seed} skipped", file=sys.stderr)
                    continue
                expanded.append((seed, cdir, armdirname))

        if expanded:
            for seed, cdir, armdirname in sorted(expanded):
                base_lab = it["label"] or f"{armdirname} [{leaf}]"
                arms.append({
                    "name": f"{armdirname}[{leaf}]",
                    "dir": cdir,
                    "n_params": cfg["n_params"],
                    "label": f"{base_lab} s{seed}",
                    "family": fam, "scope": scope, "config": it["config"],
                })
            print(f"[all-seeds] {raw_arm} [{leaf}] -> seeds "
                  f"{sorted(s for s, _, _ in expanded)}")
            continue
        elif all_seeds:
            print(f"[warn] all-seeds: no {raw_arm}_s*/chains/{leaf} found; "
                  f"falling back to single arm", file=sys.stderr)

        # ---- single-arm construction (default, or all-seeds fallback) ----
        if apply_arm_name is not None:
            apply_arm_name(cfg)          # adds _s<init_seed> exactly as stages 1-3 do
        chains_root = arm_dirs(cfg)["chains"]
        d = chains_root / leaf
        if not d.is_dir():                       # legacy flat layout fallback
            if chains_root.is_dir() and next(iter(glob.glob(str(chains_root / "*.dat"))), None):
                print(f"[warn] {leaf} not found; using flat {chains_root}", file=sys.stderr)
                d = chains_root
            else:
                print(f"[warn] MISSING {d} -- arm skipped", file=sys.stderr)
                continue
        if next(iter(glob.glob(str(d / "*.dat"))), None) is None:
            print(f"[warn] no .dat in {d} -- arm skipped", file=sys.stderr)
            continue

        arms.append({
            "name": f"{cfg['arm_name']}[{leaf}]",
            "dir": str(d),
            "n_params": cfg["n_params"],
            "label": it["label"] or f"{cfg['arm_name']} [{leaf}]",
            "family": fam, "scope": scope, "config": it["config"],
        })
    return arms


def load_arms(args):
    if args.item:
        arms = arms_from_items([parse_item(x) for x in args.item],
                               all_seeds=args.all_seeds,
                               seeds_filter=set(args.seeds) if args.seeds else None)
    else:
        pool = []
        for jp in args.arms_json:
            if os.path.exists(jp):
                pool.extend(json.load(open(jp)))
            else:
                print(f"[warn] {jp} not found", file=sys.stderr)
        arms = pool
        if args.select:
            arms = [a for a in arms if any(re.search(s, a["name"]) for s in args.select)]
    # de-dup on dir, preserve order
    seen, out = set(), []
    for a in arms:
        if a["dir"] in seen:
            continue
        seen.add(a["dir"]); out.append(a)
    # explicit --labels wins over any per-item label
    for k, a in enumerate(out):
        if args.labels and k < len(args.labels):
            a["label"] = args.labels[k]
        a.setdefault("label", a["name"])
    return out


def common_sims(arms):
    sets = []
    for a in arms:
        s = {chain_sim(f) for f in list_chains(a["dir"])}
        s.discard(None)
        sets.append(s)
    return set.intersection(*sets) if sets else set()


# ============================================================ SBC figure (PDF)
def sbc_reference(nbins, n_used):
    denom = float(n_used) / float(nbins)
    s = 1.0 / math.sqrt(denom) if denom > 0 else 0.0
    bands = dict(low1=1 - s, high1=1 + s, low2=1 - 2 * s, high2=1 + 2 * s)

    def cdf_hist(loc, scale, n=400000):
        d = np.random.default_rng(0).normal(loc, scale, n)
        cdf = 0.5 * (1 + np.vectorize(math.erf)(d / np.sqrt(2)))
        idx = np.clip((cdf * nbins).astype(int), 0, nbins - 1)
        return np.bincount(idx, minlength=nbins).astype(float) / (n / nbins)

    return bands, cdf_hist(0.2, 1.0), cdf_hist(0.0, 0.85)

# ============================================================ SBC figure (PDF)
def plot_sbc(report, out_pdf, nbins, mode_txt):
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    # divergent astro palette from cmastro for the arm curves; safe fallback
    try:
        from cmastro import cmaps
        _cmap = cmaps["cma:emph"]
    except Exception:
        _cmap = plt.get_cmap("Spectral")
    n_arms = max(len(report), 1)
    arm_colors = [_cmap(v) for v in np.linspace(0.05, 0.95, n_arms)]

    xval = np.arange(0.5 / nbins, 1.0, 1.0 / nbins)
    n_used = max((r["used"] for r in report), default=0)
    bands, bias_h, under_h = sbc_reference(nbins, n_used)
    P = len(COMMON_PARAMS)

    # --- Poisson/multinomial calibration floor -------------------------------
    # For calib = mean_bins (h-1)^2 with h normalized so a flat histogram = 1:
    #   E[calib | perfect posterior] = (nbins-1)/N        (per-arm N)
    #   percentage = 100 * calib / floor  == reduced chi^2 * 100 (100% = at floor)
    #   Pearson X^2 = N*calib ~ chi^2_(nbins-1);  1-sigma tol on ratio = sqrt(2/(nbins-1))
    tol_pct   = 100.0 * math.sqrt(2.0 / max(nbins - 1, 1))
    floor_ref = (nbins - 1) / max(n_used, 1)

    fig, axes = plt.subplots(P, 1, figsize=(7.0, 2.35 * P),
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]

    for jj in range(P):
        ax = axes[jj]
        # visible confidence shading (== per-bin Poisson 2sigma/1sigma envelope)
        ax.fill_between([0, 1], bands["low2"], bands["high2"], color="#a6bddb", alpha=0.45, lw=0, zorder=0)
        ax.fill_between([0, 1], bands["low1"], bands["high1"], color="#3690c0", alpha=0.35, lw=0, zorder=0)
        ax.axhline(1.0, color="0.35", ls="-", lw=0.8, zorder=1)
        # reference deviation curves
        ax.plot(xval, bias_h,  ls="--", lw=1.2, color="k", zorder=2, label=r"0.2$\sigma$ bias")
        ax.plot(xval, under_h, ls=":",  lw=1.5, color="k", zorder=2, label=r"0.15$\sigma$ under-conf.")
        # arm rank histograms (labels carry no chi^2 -> percentages go in-panel below)
        for k, r in enumerate(report):
            ax.plot(xval, np.asarray(r["rank_hist"])[jj], drawstyle="steps-mid",
                    lw=1.7, color=arm_colors[k], zorder=3, label=r["label"])

        ax.set_ylim(0, 2.0); ax.set_xlim(0, 1)
        ax.set_ylabel("rank freq", fontsize=8.5); ax.tick_params(labelsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.text(0.012, 0.94, PLABELS[jj], transform=ax.transAxes, fontsize=10,
                fontweight="semibold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9))

        # chi^2 AS % OF POISSON FLOOR, one color-matched row, bottom-left (per-arm N)
        frags = [TextArea(r"$\chi^2/\mathrm{floor}$:",
                          textprops=dict(color="0.15", fontsize=7.5, fontweight="bold"))]
        for k, r in enumerate(report):
            floor = (nbins - 1) / max(r["used"], 1)
            pct = 100.0 * r["calib"][jj] / floor if floor > 0 else float("nan")
            frags.append(TextArea(f"{pct:.0f}%",
                         textprops=dict(color=arm_colors[k], fontsize=7.5, fontweight="bold")))
        row = HPacker(children=frags, align="baseline", pad=0, sep=7)
        box = AnchoredOffsetbox(loc="lower left", child=row, pad=0.3, borderpad=0.5,
                                frameon=True, bbox_to_anchor=(0.0, 0.0), bbox_transform=ax.transAxes)
        box.patch.set(alpha=0.85, edgecolor="0.8", facecolor="white")
        ax.add_artist(box)

    axes[-1].set_xlabel("fractional rank (CDF of truth)", fontsize=9.5)

    # one common legend for the whole figure (top of panel 0): 2 cols -> 3 rows
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                   fontsize=7.5, columnspacing=1.3, handlelength=1.9, handletextpad=0.5,
                   borderaxespad=0.15, frameon=True, framealpha=0.92, edgecolor="0.8")

    note = (r"Poisson floor $\chi^2=(n_\mathrm{bins}-1)/N$ = " + f"{floor_ref:.3f}\n"
            f"100% = perfect posterior  (" + r"$\pm$" + f"{tol_pct:.0f}% at 1" + r"$\sigma$)")
    fig.text(0.008, 0.988, note, ha="left", va="top", fontsize=6.2, color="0.4", linespacing=1.35)
    axes[0].set_title(f"{mode_txt},  N={n_used}", fontsize=8, color="0.35", loc="right", pad=2)

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[sbc] wrote {out_pdf}")
    

# ============================================================ corner figure (PDF)
def credible_levels(H, fracs=(0.68, 0.95)):
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat); csum /= csum[-1]
    out = []
    for f in fracs:
        i = np.searchsorted(csum, f)
        out.append(flat[i] if i < len(flat) else flat[-1])
    return out[::-1]


def pick_corner_sim(arms, mode, seed, thin, burnin, explicit, dlogp):
    maps = [{chain_sim(f): f for f in list_chains(a["dir"]) if chain_sim(f) is not None}
            for a in arms]
    shared = set(maps[0])
    for m in maps[1:]:
        shared &= set(m)
    shared = np.array(sorted(shared))
    if len(shared) == 0:
        sys.exit("[corner] selected arms share no simulations")
    print(f"[corner] shared sims: {len(shared)}")
    if explicit is not None:
        if explicit not in set(shared.tolist()):
            sys.exit(f"[corner] sim {explicit} not shared by all arms")
        return explicit, maps
    if mode == "random":
        return int(np.random.default_rng(seed).choice(shared)), maps
    # median: sim closest to the reference arm's median GV, computed on the SAME
    # chains the table uses so the "representative" sim really is representative
    ref, rmap = arms[0], maps[0]
    gvs = {}
    for s in shared:
        ld = load_chain(rmap[s], ref["n_params"], burnin, thin, dlogp)
        if ld is None:
            continue
        cov = np.cov(ld[0][:, COMMON_PARAMS], rowvar=False)
        sign, ldet = np.linalg.slogdet(cov)
        if sign > 0 and np.isfinite(ldet):
            gvs[int(s)] = np.exp(0.5 * ldet)
    med = np.median(list(gvs.values()))
    best = min(gvs, key=lambda s: abs(gvs[s] - med))
    print(f"[corner] median-GV sim (ref '{ref['label']}'): {best}")
    return best, maps


def plot_corner(arms, maps, sim, out_pdf, thin, burnin, bins, smooth, width, dlogp):
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    arm_samples, truth = [], None
    for a, m in zip(arms, maps):
        if sim not in m:
            continue
        ld = load_chain(m[sim], a["n_params"], burnin, thin, dlogp)
        if ld is None:
            continue
        s, tr = ld
        if truth is None:
            truth = tr
        arm_samples.append((a["label"], s))
    if not arm_samples:
        print(f"[corner] no usable chains for sim {sim}", file=sys.stderr)
        return

    d = len(COMMON_PARAMS)
    fig, axes = plt.subplots(d, d, figsize=(width, width), squeeze=False)
    lims = []
    for k in range(d):
        lo = min(np.percentile(s[:, COMMON_PARAMS[k]], 0.3) for _, s in arm_samples)
        hi = max(np.percentile(s[:, COMMON_PARAMS[k]], 99.7) for _, s in arm_samples)
        pad = 0.06 * (hi - lo)
        lims.append((lo - pad, hi + pad))

    for r in range(d):
        for c in range(d):
            ax = axes[r][c]
            if c > r:
                ax.axis("off"); continue
            jr, jc = COMMON_PARAMS[r], COMMON_PARAMS[c]
            if r == c:
                for k, (lab, s) in enumerate(arm_samples):
                    ax.hist(s[:, jr], bins=bins, range=lims[r], density=True,
                            histtype="step", lw=1.4, color=PALETTE[k % len(PALETTE)])
                ax.axvline(truth[jr], color="k", lw=1.0, ls="--")
                ax.set_xlim(*lims[r]); ax.set_yticks([])
            else:
                for k, (lab, s) in enumerate(arm_samples):
                    H, xe, ye = np.histogram2d(s[:, jc], s[:, jr], bins=bins,
                                               range=[lims[c], lims[r]])
                    if gaussian_filter is not None and smooth > 0:
                        H = gaussian_filter(H, smooth)
                    if H.sum() <= 0:
                        continue
                    lv = credible_levels(H)
                    xc = 0.5 * (xe[1:] + xe[:-1]); yc = 0.5 * (ye[1:] + ye[:-1])
                    col = PALETTE[k % len(PALETTE)]
                    ax.contourf(xc, yc, H.T, levels=lv + [H.max()],
                                colors=[col, col], alpha=0.18)
                    ax.contour(xc, yc, H.T, levels=lv, colors=col, linewidths=1.1)
                ax.axvline(truth[jc], color="k", lw=0.8, ls="--", alpha=0.7)
                ax.axhline(truth[jr], color="k", lw=0.8, ls="--", alpha=0.7)
                ax.set_xlim(*lims[c]); ax.set_ylim(*lims[r])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if r == d - 1:
                ax.set_xlabel(PLABELS[c])
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(PLABELS[r])
            elif c != 0:
                ax.set_yticklabels([])

    el = [Line2D([0], [0], color=PALETTE[k % len(PALETTE)], lw=2, label=lab)
          for k, (lab, _) in enumerate(arm_samples)]
    el.append(Line2D([0], [0], color="k", lw=1.0, ls="--", label="truth"))
    fig.legend(handles=el, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9)
    fig.suptitle(f"sim {sim}", y=0.995, fontsize=10)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"[corner] wrote {out_pdf}")


# ============================================================ tables
def write_csv(report, path, dlogp, mode_txt):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# mode: {mode_txt}"])      # provenance: which chain cut produced these
        w.writerow(["arm", "n_used", "GV_4x4"]
                   + [f"sigma_{j}" for j in COMMON_PARAMS]
                   + [f"calib_{j}" for j in COMMON_PARAMS])
        for r in report:
            w.writerow([r["name"], r["used"], f"{r['gv']:.6e}"]
                       + [f"{s:.6e}" for s in r["sigma"]]
                       + [f"{c:.4f}" for c in r["calib"]])
    print(f"[table] wrote {path}  ({mode_txt})")


def write_tex(report, path):
    with open(path, "w") as fh:
        fh.write("% auto-generated by eval_all.py -- do not edit\n")
        fh.write("\\begin{tabular}{lr" + "r" * len(COMMON_PARAMS)
                 + "r" * len(COMMON_PARAMS) + "}\n\\toprule\n")
        fh.write("Arm & GV & "
                 + " & ".join(f"$\\sigma_{{{n}}}$" for n in ["Fx", "\\tau", "rHS", "Mmin"])
                 + " & "
                 + " & ".join(f"$\\chi^2_{{{n}}}$" for n in ["Fx", "\\tau", "rHS", "Mmin"])
                 + r" \\" + "\n\\midrule\n")
        for r in report:
            fh.write(f"{r['label']} & {r['gv']:.2e} & "
                     + " & ".join(f"{s:.2e}" for s in r["sigma"]) + " & "
                     + " & ".join(f"{c:.3f}" for c in r["calib"])
                     + r" \\" + "\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[table] wrote {path}")


def print_summary(report):
    print("\n=== GV (median sqrt|Sigma_4x4|) / sigma (median IQR) / chi^2-to-flat ===")
    for r in report:
        print(f"  {r['label']:26s} N={r['used']:4d} GV={r['gv']:.3e} "
              f"sigma={[round(s,4) for s in r['sigma']]} "
              f"chi2={[round(c,3) for c in r['calib']]}")


# ============================================================ latent diagnostics
# (absorbed from eval_advanced.py; publication-quality PDF output, filtered SBC
#  set. These are the plots that catch compressor collapse: a near-zero-variance
#  t dim, or t uncorrelated with every theta, shows up immediately here.)
def _load_summaries(arm):
    sdir = os.path.join(os.path.dirname(arm["dir"].rstrip("/")), "..", "summaries")
    sdir = os.path.normpath(sdir)
    tp, thp = os.path.join(sdir, "t.npy"), os.path.join(sdir, "theta.npy")
    if not (os.path.exists(tp) and os.path.exists(thp)):
        # try one level up (layout: {arm}/chains/{leaf}/  vs  {arm}/summaries/)
        sdir2 = os.path.normpath(os.path.join(os.path.dirname(arm["dir"].rstrip("/")), "summaries"))
        tp2, thp2 = os.path.join(sdir2, "t.npy"), os.path.join(sdir2, "theta.npy")
        if os.path.exists(tp2) and os.path.exists(thp2):
            sdir, tp, thp = sdir2, tp2, thp2
        else:
            return None
    t = np.load(tp).astype(np.float64)
    theta = np.load(thp).astype(np.float64)
    sim_p = os.path.join(sdir, "original_sim_ids.npy")
    sims = np.load(sim_p) if os.path.exists(sim_p) else None
    return {"t": t, "theta": theta, "sims": sims, "dir": sdir}


def latent_tt_colored(t, theta, out_path, name, pnames):
    n_t, n_p = t.shape[1], theta.shape[1]
    pairs = [(i, j) for i in range(n_t) for j in range(i + 1, n_t)]
    if not pairs:
        return
    m = min(len(t), 20000)
    idx = np.random.default_rng(0).choice(len(t), m, replace=False)
    ts, ths = t[idx], theta[idx]
    fig, axes = plt.subplots(len(pairs), n_p, figsize=(3.1 * n_p, 3.0 * len(pairs)), squeeze=False)
    for ri, (i, j) in enumerate(pairs):
        for ci in range(n_p):
            ax = axes[ri][ci]
            ax.scatter(ts[:, i], ts[:, j], c=ths[:, ci], cmap="Spectral",
                       s=3, alpha=0.5, rasterized=True)
            ax.set_xlabel(rf"$t_{{{i+1}}}$", fontsize=8); ax.set_ylabel(rf"$t_{{{j+1}}}$", fontsize=8)
            if ri == 0:
                ax.set_title(f"colored by {pnames[ci]}", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{name}: latent t-t colored by parameter (round=MSE-like, banana=VMIM-like)",
                 y=1.002, fontsize=11)
    fig.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def latent_t_theta_corr(t, theta, out_path, name, pnames):
    n_t, n_p = t.shape[1], theta.shape[1]
    C = np.zeros((n_t, n_p))
    for i in range(n_t):
        for j in range(n_p):
            C[i, j] = abs(np.corrcoef(t[:, i], theta[:, j])[0, 1])
    fig, ax = plt.subplots(figsize=(1.3 * n_p + 3, 0.7 * n_t + 2))
    im = ax.imshow(C, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_p)); ax.set_xticklabels(pnames, rotation=30, ha="right")
    ax.set_yticks(range(n_t)); ax.set_yticklabels([rf"$t_{{{i+1}}}$" for i in range(n_t)])
    for i in range(n_t):
        for j in range(n_p):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] > 0.5 else "black", fontsize=8)
    ax.set_title(f"{name}: |corr(t, theta)|")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def latent_diagnostics(arm, out_dir):
    s = _load_summaries(arm)
    ldir = os.path.join(out_dir, "latent", re.sub(r"[^A-Za-z0-9]+", "_", arm["name"]))
    if s is None:
        print(f"[latent] {arm['name']}: no summaries/ dir found -- skipping", file=sys.stderr)
        return
    os.makedirs(ldir, exist_ok=True)
    t, theta = s["t"], s["theta"]
    pnames = [LABELS[j] for j in range(min(theta.shape[1], len(LABELS)))]
    stds = t.std(0)
    # collapse warning: a near-zero-variance t dim carries no information
    flat = np.where(stds < 1e-3)[0]
    flag = f"  [WARN collapsed dims: {flat.tolist()}]" if len(flat) else ""
    print(f"[latent] {arm['name']}: t.shape={t.shape} per-dim std={np.round(stds,4).tolist()}{flag}")
    latent_tt_colored(t, theta, os.path.join(ldir, "tt_colored.pdf"), arm["name"], pnames)
    latent_t_theta_corr(t, theta, os.path.join(ldir, "t_theta_corr.pdf"), arm["name"], pnames)
    print(f"[latent] {arm['name']}: wrote latent diagnostics -> {ldir}")


# ============================================================ driver
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--item", action="append", default=[], metavar="CFG|FAM|SCOPE[|LABEL]",
                     help="repeatable. e.g. --item 'configs/n1.yaml|nsf|std' "
                          "--item 'configs/n2.yaml|nsf|std|noise x2'. "
                          "Empty fields fall back to the yaml's nle.model / "
                          "nle.standardize.")
    src.add_argument("--arms-json", nargs="+", help="{name,dir,n_params} json(s)")
    ap.add_argument("--select", nargs="*", default=None,
                    help="substrings/regexes to keep (e.g. nsf standard_t)")
    ap.add_argument("--labels", nargs="*", default=None, help="legend labels, in order")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--thin", type=int, default=11)
    ap.add_argument("--burnin-frac", type=float, default=0.0)
    ap.add_argument("--dlogp", type=float, default=10.0,
                    help="logp>max-dlogp filter for GV/sigma ONLY; SBC is always "
                         "unfiltered. -1 disables (widths on full chain).")
    ap.add_argument("--sbc-unfiltered", action="store_true",
                    help="compute SBC ranks on the FULL chain while widths stay "
                         "filtered. Default: everything uses the --dlogp cut, "
                         "matching the [filtered_dlogp10] figures in the report.")
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--no-intersect", action="store_true",
                    help="evaluate each arm on its own sims (SBC uses more data, "
                         "but arms are then NOT strictly comparable)")
    ap.add_argument("--corner-sim", default="median",
                    help="'median', 'random', an integer sim id, or 'none'")
    ap.add_argument("--corner-bins", type=int, default=40)
    ap.add_argument("--corner-smooth", type=float, default=1.0)
    ap.add_argument("--corner-width", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-seeds", action="store_true",
                    help="expand each --item to every {arm}_s<seed> run found on disk")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="restrict --all-seeds expansion to these seeds (default: all found)")
    ap.add_argument("--latent", action="store_true",
                    help="also emit latent-space diagnostics per arm (t-t degeneracy "
                         "scatter, |corr(t,theta)| heatmap, per-dim collapse check) from "
                         "the arm's summaries/ dir. Off by default.")
    args = ap.parse_args()

    arms = load_arms(args)
    if not arms:
        sys.exit("no arms matched")
    os.makedirs(args.out, exist_ok=True)
    print("arms:")
    for a in arms:
        print(f"   {a['label']:28s} <- {a['name']}  ({a['dir']})")

    sims_keep = None
    if not args.no_intersect:
        sims_keep = common_sims(arms)
        print(f"shared sims across arms: {len(sims_keep)}")
        if not sims_keep:
            print("[warn] no shared sims; per-arm sims", file=sys.stderr)
            sims_keep = None

    dlogp = None if args.dlogp < 0 else args.dlogp
    if dlogp is None:
        mode_txt = "unfiltered"
        sbc_tag = "unfiltered"
    elif args.sbc_unfiltered:
        mode_txt = "SBC unfiltered; widths dlogp<%g" % dlogp
        sbc_tag = "unfiltered"
    else:
        mode_txt = f"filtered dlogp{dlogp:g}"
        sbc_tag = f"filtered_dlogp{dlogp:g}"
    print(f"[mode] {mode_txt}")

    report = []
    for a in arms:
        r = eval_arm(a, args.burnin_frac, args.thin, dlogp, args.nbins, sims_keep,
                     sbc_unfiltered=args.sbc_unfiltered)
        if r is None:
            print(f"[warn] {a['label']}: no usable chains", file=sys.stderr)
            continue
        r["label"] = a["label"]
        report.append(r)
    if not report:
        sys.exit("no arm produced usable chains")

    json.dump({"arms": arms, "report": report},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    write_csv(report, os.path.join(args.out, "metrics.csv"), dlogp, mode_txt)
    write_tex(report, os.path.join(args.out, "metrics.tex"))
    print_summary(report)

    # ---- publication SBC (filtered) ----
    plot_sbc(report, os.path.join(args.out, f"sbc_{sbc_tag}.pdf"),
             args.nbins, mode_txt)

    # ---- publication overlay corner (arms overlaid on one shared sim) ----
    made_corner = []
    if str(args.corner_sim).lower() != "none":
        try:
            explicit = int(args.corner_sim); cmode = None
        except ValueError:
            explicit, cmode = None, args.corner_sim
        sim, maps = pick_corner_sim(arms, cmode, args.seed, args.thin,
                                    args.burnin_frac, explicit, dlogp)
        cpath = os.path.join(args.out, f"corner_sim{sim}.pdf")
        plot_corner(arms, maps, sim, cpath,
                    args.thin, args.burnin_frac, args.corner_bins,
                    args.corner_smooth, args.corner_width, dlogp)
        made_corner.append(os.path.basename(cpath))

    # ---- optional latent-space diagnostics (compressor collapse / degeneracy) ----
    if args.latent:
        for a in arms:
            latent_diagnostics(a, args.out)

    outs = ["metrics.csv", "metrics.tex", "metrics.json", f"sbc_{sbc_tag}.pdf"] + made_corner
    if args.latent:
        outs.append("latent/<arm>/*.pdf")
    print(f"\ndone -> {args.out}/  ({', '.join(outs)})")


if __name__ == "__main__":
    main()