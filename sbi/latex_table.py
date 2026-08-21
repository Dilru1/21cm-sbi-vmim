#!/usr/bin/env python3
r"""Publication-quality LaTeX table from eval.py's report dict (or metrics.csv).

Design choices, because a 13x10 dump is not a table:
  * GV and every sigma: scientific notation, 2 sig figs, as $a.b\times10^{c}$.
  * calibration: fixed 2 decimals (it's an O(0.1) chi2-to-flat scalar, not sci).
  * BEST value per column is \textbf{}: min for GV/sigma (tighter=better) and
    for calib (flatter=better). Ties bold all winners.
  * booktabs rules, siunitx-free (portable: no extra package needed).
  * arm names sanitised: 'mlp_new/pdf[raw_t_gmm]' -> readable, math-safe.
  * one table PER MODE (unfiltered / filtered) -- never mixed, since a reader
    compares within a mode.

Use as a library (from eval.py):     write_latex(report, f"{out}/metrics.tex")
Use standalone on an existing CSV:   python latex_table.py metrics.csv -o metrics.tex
"""

import argparse
import csv
import math
import re

# GV columns get sci-notation; calib columns get fixed decimals. Everything is
# discovered from the header so this tracks eval.py's COMMON_PARAMS without edits.
CALIB_PREFIX = "calib_"
SIG_PREFIX = "sigma_"
GV_COLS = ("GV_4x4", "GV_4x4_mean")


def sci(x, sig=2):
    r"""2-sig-fig scientific as LaTeX math: 1.14e-06 -> $1.1\times10^{-6}$."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    x = float(x)
    if x == 0:
        return r"$0$"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10.0**exp
    mant = round(mant, sig - 1)
    if mant >= 10.0:  # rounding pushed 9.96 -> 10.0
        mant /= 10.0
        exp += 1
    return rf"${mant:.{sig - 1}f}\times10^{{{exp}}}$"


def fixed(x, dp=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{float(x):.{dp}f}"


def _short_param(label):
    """'sigma_$\\log_{10}(F_x)$' / 'calib_$\\tau$' -> the math label alone."""
    for pre in (SIG_PREFIX, CALIB_PREFIX):
        if label.startswith(pre):
            return label[len(pre) :]
    return label


def _clean_arm(name):
    r"""'mlp_new/pdf[standard_t_nsf]' -> 'pdf \textsubscript{std, nsf}'-ish,
    but kept simple and math-safe: escape underscores, keep the [scope_family]."""
    # split trailing [scope_family] if present
    m = re.match(r"^(.*?)\[(.*?)\]$", name)
    if m:
        base, tag = m.group(1), m.group(2)
    else:
        base, tag = name, ""
    base = base.split("/")[-1]  # drop 'mlp_new/'

    def esc(s):
        return s.replace("_", r"\_")

    if tag:
        # standard_t_gmm -> std/gmm ; raw_t_maf -> raw/maf
        tag = tag.replace("standard_t_", "std/").replace("raw_t_", "raw/")
        return rf"\texttt{{{esc(base)}}} [{esc(tag)}]"
    return rf"\texttt{{{esc(base)}}}"


def _rows_from_report(report):
    """Yield (mode, [row_dicts]) from eval.py's in-memory report dict."""
    from_eval = []
    # LABELS order in eval.py; rebuild the header the CSV writer would produce
    for mode, arms in report.items():
        rows = []
        for r in arms:
            d = {
                "arm": r["name"],
                "n_used": r["used"],
                "GV_4x4": r["gv"],
                "GV_4x4_mean": r["gv_mean"],
            }
            # sigma_* / calib_* keyed by the SAME LaTeX labels eval.py uses
            from eval import COMMON_PARAMS, LABELS  # reuse the single source

            for j in COMMON_PARAMS:
                d[f"{SIG_PREFIX}{LABELS[j]}"] = r["sigma"][COMMON_PARAMS.index(j)]
                d[f"{CALIB_PREFIX}{LABELS[j]}"] = r["calib"][COMMON_PARAMS.index(j)]
            rows.append(d)
        from_eval.append((mode, rows))
    return from_eval


def _rows_from_csv(path):
    reader = list(csv.DictReader(open(path)))
    modes = {}
    for r in reader:
        modes.setdefault(r["mode"], []).append(r)
    return list(modes.items())


def _numeric_cols(header_like):
    sig = [c for c in header_like if c.startswith(SIG_PREFIX)]
    cal = [c for c in header_like if c.startswith(CALIB_PREFIX)]
    return list(GV_COLS), sig, cal


def _best_mask(rows, cols):
    """For each col, index set of row(s) holding the min (best) finite value."""
    best = {}
    for c in cols:
        vals = []
        for i, r in enumerate(rows):
            try:
                v = float(r[c])
                if math.isfinite(v):
                    vals.append((v, i))
            except (TypeError, ValueError):
                pass
        if not vals:
            best[c] = set()
            continue
        m = min(v for v, _ in vals)
        best[c] = {i for v, i in vals if v == m}
    return best


def _emit_one(mode, rows, compact):
    gv_cols, sig_cols, cal_cols = _numeric_cols(rows[0].keys())
    num_cols = gv_cols + sig_cols + cal_cols
    best = _best_mask(rows, num_cols)

    # compact: drop GV_mean (keep the median GV, which is the reference stat)
    if compact and "GV_4x4_mean" in gv_cols:
        gv_cols = ["GV_4x4"]
        num_cols = gv_cols + sig_cols + cal_cols

    1 + len(num_cols)
    colspec = "l" + "r" * len(num_cols)

    def cell(r, c, i):
        raw = r[c]
        is_best = i in best.get(c, ())
        if c.startswith(CALIB_PREFIX):
            s = fixed(raw)
            return rf"\textbf{{{s}}}" if is_best else s  # plain number: \textbf ok
        # sci() returns '$..\times10^{..}$'; bold math must be \boldmath, not
        # \textbf{$..$} (which errors under many engines). Wrap the math body.
        s = sci(raw)
        if is_best and s.startswith("$") and s.endswith("$"):
            return r"$\boldsymbol{" + s[1:-1] + "}$"
        return s

    head_gv = [r"$\mathrm{GV}$"] + ([] if compact else [r"$\overline{\mathrm{GV}}$"])
    head_gv = head_gv[: len(gv_cols)]
    head_sig = [rf"$\sigma_{{{_short_param(c).strip('$')}}}$" for c in sig_cols]
    head_cal = [rf"$\chi^2_{{{_short_param(c).strip('$')}}}$" for c in cal_cols]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    safe_mode = mode.replace("_", r"\_")
    lines.append(
        rf"  \caption{{Posterior width and calibration, {safe_mode}. "
        r"$\mathrm{GV}=\mathrm{median}\,\sqrt{\det\Sigma_{4\times4}}$; "
        r"$\sigma$ is the median half-68\% interval; "
        r"$\chi^2$ is the per-parameter rank-flatness statistic "
        r"(smaller is better throughout). Best per column in bold.}"
    )
    lines.append(rf"  \label{{tab:eval_{re.sub(r'[^a-z0-9]+', '_', mode.lower())}}}")
    lines.append(r"  \begin{tabular}{" + colspec + "}")
    lines.append(r"    \toprule")
    # two-level header: group sigma and calib
    grp = [
        r"Arm",
        r"\multicolumn{%d}{c}{GV}" % len(gv_cols) if not compact else "GV",
        r"\multicolumn{%d}{c}{$\sigma$ (width)}" % len(sig_cols),
        r"\multicolumn{%d}{c}{$\chi^2$ (calibration)}" % len(cal_cols),
    ]
    lines.append("    " + " & ".join(grp) + r" \\")
    # cmidrule under the two groups
    gv_n = len(gv_cols)
    a = 2 + gv_n
    lines.append(
        rf"    \cmidrule(lr){{{a}-{a + len(sig_cols) - 1}}} "
        rf"\cmidrule(lr){{{a + len(sig_cols)}-{a + len(sig_cols) + len(cal_cols) - 1}}}"
    )
    sub = ["", *head_gv, *[h for h in head_sig], *[h for h in head_cal]]
    lines.append("    " + " & ".join(sub) + r" \\")
    lines.append(r"    \midrule")
    for i, r in enumerate(rows):
        cells = [_clean_arm(r["arm"])] + [cell(r, c, i) for c in num_cols]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_latex(report_or_csv, out_path, compact=True, from_csv=False):
    modes = _rows_from_csv(report_or_csv) if from_csv else _rows_from_report(report_or_csv)
    blocks = [_emit_one(mode, rows, compact) for mode, rows in modes if rows]
    with open(out_path, "w") as fh:
        fh.write("% auto-generated by latex_table.py -- needs \\usepackage{booktabs}\n")
        fh.write("\n\n".join(blocks) + "\n")
    return out_path


# ===========================================================================
#  BASELINE-COMPARISON layout: stats as ROWS, (summary set x {Base,models}) as
#  COLUMNS, powers of ten factored into the row labels. Matches the reference
#  paper's table so it drops straight into the report beside it.
# ===========================================================================
# reference paper's numbers, in the reference's OWN units: <V> is x10^7, the
# sigmas are raw. Edit here or override with --baseline-json.
DEFAULT_BASELINE = {
    "PS+PDF": {"V": 0.48, "Fx": 0.019, "tau": 0.052, "rHS": 0.035, "Mmin": 0.063},
    "PS": {"V": 1.33, "Fx": 0.024, "tau": 0.075, "rHS": 0.049, "Mmin": 0.087},
    "PDF": {"V": 7.53, "Fx": 0.0375, "tau": 0.080, "rHS": 0.060, "Mmin": 0.109},
}
CMP_ROWS = ["V", "Fx", "tau", "rHS", "Mmin"]
CMP_SCALE = {"V": 7, "Fx": 2, "tau": 2, "rHS": 2, "Mmin": 2}  # 10^ pulled to label
CMP_ROWLAB = {
    "V": r"\langle V\rangle",
    "Fx": r"\langle\sigma_{\log_{10}(F_X)}\rangle",
    "tau": r"\langle\sigma_{\tau}\rangle",
    "rHS": r"\langle\sigma_{r_{H/S}}\rangle",
    "Mmin": r"\langle\sigma_{\log_{10}(M_{\min})}\rangle",
}
# which CSV column feeds each stat row (V from GV; sigmas from sigma_* by index)
_SIG_ORDER = ["Fx", "tau", "rHS", "Mmin"]  # == COMMON_PARAMS order


def _canon_set(arm_name):
    """'mlp_new/pdf_ps[standard_t_gmm]' -> ('PS+PDF','gmm'). Maps the pipeline's
    set names to the reference paper's labels."""
    m = re.match(r"^(.*?)\[(.*?)\]$", arm_name)
    base, tag = (m.group(1), m.group(2)) if m else (arm_name, "")
    sset = base.split("/")[-1].lower()
    fam = (
        tag.replace("standard_t_", "").replace("raw_t_", "").replace("std/", "").replace("raw/", "")
    )
    scope = "std" if ("standard" in tag or tag.startswith("std")) else "raw"
    pretty = {"pdf": "PDF", "ps": "PS", "pdf_ps": "PS+PDF", "ps_pdf": "PS+PDF"}
    return pretty.get(sset, sset.upper()), f"{fam}/{scope}"


def _cmp_fmt(v, scale, prescaled=False):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    x = float(v) if prescaled else float(v) * (10.0**scale)
    if x == 0:
        return "0"
    dig = max(0, min(4, 2 - int(math.floor(math.log10(abs(x))))))
    return f"{x:.{dig}f}"


def write_compare(csv_path, out_path, baseline=None, mode_pick=None, scope_filter="std"):
    """Reference-style table from a metrics.csv. One column block per summary
    set: Base (reference) + one column per model family. scope_filter keeps only
    'std' or 'raw' t arms (default std) so the table stays narrow; None keeps all.
    """
    baseline = baseline or DEFAULT_BASELINE
    rows = list(csv.DictReader(open(csv_path)))
    if mode_pick:
        rows = [r for r in rows if r["mode"] == mode_pick]
    elif rows:
        # default: the filtered mode if present, else whatever's there
        modes = {r["mode"] for r in rows}
        pick = next((m for m in modes if "filtered" in m), sorted(modes)[0])
        rows = [r for r in rows if r["mode"] == pick]

    # gather: set -> {model_tag -> {stat -> value}}
    groups = {}
    sig_cols = [c for c in (rows[0].keys() if rows else []) if c.startswith(SIG_PREFIX)]
    for r in rows:
        sset, tag = _canon_set(r["arm"])
        if scope_filter and not tag.endswith("/" + scope_filter):
            continue
        fam = tag.split("/")[0]
        d = {"V": float(r["GV_4x4"])}
        for k, c in zip(_SIG_ORDER, sig_cols, strict=False):
            d[k] = float(r[c])
        groups.setdefault(sset, {})[fam] = d

    # order sets to match the reference; keep only sets we actually have
    set_order = [s for s in ["PS+PDF", "PS", "PDF"] if s in groups] + [
        s for s in groups if s not in ("PS+PDF", "PS", "PDF")
    ]

    # flat column list: per set -> Base (if baseline known) then sorted models
    flat = []
    for s in set_order:
        if s in baseline:
            flat.append((s, "Base"))
        for fam in sorted(groups[s]):
            flat.append((s, fam))

    def val(rk, s, kind):
        if kind == "Base":
            return (
                ("pre", baseline.get(s, {}).get(rk)) if rk == "V" else baseline.get(s, {}).get(rk)
            )
        return groups[s][kind].get(rk)

    # best (min) per row, in displayed units
    def disp(v, sc):
        if isinstance(v, tuple):  # prescaled baseline V
            return None if v[1] is None else float(v[1])
        return None if v is None else float(v) * 10.0**sc

    matrix = {rk: [val(rk, s, k) for (s, k) in flat] for rk in CMP_ROWS}
    best = {}
    for rk in CMP_ROWS:
        sc = CMP_SCALE[rk]
        dv = [(disp(v, sc), i) for i, v in enumerate(matrix[rk])]
        dv = [(x, i) for x, i in dv if x is not None]
        best[rk] = min(dv)[1] if dv else -1

    colspec = "l" + "".join(
        ("|c" if (i > 0 and flat[i][0] != flat[i - 1][0]) else "c") for i in range(len(flat))
    )
    L = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \caption{Posterior generalised variance $\langle V\rangle$ and "
        r"marginal widths per summary set: reference (\textbf{Base}) vs.\ the "
        r"learned NLE families. Powers of ten are in the row labels; smallest "
        r"per row in bold.}",
        r"  \label{tab:posterior_compare}",
        r"  \begin{tabular}{" + colspec + "}",
        r"    \toprule",
    ]
    # group header
    h1, i = ["Statistic"], 0
    while i < len(flat):
        s = flat[i][0]
        span = sum(1 for c in flat if c[0] == s)
        h1.append(rf"\multicolumn{{{span}}}{{c}}{{{s}}}")
        i += span
    L.append("    " + " & ".join(h1) + r" \\")
    cmid, start = [], 2
    for s in set_order:
        span = (1 if s in baseline else 0) + len(groups[s])
        cmid.append(rf"\cmidrule(lr){{{start}-{start + span - 1}}}")
        start += span
    L.append("    " + " ".join(cmid))
    h2 = [""] + [r"\textbf{Base}" if k == "Base" else rf"\texttt{{{k}}}" for _, k in flat]
    L.append("    " + " & ".join(h2) + r" \\")
    L.append(r"    \midrule")
    for rk in CMP_ROWS:
        sc = CMP_SCALE[rk]
        lab = rf"${CMP_ROWLAB[rk]}$\,($\times10^{{{sc}}}$)"
        cells = [lab]
        for idx, v in enumerate(matrix[rk]):
            pre = isinstance(v, tuple)
            s = _cmp_fmt(v[1] if pre else v, sc, prescaled=pre)
            cells.append(rf"\textbf{{{s}}}" if idx == best[rk] else s)
        L.append("    " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(out_path, "w") as fh:
        fh.write("% needs \\usepackage{booktabs}\n" + "\n".join(L) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="metrics.csv from eval.py")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--full", action="store_true", help="keep GV_mean column too")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="emit the reference-style baseline-comparison table "
        "(stats as rows, Base+models as columns) instead of the "
        "wide per-arm table.",
    )
    ap.add_argument(
        "--baseline-json",
        default=None,
        help="JSON of reference numbers per summary set; overrides the built-in Semelin defaults.",
    )
    ap.add_argument(
        "--scope",
        default="std",
        choices=["std", "raw", "all"],
        help="which t-scaling arms to include in --compare (default std)",
    )
    ap.add_argument(
        "--mode", default=None, help="which eval mode row to use (default: the filtered one)"
    )
    args = ap.parse_args()
    out = args.out or (args.csv.rsplit(".", 1)[0] + ("_compare.tex" if args.compare else ".tex"))
    if args.compare:
        bl = json.load(open(args.baseline_json)) if args.baseline_json else None
        write_compare(
            args.csv,
            out,
            baseline=bl,
            mode_pick=args.mode,
            scope_filter=None if args.scope == "all" else args.scope,
        )
    else:
        write_latex(args.csv, out, compact=not args.full, from_csv=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    import json

    main()
