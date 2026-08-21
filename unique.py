"""Check all 5 astro parameters for hidden grid structure, in the SAME
normalized [0,1] coordinate the VMIM head actually sees (mirrors
CubeDataset.__init__'s params_norm transform exactly -- if that transform
ever changes, update this file to match, or the suggested half-spacings
will be wrong).

Usage:
    python grid_check.py /data/ddehiwalage-don/data/astro_params_masked_from_original.npy
"""

import sys

import numpy as np

PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/data/ddehiwalage-don/data/astro_params_masked_from_original.npy"
)

NAMES = ["fX (idx0)", "tau (idx1)", "rHS (idx2)", "Mmin (idx3)", "fesc (idx4, NOT inferred)"]


def normalize_column(raw, idx):
    """EXACT mirror of CubeDataset's params_norm transform per column."""
    if idx == 0:
        v = np.log10(raw / 0.00192)
        return (v - (-1.0)) / (1.0 - (-1.0))
    if idx == 1:
        v = np.log10(raw)
        return (v - np.log10(738.9)) / (np.log10(10504.7) - np.log10(738.9))
    if idx == 2:
        return raw.copy()  # already in [0,1], no transform
    if idx == 3:
        # NOTE: CubeDataset applies NO log10 here -- it assumes the raw
        # column is ALREADY log10(Mmin) (range ~[8, 9.6] matches
        # log10([1e8, 4e9])). If your raw file instead stores LINEAR Mmin,
        # this diagnostic (and CubeDataset itself) would be silently wrong
        # -- sanity check raw.min()/raw.max() below before trusting this.
        return (raw - 8.0) / (9.6 - 8.0)
    if idx == 4:
        return (raw - 0.05) / (0.5 - 0.05)
    raise ValueError(idx)


def auto_collapse(sorted_unique, small_count_cutoff=25, min_elbow_ratio=4.0):
    """Collapse near-duplicate floats into real grid points via an
    elbow-detected threshold -- but ONLY when there's evidence of actual
    float-noise contamination (many raw uniques, clear bimodal gap split).

    Guards against the failure mode where a parameter has an EXACT, already-
    coarse grid with no floating noise (e.g. rHS: {0, 0.2, ..., 1.0} stored
    as exact literals): with no noise cluster to separate from, elbow
    detection on a near-uniform gap array finds a spurious split and merges
    real, distinct grid points. In that regime we just trust np.unique
    directly -- no collapsing.
    """
    if len(sorted_unique) < 3:
        return sorted_unique, 0.0
    if len(sorted_unique) <= small_count_cutoff:
        # already a small, almost certainly exact grid -- don't touch it
        return sorted_unique, 0.0

    diffs = np.diff(sorted_unique)
    diffs_sorted = np.sort(diffs)
    ratios = diffs_sorted[1:] / np.clip(diffs_sorted[:-1], 1e-15, None)
    elbow = np.argmax(ratios)
    if ratios[elbow] < min_elbow_ratio:
        # no clear separation between "noise" and "real" gaps -> trust as-is
        return sorted_unique, 0.0
    threshold = np.sqrt(diffs_sorted[elbow] * diffs_sorted[elbow + 1])

    grid = [sorted_unique[0]]
    for v in sorted_unique[1:]:
        if v > grid[-1] + threshold:
            grid.append(v)
    return np.array(grid), threshold


def main():
    astro_param = np.load(PATH).astype(np.float64)
    print(f"loaded {PATH}  shape={astro_param.shape}\n")

    suggestions = {}
    for idx in range(5):
        raw = astro_param[:, idx]
        print(f"--- {NAMES[idx]} ---")
        print(f"  raw range: [{raw.min():.6g}, {raw.max():.6g}]")

        normed = normalize_column(raw, idx)
        u = np.unique(normed)
        print(f"  raw unique (normalized coord): {u.shape[0]}")

        grid, thresh = auto_collapse(np.sort(u))
        print(f"  auto-collapse threshold: {thresh:.6g}")
        print(f"  collapsed grid points: {grid.shape[0]}")
        if grid.shape[0] <= 20:
            print(f"  grid: {np.round(grid, 5)}")
        if grid.shape[0] > 1:
            spacing = np.diff(grid)
            print(
                f"  spacing (normalized): min={spacing.min():.5f} "
                f"max={spacing.max():.5f} mean={spacing.mean():.5f}"
            )
            half = spacing.min() / 2.0
            print(f"  suggested dequant half (this param): {half:.5f}")
            suggestions[idx] = half
        else:
            print("  single value / continuous -- no grid structure detected")
        print()

    print("=" * 60)
    print("paste-ready yaml `dequant:` block (n_params=4 -> idx 0..3 only,")
    print("fesc/idx4 excluded since it isn't one of the inferred parameters):")
    print("  dequant:")
    for idx in range(4):
        if idx in suggestions:
            print(f"    {idx}: {{half: {suggestions[idx]:.5f}}}   # {NAMES[idx]}")
        else:
            print(f"    # {NAMES[idx]}: no grid detected, skip")


if __name__ == "__main__":
    main()
