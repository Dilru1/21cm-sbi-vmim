# report_plots/ — Chapter 3 scripts (two-path version)

Figure-generation **scripts** (no plots included) for Chapter 3, restructured
around the two data representations of the report:

  Path A  hand-crafted summaries (noised power spectrum + PDF l-moments)
          -> MLP + VMIM compressor arm
  Path B  raw clean cubes (3 x 32^3)
          -> 3D CNN + VMIM compressor arm

```
report_plots/
├── figs_manual_summaries.py  # Path A figures (reads the lmom_/noised_ps_ text tables)
├── figs_cubes.py             # Path B figures (memmaps the cube .dat files)
├── common.py                 # shared loaders + demo mocks
├── style.py                  # thesis-wide matplotlib style
├── pipeline_dag.tex          # standalone TikZ, TWO-BRANCH pipeline -> fig:pipeline
├── 3_issue_and_data.tex      # EDITED chapter, rebalanced around the two paths
└── README.md
```

## Figure ↔ label map

| script | output | label |
|---|---|---|
| figs_manual_summaries.py | ps_curves.pdf      | fig:ps_curves |
|                          | lmom_dist.pdf      | fig:lmom_dist |
|                          | summary_snr.pdf    | fig:summary_snr |
|                          | corr_structure.pdf | fig:corr_structure |
| figs_cubes.py            | cube_3d.pdf        | fig:cube_3d |
|                          | cube_slices.pdf + snr_table.tex | fig:cube_slices / tab:snr |
|                          | voxel_pdf.pdf      | fig:voxel_pdf |
|                          | skew_kurt.pdf      | fig:skew_kurt |
|                          | moments_vs_theta.pdf | fig:moments_vs_theta |
|                          | prior_design.pdf   | fig:prior_design |
| pipeline_dag.tex (pdflatex) | pipeline_dag.pdf | fig:pipeline |

## Cluster usage

Path A (text tables are large: ~10^7 rows/redshift; np.fromfile(sep=' ')
parses them in C, and --noise-stride / --max-sims bound memory):
```bash
python figs_manual_summaries.py \
  --ps   "/data/ddehiwalage-don/data/powerspectra/noised_ps_100h_z={z}.dat" \
  --lmom "/data/ddehiwalage-don/data/l_moments/lmom_100h_z={z}.dat" \
  --out figures --max-sims 3000 --noise-stride 10
```

Path B:
```bash
python figs_cubes.py --out figures --max-sims 1200 --sim-index 0
```
(all data paths default to /data/ddehiwalage-don/... from the pipeline configs;
override with --params/--sim-ids/--clean/--noise as needed)

DAG: `pdflatex pipeline_dag.tex`.

Single figures: `--figs ps_curves`, `--figs cube_3d`, etc.
Style preview without cluster data: add `--demo`.

## Assumptions to verify on the real files (flagged in code comments too)

1. **Row layout of the summary tables**: sim-major, 1000 noise rows per sim
   (the convention of your legacy view_pdf_distrib.py / view_ps_distrib.py).
   If a file has 9828 sims while the cubes pipeline keeps 9827, the param
   alignment handles it (rows without a masked param are dropped as invalid),
   but check the printed "valid params" count.
2. **k-bin values**: reconstructed from Semelin+25 (8 log bins, sqrt(2) steps,
   0.03–0.5 h/cMpc); ps_curves x-axis uses these geometric centres.
3. **l-moment column order**: assumed l1..l6 left to right; l-moment axis
   labels and the l1/mean-removal caption depend on it.
4. **tau units**: figures use code units (Myr); tab:dataset still quotes the
   papers' Gyr — reconcile before submission.
5. The (tau, Mmin) censored-region box in prior_design is a percentile-based
   guide; trace the true empty region on real data if you prefer.
