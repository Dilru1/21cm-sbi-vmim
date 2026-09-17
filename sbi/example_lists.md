# Example .list files

Format per line, comma separated. Blank lines and `#` comments are ignored:

    name, path[, n_params][, chains_subdir]

- **name** — legend label; must be unique across all lists in one run.
- **path** — a directory holding `*.dat` directly, or an arm root (then
  `chains/` is searched one level down).
- **n_params** — usually omit; read from the first `*_truth.npy`.
- **chains_subdir** — substring picking one `chains/<subdir>` when the arm has
  several (e.g. `standard_t_nsf`). Required only when ambiguous.

## baseline.list
```
# flat dirs of .dat, 5-param legacy chains
baseline pdf,     /gscratch/ddehiwalage-don/sbi_runs/baseline_pdf
baseline ps,      /gscratch/ddehiwalage-don/sbi_runs/baseline_ps
baseline pdf_ps,  /gscratch/ddehiwalage-don/sbi_runs/baseline_pdf_ps
```

## cnn_mse.list
```
# arm roots: chains/ has both standard_t_nsf and raw_t_nsf, so pick one
cnn mse seed 80, /gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/seed_n1_s80, , standard_t_nsf
cnn mse seed 81, /gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/seed_n1_s81, , standard_t_nsf
cnn mse seed 82, /gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/seed_n1_s82, , standard_t_nsf
```

## Run
```bash
python stage4_eval.py \
  --list baseline.list \
  --list cnn_mse.list \
  --out eval_aug30/baseline_vs_mse \
  --dlogp 10 --filtered-only
```

Add `--dry-run` to resolve and audit the directories without running eval.
