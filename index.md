---
title: 21cm-sbi-vmim
---

# 21cm-sbi-vmim

Documentation for **21cm-sbi-vmim** — simulation-based inference of
reionization-era astrophysical parameters from the 21 cm signal. The project
compares **VMIM**-trained neural summaries against MSE and fixed summaries
(power spectrum / PDF) under increasing thermal noise and dequantization jitter,
using neural likelihood estimation and simulation-based calibration.

## The pipeline

```{mermaid}
flowchart LR
  A[Stage 1<br/>compress] --> B[Stage 2<br/>NLE]
  B --> C[Stage 3<br/>MCMC]
  C --> D[Stage 4<br/>eval]
```

Every arm (a compressor + inference variant) shares the identical stage-2/3/4
machinery, so any difference in the final posteriors is attributable to the
summary statistic alone.

## What's inside this book

- **Git + MLOps guide** — a from-scratch walkthrough of the repository and its
  tooling, assuming no prior experience.
- **MLOps roadmap** — the phased plan for reproducibility, experiment tracking,
  and automation (and what to deliberately skip).
- **Diagnostics & results** — a runnable control panel for the compressor/NLE
  training curves and the final SBC / corner / metrics evaluation.
- **Command reference** — the exact cluster commands used to reproduce the runs.

## Links

- Source code: [github.com/Dilru1/21cm-sbi-vmim](https://github.com/Dilru1/21cm-sbi-vmim)
