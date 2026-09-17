# Pipeline figures (PlotNeuralNet + TikZ)

Five figures for the thesis, matching Chapter 4 (`4_model_method_algo.tex`)
and the `sbi/` code (`cnn_up.py`, `mlp.py`, `heads.py`, `nle.py`, `mcmc.py`).

| File | Figure | How it's made |
|---|---|---|
| `fig_pipeline_overview.(tex,pdf)` | **Figure 2.1 / `fig:pipeline`** — global 4-stage DAG | pure TikZ |
| `fig_mlp_compressor.(tex,pdf)` | Stage 1, path A: residual MLP on s ∈ R^30 + VMIM head | PlotNeuralNet `Box` layers, hand-written tex |
| `fig_cnn_compressor.(py,tex,pdf)` | Stage 1, path B: 3D SE-ResNet on cubes + VMIM head | PlotNeuralNet Python API (`pycore.tikzeng`) |
| `fig_nle_stage2.(tex,pdf)` | Stage 2: NLE q_ψ(t'|θ), GMM/MAF/NSF families | TikZ + PlotNeuralNet styles |
| `fig_mcmc_stage3.(tex,pdf)` | Stage 3: ensemble MCMC + log-posterior filter | pure TikZ |

## How to rebuild

This folder must sit INSIDE `PlotNeuralNet-master/` (it imports `../layers/`
and `../pycore/`). Then:

```bash
cd PlotNeuralNet-master/myfigs

# CNN figure: regenerate tex from Python, then compile
python3 fig_cnn_compressor.py
pdflatex fig_cnn_compressor.tex

# the others compile directly
pdflatex fig_mlp_compressor.tex
pdflatex fig_nle_stage2.tex
pdflatex fig_mcmc_stage3.tex
pdflatex fig_pipeline_overview.tex
```

Requirements: any TeX distribution with TikZ + amsmath/amssymb (TeX Live works).
No Python packages needed beyond the repo itself.

## The PlotNeuralNet workflow (recommended way to use the package)

1. Write a Python script that builds a list of layer strings with
   `pycore.tikzeng` helpers: `to_Conv`, `to_ConvRes`, `to_Pool`,
   `to_connection`, `to_skip`, ... and dump it with `to_generate(arch, "f.tex")`.
   Use box `height`/`depth` ∝ spatial size and `width` ∝ log(channels) so the
   shrinking-volume / growing-channels story is visible at a glance.
2. Anything the helpers can't express (pooling taps, dashed "training-only"
   enclosures, math annotations), inject as **raw TikZ strings** in the same
   `arch` list — every `Box` pic exports anchors like `name-east`,
   `name-south`, `name-northwest`, so you can hook custom arrows/nodes onto them.
3. Compile with `pdflatex`, look, tweak offsets, repeat.
   `fig_cnn_compressor.py` is a complete example of steps 1–2.

PlotNeuralNet shines for **conv nets** (3D boxes). For MLPs, flow diagrams,
and DAGs it is simpler to write TikZ directly while reusing the package's
`Box.sty` and colour macros for a consistent look — that is what the other
four files do.

## Dropping Figure 2.1 into the thesis

Copy `fig_pipeline_overview.pdf` to `chapters/figures/chap4/pipeline_dag.pdf`
and un-comment the includegraphics line already present in
`4_model_method_algo.tex`:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{chapters/figures/chap4/pipeline_dag.pdf}
  \caption{... (existing caption) ...}
  \label{fig:pipeline}
\end{figure}
```

Suggested labels for the detail figures: `fig:compressor_mlp`,
`fig:compressor_cnn`, `fig:nle`, `fig:mcmc`, referenced from
Sections 4.1.1–4.1.3.

## Things to check / easy to change

- **MLP widths**: the figure follows the thesis text ("three hidden layers of
  width 128"). The code default in `sbi/compressors/mlp.py` is
  `dense_layers=(256, 256, 128, 64)` — make text, code, and figure agree.
  Edit the three `xlabel={{128,}}` entries in `fig_mlp_compressor.tex`.
- **t dimension**: figures show t ∈ R^4 (thesis, k = dim(θ)); the code default
  is `t_dim=8`. Same remark.
- **CNN figure** follows `cnn_up.py` (stride-1 stem, RB1–RB2 at full 32³,
  strides 2 in RB3–RB5, mean+std read-out of RB2..RB5 → 1920 → 256 → 128 → t),
  which is the variant described in the thesis. `cnn.py` (all-stride-2,
  flatten-512 read-out) is the older baseline.
- Redshift channels annotated as z = 8.18, 10.32, 12.06 (from Chapter 3).
