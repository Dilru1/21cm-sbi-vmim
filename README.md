# Simulation-based inference with learned summaries of 21-cm fields

Pipeline for comparing three ways of compressing simulated 21-cm data before neural
likelihood estimation:

| | input | compressor | objective |
|---|---|---|---|
| **PDF+PS reference** | 30 hand-crafted summaries (15 PDF + 15 power spectrum) | none | |
| **Path A** | the same 30 summaries | MLP to `t_dim` = 4, 8, 16 | VMIM |
| **Path B** | mean-removed noisy coeval cubes, 3 redshifts | 3D residual CNN to `t_dim` = 4 | VMIM or MSE |

Each compressed summary `t` goes through the same pipeline: a neural likelihood
`q(t | theta)`, MCMC posteriors for a fixed set of simulation-based-calibration (SBC)
targets, then calibration and constraining-power metrics.

## Layout

```
src/sbi/          library
  config.py       YAML loading (${VAR} expanded from the environment), run directories, seeding
  data.py         summary tables, simulation-level train/val split, SBC targets
  cubes.py        cube dataset: signal + scaled noise, mean removal, standardisation, augmentation, export table
  compressors.py  MLP, 3D CNN, Gaussian-mixture head q(theta | t)
  train.py        compressor training loop (VMIM / MSE)
  nle.py          neural spline flow likelihood q(t | theta)
  mcmc.py         stretch-move ensemble sampler with the (tau, Mmin) wedge prior
  evaluate.py     SBC ranks, calibration ratio, posterior volume
scripts/          stage1_compress.py  merge_shards.py  stage2_nle.py  stage3_mcmc.py  stage4_eval.py
configs/          one YAML per experiment (baseline/, patha/, pathb/) and stage-4 groupings (eval/)
slurm/            one sbatch script per stage, submit.sh, env.example.sh
tests/            smoke_test.py: the whole pipeline on tiny synthetic data
```

## Stages

A run is one config plus a compressor seed `--cseed` and an NLE seed `--nseed`. Every
config lists its seeds (the paper uses compressor seeds 43, 44, 45 and NLE seeds 42, 43, 44).

1. **Compress**: `scripts/stage1_compress.py CONFIG --cseed C` trains the compressor
   (400 epochs, final weights kept) and exports `t` for every row of the training table.
   For Path B the table is every simulation times `n_noise` deterministic noise draws.
   That export can be split over the noise axis (`--shard k --nshard n`) and then joined
   with `merge_shards.py`. The baseline skips this stage.
2. **Likelihood**: `scripts/stage2_nle.py CONFIG --cseed C --nseed N` trains a neural
   spline flow on standardised `t`.
3. **Posteriors**: `scripts/stage3_mcmc.py CONFIG --cseed C --nseed N --task k --ntasks n`
   runs 160 walkers for 2000 steps (500 burn-in) per SBC target.
4. **Evaluation**: `scripts/stage4_eval.py configs/eval/paper.yaml` gives SBC rank
   histograms, the calibration ratio (1 = ranks consistent with uniform) and the
   posterior volume `GV = median sqrt(det Cov)`. It evaluates on the simulations shared
   by all runs, after a stuck-walker cut, thinning by 11 and a 10-nat log-posterior cut.

Outputs go to `{out_root}/{name}/c{cseed}/` (`compressor/`, `summaries/`, `nle_n{nseed}/`,
`chains_n{nseed}/`). The baseline writes to `{out_root}/{name}/raw/`.

## Running

```bash
pip install -r requirements.txt
python tests/smoke_test.py                  # a few minutes on CPU

cp slurm/env.example.sh slurm/env.sh        # set data/output paths and environment activation
bash slurm/submit.sh configs/patha/vmim_k4.yaml         # every seed of one experiment
sbatch slurm/stage4.sbatch configs/eval/paper.yaml
```

Run all commands from the repository root.

## Data format

Configs refer to data through environment variables (`${SBI_DATA}`, `${SBI_SUMMARIES}`,
`${SBI_SBC_TARGETS}`, `${SBI_OUT}`) set in `slurm/env.sh`.

* **Summary table** (`.npz`): `theta` (rows, n_params), already mapped to [0, 1];
  `t` (rows, n_summaries); `original_sim_ids` (rows,). Each simulation has several rows,
  one per noise realisation.
* **Cubes**: one raw float64 file per redshift channel for the signal (row = simulation)
  and one for a pool of noise cubes, each of shape `box^3`. `params_path` holds the
  physical parameters, row-aligned with `sim_ids_path`. `data.params` maps each one to
  [0, 1] (linear or log10, with bounds).
* **SBC targets** (`.npy`, pickled list of dicts): `{'sim': id, 'offset': k}` selects row
  k of simulation `id` in the summary table.

### Other simulators

Nothing past stage 1 knows about 21 cm: stages 2-4 only read `theta`, `t` and the sim ids.
* **Hand-crafted summaries**: data from another simulator (21cmFAST, CAMELS 3D grids, etc.)
  runs through Path A as soon as it is written in the summary-table format.
* **Fields**: Path B needs signal and noise cubes in the format above. For noise-free
  data, use a noise file of zeros.
* **To check per simulator**: the parameter map (`data.params`), `n_params`, the
  `mcmc.use_wedge_prior` flag (the wedge is specific to the 21-cm parameter set) and the
  number of channels, which is inferred from `signal_paths`.

## Reproducibility

Under the same seeds and software versions, the code reproduces the original research
code: the same network initialisation and forward passes, and identical CPU training
trajectories. Training uses `cudnn.benchmark`, so GPU (Path B) runs are not bitwise
reproducible. Use `configs/eval/example_chains.yaml` to compare a rerun with earlier chains.

Note: the package directory is named `sbi`, as is the PyPI package `sbi` (mackelab).
Scripts put `src/` first on the path, so don't install both into the same environment.
