# Copy to slurm/env.sh (git-ignored) and edit for your cluster. Sourced by every sbatch
# script and by submit.sh, which must be run from the repository root.

# ---- data and output (expanded as ${VAR} inside the YAML configs) ----
export SBI_DATA=/path/to/data                       # cubes, params, sim ids (Path B)
export SBI_SUMMARIES=/path/to/pdf_ps_summaries.npz  # theta, t, original_sim_ids (Path A, baseline)
export SBI_SBC_TARGETS=/path/to/sbc_targets.npy     # [{'sim', 'offset'}, ...]
export SBI_OUT=/path/to/output

# ---- python used by submit.sh to read the configs ----
export SBI_PYTHON=python

# ---- environment activation: sbi_env cpu | gpu | nle  (stage 1 MLP/3/4 | CNN | stage 2) ----
sbi_env() {
    # module load ...; conda activate ...
    :
}

# ---- optional node exclusions passed on the sbatch command line ----
export S1_EXCLUDE=""
export S2_EXCLUDE=""
export S3_EXCLUDE=""
