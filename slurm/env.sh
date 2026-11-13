# obspm tycho cluster (git-ignored)
export SBI_DATA=/data/ddehiwalage-don/data
export SBI_SUMMARIES=/gscratch/ddehiwalage-don/sbi_runs/BASELINE/baseline_pdf_ps/baseline_pdf_ps_Nnoise1000.npz
export SBI_SBC_TARGETS=/gscratch/ddehiwalage-don/sbi_runs/sbc_targets.npy
export SBI_OUT=/gscratch/ddehiwalage-don/sbi21cm_runs

export SBI_PYTHON=/data/ddehiwalage-don/conda_envs/torch_nadam/bin/python

# cpu: stages 1 (MLP), 3, 4 | gpu: CNN training/export | nle: stage 2
# (the paper runs used these exact environments)
sbi_env() {
    module purge
    if [[ "$1" == nle || ( "$1" == gpu && "$(hostname -s)" == tycho91 ) ]]; then
        [[ "$1" == gpu ]] && module load cuda/12.0
        module load anaconda/2024.10-1
        source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch_gpu
    else
        [[ "$1" == gpu ]] && module load cuda/11.8
        module load anaconda/2024.10-1
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate /data/ddehiwalage-don/conda_envs/torch_nadam
    fi
}

# old CPUs (pre-2021) and GPU nodes 57/58/83/91 are excluded from CPU stages
OLD="tycho[05-06,09-14,16-17,20-24,27-36,38-45,48-49,53-54,56-60,83,91]"
export S1_EXCLUDE="tycho28,tycho35,tycho57,tycho58,tycho60,tycho68,tycho70,tycho74,tycho83,tycho88,tycho91"
export S2_EXCLUDE="$OLD"
export S3_EXCLUDE="$OLD,tycho68,tycho70,tycho88"
