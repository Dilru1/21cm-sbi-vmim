# Command reference / lab notebook

> Scratchpad of the exact commands used to run the pipeline on the cluster.
> Preserved from the original `readme.md`. Paths are specific to the Tycho/obspm environment.

---

## Interactive Jobs

salloc -t 60 -p short -J Interase_DIL --ntasks=1 --cpus-per-task=8 --mem=96G srun --pty bash
salloc -t 10 -p short -J Interase_DIL --gres=gpu:1 --ntasks=1 --nodelist=tycho91 --cpus-per-task=1 --mem=16G srun --pty bash
salloc -t 60 -p short -J Interase_DIL --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=64G srun --pty bash
srun -t 60 -p short -J Interase_DIL --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --nodelist=tycho91 --mem=64G


## Load module
module purge
module load cuda/12.0
module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu

# Stage 1 Plots

### EX1
python tools/plot_training_compressor.py \
    "ns1 seed1 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1/seed_s0/nle \
    --names Fx tau rHS Mmin --out SBC_OUT/ex1/stage1_compressor.pdf

python tools/plot_training_compressor.py \
    "ns1 seed0 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1/seed_s0/nle \
    "ns2 seed1 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1/seed_s1/nle \
    "ns3 seed2 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1/seed_s2/nle \
    "ns4 seed3 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1/seed_s3/nle \
    "ns4 seed4 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1/seed_s3/nle \
    --names Fx tau rHS Mmin --out SBC_OUT/ex1/stage1_compressor_nojit.pdf

python tools/plot_training_compressor.py \
    "ns1 seed0 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter/seed_s0/nle \
    "ns2 seed1 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter/seed_s1/nle \
    "ns3 seed2 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter/seed_s2/nle \
    "ns4 seed3 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter/seed_s3/nle \
    "ns4 seed4 Jitter"=/gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter/seed_s3/nle \
    --names Fx tau rHS Mmin --out SBC_OUT/ex1/stage1_compressor_jit.pdf


python tools/plot_training_nle.py configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml --out nle_training_comparison.png


bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=0 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=1 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=2 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=3 "nsf" "std" 23
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=4 "nsf" "std" 1


bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=0 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=1 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=2 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=3 "nsf" "std" 1
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=4 "nsf" "std" 1



bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml -o compressor.init_seed=3 "nsf" "std" 23


python check_chains.py /gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1  --expected 908
python check_chains.py /gscratch/ddehiwalage-don/sbi_master/cnn_vmim_/n1_jitter  --expected 908

### ---------------------------- ################# ------------------ ################### -------------------------------------


sbatch slurm/stage4.sbatch list --name n1_sweep --family nsf --scope std configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml

sbatch slurm/stage4.sbatch list --name n1_sweep_no_jit --family nsf --scope std configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml







---
python tools/plot_training_nle.py configs/mse_/arm_cnn_mse_1.yaml --out nle_training_comparison_mse.png
python tools/plot_training_nle.py configs/mlp_/arm_mlp_ps.yaml --out nle_training_comparison_mlp_ps.png
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_1.yaml --out nle_training_comparison_vmim.png






python tools/plot_training_compressor.py \
  mse_cnn_noise_1=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp1/nle \
  mse_cnn_noise_2=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp2/nle \
  --names Fx tau rHS Mmin --out eval_report_mse/compressor_loss.png

#############COMPRESSOR TRAIN

#for mlp
python tools/plot_training_compressor.py \
    "VMIM pdf"=/gscratch/ddehiwalage-don/sbi_runs/mlp/pdf/nle \
    "VMIM ps"=/gscratch/ddehiwalage-don/sbi_runs/mlp/ps/nle \
    "VMIM pdf+ps"=/gscratch/ddehiwalage-don/sbi_runs/mlp/pdf_ps/nle \
    --names Fx tau rHS Mmin --dequant 2:0.1000 3:0.0813 --out sum_out/mlp/fig_comp_mlp.pdf

#for mse
python tools/plot_training_compressor.py \
    "MSE \$n_s{=}1\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp1/nle \
    "MSE \$n_s{=}2\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp2/nle \
    --names Fx tau rHS Mmin --loss-log --out sum_out/msecnn/fig_comp_mse.pdf

#for vmim
python tools/plot_training_compressor.py \
    "VMIM \$n_s{=}1\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/exp1/nle \
    "VMIM \$n_s{=}2\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/exp2/nle \
    --names Fx tau rHS Mmin --dequant 2:0.1000 3:0.0813 --out sum_out/vmimcnn/fig_comp_vmim.pdf


#############COMPRESSOR OUT
#for mlp
python tools/plot_latent_noise_diag.py \
        s_pdf=/gscratch/ddehiwalage-don/sbi_runs/mlp/pdf/summaries \
        s_ps=/gscratch/ddehiwalage-don/sbi_runs/mlp/ps/summaries \
        s_pdf_ps=/gscratch/ddehiwalage-don/sbi_runs/mlp/pdf_ps/summaries \
        --names Fx tau rHS Mmin --out sum_out/mlp/latent


#for cnn
python tools/plot_latent_noise_diag.py \
        MSE_N1=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp1/summaries \
        MSE_N2=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp2/summaries \
        VMIM_N1=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/exp1/summaries \
        VMIM_N2=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/exp2/summaries \
        --names Fx tau rHS Mmin --out latent


#############STAGE2NLE
#MSE Noise1 vs Noise2
python tools/plot_training_nle.py configs/mse_/arm_cnn_mse_1.yaml --out new_plots/nle_training_mse_n1.png

python tools/plot_training_nle.py configs/mse_/arm_cnn_mse_2.yaml --out new_plots/nle_training_mse_n2.png

#VMIM Noise1 vs Noise2 (With Jitter)

python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_jitter_n1.yaml --out new_plots/nle_training_vmim_jitter_n1.png
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_jitter_n2.yaml --out new_plots/nle_training_vmim_jitter_n2.png

#VMIM Noise1 vs Noise2 (Without Jitter)
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_no_jitter_n1.yaml --out new_plots/nle_training_vmim_no_jitter_n1.png
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_no_jitter_n2.yaml --out new_plots/nle_training_vmim_no_jitter_n2.png


python tools/plot_training_nle.py configs/mse_/arm_cnn_vmim_no_jitter_n1.yaml --out new_plots/nle_training_no_jitter_n1.png
python tools/plot_training_nle.py configs/mse_/arm_cnn_vmim_jitter_n1.yaml --out new_plots/nle_training_jitter_n1.png



python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_jitter_n1.yaml --out new_plots/nle_loss/loss_nle_n1_jitter_2.png
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_no_jitter_n1.yaml --out new_plots/nle_loss/loss_nle_n1_no_jitter_2.png
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_jitter_n2.yaml --out new_plots/nle_loss/loss_nle_n2_jitter_2.png
python tools/plot_training_nle.py configs/vmim_/arm_cnn_vmim_no_jitter_n2.yaml --out new_plots/nle_loss/loss_nle_n2_no_jitter_2.png


##### JITEER EFFECTS
#for vmim
python tools/plot_training_compressor.py \
    "VMIM Jitter theta2"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/exp1/nle \
    "VMIM No Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/nle \
    --names Fx tau rHS Mmin --dequant 2:0.1000 --out effect_of_jitter.pdf



SBATCH_DEPENDENCY=afterok:5038641 bash submit_nle_grid.sh configs/vmim_/arm_cnn_vmim_jitter_n2.yaml



#evaluate jitter effects


#for vmim
python tools/plot_training_compressor.py \
    "VMIM \$n_s{=}1\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n1/nle \
    "VMIM \$n_s{=}2\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n2/nle  \
    "VMIM \$n_s{=}1\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n1/nle \
    "VMIM \$n_s{=}2\$"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n2/nle  \
    --names Fx tau rHS Mmin --dequant 2:0.1000 3:0.0813 --out sum_out/vmimcnn/fig_comp_vmim.pdf





python tools/plot_training_compressor.py \
    "nojit ns1"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n1/nle \
    "nojit ns2"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n2/nle  \
    --names Fx tau rHS Mmin --out new_plots/fig_noise.pdf

##################################################################################
##################################################################################
############################                ######################################



python tools/plot_training_compressor.py \
    "nojit ns1"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n1/nle \
    "nojit ns2"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n2/nle  \
    --names Fx tau rHS Mmin --out new_plots/fig_noise_all.pdf

python tools/plot_training_compressor.py \
    "nojit ns1"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter_v2/n1/nle \
    "jit_0.1"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n1/nle \
    "jit_0.05"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter_0.05/n1/nle \
    --names Fx tau rHS Mmin --dequant 2:0.1000 --out new_plots/fig_jitter_2.pdf



python tools/plot_latent_island.py \
        VMIM_N1_jit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n1/summaries \
        VMIM_N1_nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n1/summaries \
        VMIM_N2_jit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n2/summaries \
        VMIM_N2_nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n2/summaries \
        --names Fx tau rHS Mmin --out new_plots/latent_vmim  --smear-overlay --n-smear 5 --max-sims 40

python tools/plot_latent_island.py \
    VMIM_N1_nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n1/summaries \
    VMIM_N1_jit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n1/summaries \
    VMIM_N2_nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n2/summaries \
    VMIM_N2_jit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n2/summaries \
    --names Fx tau rHS Mmin --out new_plots/latent_vmim \
    --corner --color-by rHS --max-sims 60

python tools/plot_latent_island.py \
        MSE_N1_nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp1/summaries \
        MSE_N2_nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_mse_up/exp2/summaries \
        --names Fx tau rHS Mmin --out new_plots/latent_mse \
        --smear-overlay --center-islands --n-smear 5 --max-sims 40 --smear-alpha 0.35





python tools/plot_latent_island.py \
  nojit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/no_jitter/n1/summaries \
  jit=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_up/jitter/n1/summaries \
  --names Fx tau rHS Mmin --grid-param rHS --corner --color-by rHS --out jitter\latent_jit






#21/07/2026

module purge
module load cuda/12.0
module load anaconda/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_gpu



python tools/plot_training_compressor.py \
    "nojit ns1 (100h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s0/nle \
    "nojit ns2 (25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n2_s0/nle  \
    "nojit ns3 (11.1h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n3_s0/nle  \
    "nojit ns4 (6.25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n4_s0/nle \
    "Jit ns1 (100h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s0/nle \
    "Jit ns2 (25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n2_jitter_s0/nle  \
    "Jit ns3 (11.1h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n3_jitter_s0/nle  \
    "Jit ns4 (6.25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n4_jitter_s0/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/noise_compare_nojit.pdf


#############################





# no-jitter ladder -> eval_reports/no_jitter_nsf/
sbatch slurm/stage4_noise.sbatch --name no_jitter_nsf --family nsf --scope std \
    configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml \
    configs_seeds/noise/arm_cnn_vmim_no_jitter_n2.yaml \
    configs_seeds/noise/arm_cnn_vmim_no_jitter_n3.yaml \
    configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml

sbatch slurm/stage4_noise.sbatch --name jitter_nsf --family nsf --scope std \
    configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml \
    configs_seeds/noise/arm_cnn_vmim_jitter_n2.yaml \
    configs_seeds/noise/arm_cnn_vmim_jitter_n3.yaml \
    configs_seeds/noise/arm_cnn_vmim_jitter_n4.yaml





python tools/plot_training_compressor.py \
    "ns1 (100h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s0/nle \
    "ns2 (25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n2_s0/nle \
    "ns3 (11.1h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n3_s0/nle \
    "ns4 (6.25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n4_s0/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/noise_compare_nojit.pdf


python tools/plot_training_compressor.py \
    "Jit ns1 (100h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s0/nle \
    "Jit ns2 (25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n2_jitter_s0/nle \
    "Jit ns3 (11.1h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n3_jitter_s0/nle \
    "Jit ns4 (6.25h)"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n4_jitter_s0/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/noise_compare_jit.pdf


python tools/plot_training_compressor.py \
    "ns1 No-Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s1/nle \
    "ns1 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s1/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/jit_vs_nojit_ns1_seed1.pdf



python tools/plot_training_compressor.py \
    "ns1 seed1 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s1/nle \
    "ns1 seed2 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s2/nle \
    "ns1 seed3 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s3/nle \
    "ns1 seed4 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s4/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/jit_ns1_seed1_4.pdf

python tools/plot_training_compressor.py \
    "ns1 seed1 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s1/nle \
    "ns1 seed2 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s2/nle \
    "ns1 seed1 NoJitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s1/nle \
    "ns1 seed2 NoJitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s2/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/jit_nojit_ns12.pdf





++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CONFIG=configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml
for s in 1 2; do
  sbatch --job-name="s1_jit_n1_s${s}" \
         slurm/stage1_cnn.sbatch "$CONFIG" -o compressor.init_seed=${s}
done
s
+++++++++++++++++++++++            +++++++++++++++++++++++++++++++++++
CONFIG=configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml
for s in 1 2 3 4; do
  sbatch --job-name="s1_jit_n1_s${s}" \
         slurm/stage1_cnn.sbatch "$CONFIG" -o compressor.init_seed=${s}
done
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for CONFIG in \
  configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml \
  configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml; do
  for s in 1 2 3; do
    tag=$(basename "$CONFIG" .yaml)
    sbatch --job-name="s1cpu_${tag}_s${s}" slurm/stage1_cnn.sbatch \
           "$CONFIG" -o compressor.init_seed=${s} -o compressor.num_workers=0
  done
done



sbatch slurm/stage1_cnn.sbatch  configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=1 -o compressor.num_workers=0

python tools/plot_training_compressor.py \
    "ns1 seed1 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s1/nle \
    "ns1 seed2 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s2/nle \
    "ns1 seed3 Jitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_jitter_s3/nle \
    "ns1 seed1 NoJitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s1/nle \
    "ns1 seed2 NoJitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s2/nle \
    "ns1 seed3 NoJitter"=/gscratch/ddehiwalage-don/sbi_runs/cnn_vmim_noise/n1_s3/nle \
    --names Fx tau rHS Mmin --out noise_ana/compressor_train/jit_vs_nojit_seeds1_3.pdf



#
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=1 "nsf" "std" 123
bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml -o compressor.init_seed=2 "nsf" "std" 123


bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml -o compressor.init_seed=1 -o nle.num_workers=0 "nsf" "std" 23

bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml -o compressor.init_seed=2 -o compressor.num_workers=0 -o nle.num_workers=0 "nsf" "std" 123


bash submit_nle_grid.sh configs_seeds/noise/arm_cnn_vmim_no_jitter_n4.yaml -o compressor.init_seed=1 "nsf" "std" 1







python stage1_compress.py configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml -o compressor.init_seed=0


python tools/plot_training_nle.py \
  configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml \
  configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml \
  --out noise_ana/loss_nle_n1_n2.png


sbatch slurm/stage4_noise.sbatch --name no_jitter_nsf --family nsf --scope std \
       configs_seeds/noise/arm_cnn_vmim_no_jitter_n1.yaml \
       configs_seeds/noise/arm_cnn_vmim_jitter_n1.yaml \
