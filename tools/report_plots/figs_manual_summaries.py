#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- Constants
REDSHIFTS = ["12.06", "10.32", "8.18"]
BOX_SIZE = 32                      # Shape of downsampled grid cubes (32^3)
N_PIXELS = BOX_SIZE ** 3            # Total pixel count per grid
NB_SIMU_TOTAL = 9827                # Total simulations in legacy database
PS_NBINS = 8                        # Pre-computed log k-bins
N_NOISE_TOTAL = 1000                # Total noise realizations per sim inside the PS table

# Directories and explicit file templates
DATA_DIR = "/data/ddehiwalage-don/data/dtb_data"
PS_TPL = "/data/ddehiwalage-don/data/powerspectra/noised_ps_100h_z={z}.dat"
PARAM_PATH = "/data/ddehiwalage-don/data/astro_params_masked_from_original.npy"
SIMIDS_PATH = "/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy"
OUT_DIR = "tools/report_plots/summary_figures"
TEXTWIDTH_IN = 7.0

# Wavenumber k center values estimated from 0.03 to 0.5 h cMpc^-1
PS_KCENTERS = np.logspace(np.log10(0.03), np.log10(0.5), PS_NBINS)

# -------------------------------------------------------- Data Loading Helpers
def read_real_cubes_memmap(z, sim_ids):
    """
    Safely opens large flat binary grid files using a read-only memory map with float32.
    Extracts only the masked indices and casts them to memory-friendly float32.
    """
    clean_path = os.path.join(DATA_DIR, f"clean_cubes_z={z}.dat")
    noise_path = os.path.join(DATA_DIR, f"noise_cubes_100h_z={z}.dat")
    
    if not os.path.exists(clean_path) or not os.path.exists(noise_path):
        raise FileNotFoundError(f"Missing expected grid data arrays for z={z} in {DATA_DIR}")
        
    cm = np.memmap(clean_path, dtype=np.float64, mode="r").reshape(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)[:NB_SIMU_TOTAL]
    clean_cubes = np.asarray(cm[sim_ids], dtype=np.float32)\

    nm = np.memmap(noise_path, dtype=np.float64, mode="r").reshape(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    noise_single = np.asarray(nm[:len(sim_ids)], dtype=np.float32)

    return clean_cubes + noise_single


def load_noise_averaged_ps_text(z, sim_ids):
    """
    Streams the text-based .dat power spectrum file row-by-row.
    Collects only the rows belonging to the masked sim_ids, groups them,
    and returns the noise-averaged array profile.
    """
    ps_path = PS_TPL.format(z=z)
    if not os.path.exists(ps_path):
        raise FileNotFoundError(f"Power spectrum table file missing: {ps_path}")
        
    # Build a fast lookup set of absolute row ranges we want to keep
    # Each sim_id occupies a contiguous chunk of N_NOISE_TOTAL text rows
    wanted_rows = set()
    for sid in sim_ids:
        start_row = sid * N_NOISE_TOTAL
        for r in range(start_row, start_row + N_NOISE_TOTAL):
            wanted_rows.add(r)
            
    kept_lines = []
    print(f"  Streaming text file lines for z={z} (filtering via masked sim IDs)...")
    
    with open(ps_path, "r") as f:
        for idx, line in enumerate(f):
            if idx in wanted_rows:
                kept_lines.append(line)
                
    # Convert only our filtered text data to a clean numpy array
    print(f"  Converting parsed text buffer to numpy matrix array...")
    extracted_data = np.loadtxt(kept_lines, dtype=np.float32)
    
    # Reshape structure to group by masked sims: (n_masked_sims, 1000, 8)
    n_masked_sims = len(sim_ids)
    extracted_data = extracted_data.reshape(n_masked_sims, N_NOISE_TOTAL, PS_NBINS)
    
    # Average along the 1000 noise realizations axis -> output shape: (n_masked_sims, 8)
    return extracted_data.mean(axis=1)


# ------------------------------------------------------------- Main Pipeline
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load astronomical model parameters and indexing layouts
    params = np.load(PARAM_PATH).astype(np.float64)
    sim_ids = np.load(SIMIDS_PATH).astype(int)
    n_sims = len(sim_ids)
    
    logfx = np.log10(params[:, 0])
    order = np.argsort(logfx)
    sorted_logfx = logfx[order]
    
    # Setup subplots structured into 1x3 grid per artifact matching redshift arrays
    fig_pdf, axes_pdf = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.8), sharey=True, constrained_layout=True)
    fig_ps, axes_ps = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.8), sharey=True, constrained_layout=True)
    
    norm_fx = mpl.colors.Normalize(np.nanmin(logfx), np.nanmax(logfx))
    #cmap_global = mpl.colormaps["plasma"]
    #for colormap
    from cmastro import cmaps
    print(cmaps.keys())
    #dict_keys(['cma:hesperia', 'cma:hesperia_r', 'cma:lacerta', 
    #'cma:lacerta_r', 'cma:laguna', 'cma:laguna_r', 'cma:emph', 
    #'cma:emph_r', 'cma:unph', 'cma:unph_r'])
    
    cmap_global = cmaps["cma:unph"]
    
    x_bins_pdf = np.linspace(-250, 50, 100)
    
    for iz, z in enumerate(REDSHIFTS):
        print(f"Analyzing and processing arrays for Redshift z = {z}...")
        
        # --- Task 1: Generate Raw Voxel PDF Data via Binary Grid Cubes ---
        cubes = read_real_cubes_memmap(z, sim_ids)
        all_pdf_profiles = []
        
        for sim_idx in range(n_sims):
            cube = cubes[sim_idx]
            # Plot the RAW distribution profile directly without mean subtraction
            counts, _ = np.histogram(cube.ravel(), bins=x_bins_pdf, density=True)
            all_pdf_profiles.append(counts)
            
        all_pdf_profiles = np.array(all_pdf_profiles)[order]
        
        # Plotting Voxel Intensity PDF Maps
        ax_pdf = axes_pdf[iz]
        ax_pdf.set_title(f"z = {z}")
        ax_pdf.set_xlabel(r"$\delta T_b$ [mK]")
        for j in range(n_sims):
            ax_pdf.plot(0.5 * (x_bins_pdf[1:] + x_bins_pdf[:-1]), all_pdf_profiles[j], 
                        lw=0.35, alpha=0.35, color=cmap_global(norm_fx(sorted_logfx[j])), rasterized=True)
            
        # --- Task 2: Power Spectrum Curves Map (Mirrors PDF Style Representation) ---
        mean_ps = load_noise_averaged_ps_text(z, sim_ids)
        all_ps_profiles = mean_ps[order]  # Shape: (n_sims, 8)
        
        ax_ps = axes_ps[iz]
        ax_ps.set_title(f"z = {z}")
        ax_ps.set_xlabel(r"$k$ [$h\,$cMpc$^{-1}$]")
        ax_pdf.set_ylim(bottom=0.0)
        ax_pdf.grid(True, which="both", alpha=0.30)
        
        # Plot one continuous curve per simulation, sorted and color-mapped by log10(f_X)
        for j in range(n_sims):
            ax_ps.plot(PS_KCENTERS, all_ps_profiles[j], 
                        lw=0.35, alpha=0.35, color=cmap_global(norm_fx(sorted_logfx[j])), rasterized=True)
            
        ax_ps.set_xscale("log")
        ax_ps.set_yscale("log")
        ax_ps.grid(True, which="both", alpha=0.30)

    # Render Visual Enhancements & Export Plots
    # 1. Save PDF Figure
    axes_pdf[0].set_ylabel("voxel PDF [mK$^{-1}$]")
    cb_pdf = fig_pdf.colorbar(mpl.cm.ScalarMappable(norm=norm_fx, cmap=cmap_global), ax=axes_pdf, pad=0.012, aspect=28)
    cb_pdf.set_label(r"$\log_{10} f_X$")
    fig_pdf.savefig(os.path.join(OUT_DIR, "voxel_pdf_all_sims.pdf"), dpi=300, bbox_inches="tight")
    fig_pdf.savefig(os.path.join(OUT_DIR, "voxel_pdf_all_sims.png"), dpi=300, bbox_inches="tight")
    print(f"Generated Figure 1 replica (Raw Values) -> {OUT_DIR}/voxel_pdf_all_sims.pdf")
    
    # 2. Save Power Spectrum Figure (Visually matching PDF structure)
    axes_ps[0].set_ylabel(r"noised $\Delta^2(k)$ [mK$^2$]")
    cb_ps = fig_ps.colorbar(mpl.cm.ScalarMappable(norm=norm_fx, cmap=cmap_global), ax=axes_ps, pad=0.012, aspect=28)
    cb_ps.set_label(r"$\log_{10} f_X$")
    fig_ps.savefig(os.path.join(OUT_DIR, "ps_curves_by_param.pdf"), dpi=300, bbox_inches="tight")
    fig_ps.savefig(os.path.join(OUT_DIR, "ps_curves_by_param.png"), dpi=300, bbox_inches="tight")

    print(f"Generated Matching Power Spectrum Curves -> {OUT_DIR}/ps_curves_by_param.pdf")

if __name__ == "__main__":
    main()