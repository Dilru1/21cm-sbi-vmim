#!/usr/bin/env python3
"""
Generate 3D stacked slice figures and independent colorbars for:
  - 1x Clean cubes
  - 3x Noise-only cubes (scales 1, 2, 3)
  - 3x Signal + Noise cubes (scales 1, 2, 3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mpl
from matplotlib.colors import Normalize

N = 32
NB_SIMU_TOTAL = 9827

def main():
    # =========================================================================
    # 1. SETUP & PATHS
    # =========================================================================
    clean_tpl = "/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat"
    noise_tpl = "/data/ddehiwalage-don/data/dtb_data/noise_cubes_100h_z={z}.dat"
    sim_ids_path = "/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy"
    
    if not os.path.exists(sim_ids_path):
        raise FileNotFoundError(f"Missing sim_ids file: {sim_ids_path}")
    sim_ids = np.load(sim_ids_path).astype(int)

    try:
        import cmastro
        cmap_name = "cma:emph_r"
        cmap = plt.get_cmap(cmap_name)
    except ImportError:
        print("Note: 'cmastro' not found. Falling back to 'magma'.")
        cmap_name = "magma"
        cmap = plt.get_cmap(cmap_name)

    # Rendering parameters
    intra_group_gap = 8.0  
    inter_group_gap = 100.0 
    redshifts = ["8.18", "10.32", "12.06"]
    num_cubes_per_z = 8
    slice_idx = 16

    # Define the 7 different runs requested
    configurations = [
        {"mode": "clean",     "scale": 1.0, "prefix": "clean"},
        {"mode": "noise",     "scale": 1.0, "prefix": "noise_scale1"},
        {"mode": "noise",     "scale": 2.0, "prefix": "noise_scale2"},
        {"mode": "noise",     "scale": 3.0, "prefix": "noise_scale3"},
        {"mode": "sig_noise", "scale": 1.0, "prefix": "sig_noise_scale1"},
        {"mode": "sig_noise", "scale": 2.0, "prefix": "sig_noise_scale2"},
        {"mode": "sig_noise", "scale": 3.0, "prefix": "sig_noise_scale3"},
    ]

    # =========================================================================
    # 2. MASTER LOOP FOR ALL CONFIGURATIONS
    # =========================================================================
    for cfg in configurations:
        mode = cfg["mode"]
        scale = cfg["scale"]
        prefix = cfg["prefix"]
        
        print(f"\n=======================================================")
        print(f"PROCESSING: {prefix.upper()}")
        print(f"=======================================================")
        
        all_slices = []
        plot_data = [] 

        # --- A. Pre-load data for this specific configuration ---
        for z in redshifts:
            clean_file = clean_tpl.format(z=z)
            noise_file = noise_tpl.format(z=z)
            
            # Memmap the files
            clean_memmap = np.memmap(clean_file, dtype=np.float64, mode="r").reshape(-1, N, N, N)[:NB_SIMU_TOTAL]
            noise_memmap = np.memmap(noise_file, dtype=np.float64, mode="r").reshape(-1, N, N, N)
            
            group_slices = []
            for cube_idx in range(num_cubes_per_z):
                actual_dat_index = sim_ids[cube_idx]
                
                # Extract slices
                clean_slice = np.asarray(clean_memmap[actual_dat_index, slice_idx], dtype=np.float32)
                noise_slice = np.asarray(noise_memmap[cube_idx, slice_idx], dtype=np.float32)
                
                # Apply the specific math for this configuration
                if mode == "clean":
                    final_slice = clean_slice
                elif mode == "noise":
                    final_slice = noise_slice * scale
                elif mode == "sig_noise":
                    final_slice = clean_slice + (noise_slice * scale)

                group_slices.append(final_slice)
                all_slices.append(final_slice)
                
            plot_data.append(group_slices)

        # Calculate global contrast (0.5 to 99.5 percentiles)
        global_vmin, global_vmax = np.percentile(all_slices, [0.5, 99.5])
        norm = Normalize(vmin=global_vmin, vmax=global_vmax)
        print(f"Global Color Scale: vmin={global_vmin:.2f}, vmax={global_vmax:.2f}")

        # --- B. FIGURE 1: THE 3D SLICES ---
        fig_cubes = plt.figure(figsize=(16, 3.5), dpi=400)
        ax = fig_cubes.add_subplot(111, projection='3d')

        def add_slice(y_pos, data_2d, alpha=1.0):
            x, z = np.meshgrid(np.arange(N + 1), np.arange(N + 1))
            y = np.full_like(x, y_pos, dtype=float)
            
            colors = cmap(norm(data_2d))
            colors[..., 3] = alpha
            
            ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, 
                            shade=False, antialiased=False)

        current_y = 0.0
        for group_slices in plot_data:
            if not group_slices: continue
            
            for cube_idx, data_2d in enumerate(group_slices):
                alpha_val = 0.9 if cube_idx == num_cubes_per_z - 1 else 0.4
                add_slice(current_y, data_2d, alpha=alpha_val)
                current_y += intra_group_gap
                
            current_y += inter_group_gap

        ax.view_init(elev=15, azim=-75)  
        ax.set_box_aspect((1, 3.5, 1))   
        ax.set_axis_off()                
        
        fig_cubes.tight_layout()
        out_file_cubes = f"{prefix}_stacked_groups.pdf"
        fig_cubes.savefig(out_file_cubes, transparent=True, bbox_inches='tight')
        plt.close(fig_cubes)
        print(f"Saved 3D slices to {out_file_cubes}")

        # --- C. FIGURE 2: THE COLORBAR ---
        fig_cb = plt.figure(figsize=(1.2, 3.5), dpi=250)
        ax_cb = fig_cb.add_axes([0.2, 0.05, 0.3, 0.9]) 
        
        sm = cm_mpl.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig_cb.colorbar(sm, cax=ax_cb, orientation='vertical')
        cb.set_label(r"$\delta T_b$ [mK]", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        
        out_file_cb = f"{prefix}_colorbar.pdf"
        fig_cb.savefig(out_file_cb, transparent=True, bbox_inches='tight')
        plt.close(fig_cb)
        print(f"Saved colorbar to  {out_file_cb}")

if __name__ == "__main__":
    main()