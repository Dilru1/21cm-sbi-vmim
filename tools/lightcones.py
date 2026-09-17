#!/usr/bin/env python3
r"""Compose a compact, text-free grid of 3D data cubes and export separate colorbars.

Layout:
    [CLEAN]     [NOISE x 1]     [SIGNAL+NOISE 1]
                [NOISE x 2]     [SIGNAL+NOISE 2]
                [NOISE x 3]     [SIGNAL+NOISE 3]

Usage:
  python make_cube_equation_grid.py --out cube_eq_grid_compact
"""
from __future__ import annotations

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mpl
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# General plotting params
plt.rcParams.update({
    "svg.fonttype": "none",        
    "pdf.fonttype": 42,
})

N = 32
NB_SIMU_TOTAL = 9827

def get_cmap():
    """Helper to load the correct colormap consistently."""
    try:
        import cmastro
        return plt.get_cmap("cma:emph_r")
    except ImportError:
        return plt.get_cmap("magma")


# --------------------------------------------------- REAL data stack renderer
def render_data_stack(kind, out_png, scale=1.0, sim_idx=0, noise_idx=0, vmin=None, vmax=None):
    """Render 3 receding slices from real .dat files as a tightly cropped transparent PNG."""
    
    clean_tpl = "/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat"
    noise_tpl = "/data/ddehiwalage-don/data/dtb_data/noise_cubes_100h_z={z}.dat"
    sim_ids_path = "/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy"
    
    sim_ids = np.load(sim_ids_path).astype(int)
    actual_dat_index = sim_ids[sim_idx]
    slice_idx = 16
    cmap = get_cmap()

    # 1. Load and process slices
    slices = []
    redshifts = ["8.18", "10.32", "12.06"]
    
    for z in redshifts:
        clean_mm = np.memmap(clean_tpl.format(z=z), dtype=np.float64, mode="r").reshape(-1, N, N, N)[:NB_SIMU_TOTAL]
        noise_mm = np.memmap(noise_tpl.format(z=z), dtype=np.float64, mode="r").reshape(-1, N, N, N)
        
        c_slice = np.asarray(clean_mm[actual_dat_index, slice_idx], dtype=np.float32)
        n_slice = np.asarray(noise_mm[noise_idx, slice_idx], dtype=np.float32)
        
        if kind == "clean":
            slices.append(c_slice)
        elif kind == "noise":
            slices.append(n_slice * scale)
        elif kind == "noised":
            slices.append(c_slice + (n_slice * scale))

    # 2. Setup plotting parameters
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(slices, [0.5, 99.5])
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig = plt.figure(figsize=(2.5, 2.5))  
    ax = fig.add_subplot(111, projection="3d")
    
    # 3. Draw the surfaces
    for i, data in enumerate(slices):
        x, z_mesh = np.meshgrid(np.arange(N + 1), np.arange(N + 1))
        y = np.full_like(x, i * 8.0, dtype=float) 
        
        colors = cmap(norm(data))
        colors[..., 3] = 0.95 if i == 2 else 0.55
        
        ax.plot_surface(x, y, z_mesh, facecolors=colors, rstride=1, cstride=1,
                        shade=False, antialiased=False)
                        
    ax.view_init(elev=15, azim=-75)
    ax.set_box_aspect((1, 2.2, 1))
    ax.set_axis_off()
    
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_png, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return vmin, vmax

# ------------------------------------------------------------- Colorbar Exporter
def export_colorbar(vmin, vmax, out_file):
    """Exports a clean, standalone vertical colorbar as a PDF."""
    cmap = get_cmap()
    fig_cb = plt.figure(figsize=(1.2, 3.5), dpi=300)
    
    # Define exact axis placement to eliminate margins
    ax_cb = fig_cb.add_axes([0.2, 0.05, 0.3, 0.9]) 
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm_mpl.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cb = fig_cb.colorbar(sm, cax=ax_cb, orientation='vertical')
    cb.set_label(r"$\delta T_b$ [mK]", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    
    fig_cb.savefig(out_file, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_cb)
    print(f"Exported standalone colorbar: {out_file}")


# ------------------------------------------------------------- composition
def place_image(ax, path, xy, zoom):
    img = mpimg.imread(path)
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), xy, frameon=False,
                        box_alignment=(0.5, 0.5), pad=0)
    ax.add_artist(ab)
    return img


def compose(clean_png, noise_pngs, noised_pngs, out, zoom=0.35):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    xc = 1.0       # Clean cube
    xn = 3.5       # Noise cube
    xr = 6.0       # Noised cube

    y_rows = [3.8, 2.25, 0.7]
    y_clean = y_rows[1] 

    place_image(ax, clean_png, (xc, y_clean), zoom)

    for i, y in enumerate(y_rows):
        place_image(ax, noise_pngs[i], (xn, y), zoom)
        place_image(ax, noised_pngs[i], (xr, y), zoom)

    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out + ".svg", transparent=True, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(out + ".pdf", transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Successfully wrote text-free diagrams to {out}.svg and {out}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-idx", type=int, default=0, help="Index of the clean simulation to use.")
    ap.add_argument("--out", default="cube_eq_grid_compact")
    args = ap.parse_args()

    os.makedirs("_tmp_renders", exist_ok=True)
    scales = [1.0, 2.0, 3.0]
    
    noise_pngs = []
    noised_pngs = []
    clean_png = "_tmp_renders/tmp_clean.png"

    # 1. Render Clean Cube & Export its Colorbar
    print("Rendering REAL clean stack...")
    c_vmin, c_vmax = render_data_stack("clean", clean_png, sim_idx=args.sim_idx)
    export_colorbar(c_vmin, c_vmax, "colorbar_signal.pdf")

    # 2. Render Noise and Noised Cubes & Export Noise Colorbar
    print("Finding global colormap limits for noise...")
    n_vmin, n_vmax = render_data_stack("noise", "_tmp_renders/dummy.png", scale=3.0)
    export_colorbar(n_vmin, n_vmax, "colorbar_noise.pdf")
    
    for s in scales:
        print(f"Rendering stacks for scale = {s}...")
        n_png = f"_tmp_renders/tmp_noise_{s}.png"
        nd_png = f"_tmp_renders/tmp_noised_{s}.png"
        
        # Lock noise to the loudest noise bounds
        render_data_stack("noise", n_png, scale=s, vmin=n_vmin, vmax=n_vmax)
        
        # Lock noised to the clean bounds so structures don't visually vanish
        render_data_stack("noised", nd_png, scale=s, sim_idx=args.sim_idx, vmin=c_vmin, vmax=c_vmax)
        
        noise_pngs.append(n_png)
        noised_pngs.append(nd_png)

    # 3. Composite everything
    print("Compositing final compact grid figure...")
    compose(clean_png, noise_pngs, noised_pngs, args.out, zoom=0.35)

if __name__ == "__main__":
    main()