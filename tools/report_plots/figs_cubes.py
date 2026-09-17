#!/usr/bin/env python3
import os
import numpy as np
import pyvista as pv

REDSHIFTS = ["12.06", "10.32", "8.18"]
BOX_SIZE = 32
DATA_DIR = "/data/ddehiwalage-don/data/dtb_data"
SIMIDS_PATH = "/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy"
OUT_DIR = "tools/report_plots/summary_figures"

def read_cube_components(z, sim_idx):
    clean_path = os.path.join(DATA_DIR, f"clean_cubes_z={z}.dat")
    noise_path = os.path.join(DATA_DIR, f"noise_cubes_100h_z={z}.dat")
    sim_ids = np.load(SIMIDS_PATH).astype(int)
    actual_file_id = sim_ids[sim_idx]
    
    cm = np.memmap(clean_path, dtype=np.float64, mode="r").reshape(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    clean_cube = np.asarray(cm[actual_file_id], dtype=np.float32)
    
    nm = np.memmap(noise_path, dtype=np.float64, mode="r").reshape(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    noise_cube = np.asarray(nm[actual_file_id], dtype=np.float32)
    
    return clean_cube, noise_cube

def make_pv_grid(data_array):
    grid = pv.ImageData()
    grid.dimensions = np.array(data_array.shape) + 1
    grid.spacing = (3.0, 3.0, 3.0) 
    grid.cell_data["dT_b"] = data_array.flatten(order="F")
    return grid

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    #pv.start_xvfb() 
    from cmastro import cmaps
    print(cmaps.keys())
    #dict_keys(['cma:hesperia', 'cma:hesperia_r', 'cma:lacerta', 
    #'cma:lacerta_r', 'cma:laguna', 'cma:laguna_r', 'cma:emph', 
    #'cma:emph_r', 'cma:unph', 'cma:unph_r'])
    
    cmap = cmaps["cma:emph"]

    opacity_transfer = [0.0, 0.1, 0.3, 0.6, 0.85, 1.0]

    # ============================================================
    # PHASE 1: COMPUTE GLOBAL DATA LIMITS ACROSS REDSHIFTS
    # ============================================================
    all_clean = []
    all_noise = []
    all_noised = []
    
    for z in REDSHIFTS:
        clean_cube, noise_cube = read_cube_components(z, sim_idx=0)
        clean_ms = clean_cube - clean_cube.mean(axis=(1, 2), keepdims=True)
        noised_ms = (clean_cube + noise_cube) - (clean_cube + noise_cube).mean(axis=(1, 2), keepdims=True)
        all_clean.append(clean_ms)
        all_noise.append(noise_cube)
        all_noised.append(noised_ms)
        
    clean_clim = [np.percentile(all_clean, 1.0), np.percentile(all_clean, 99.0)]
    noise_clim = [np.percentile(all_noise, 1.0), np.percentile(all_noise, 99.0)]
    noised_clim = [np.percentile(all_noised, 1.0), np.percentile(all_noised, 99.0)]

    # ============================================================
    # PLOT 1: COMPONENT SEPARATION VISUALIZER (2 Rows x 3 Cols)
    # ============================================================
    plotter1 = pv.Plotter(shape=(2, 3), window_size=[1600, 1000], border=False, off_screen=True)
    master_camera1 = pv.Camera()
    
    for iz, z in enumerate(REDSHIFTS):
        clean_cube, noise_cube = read_cube_components(z, sim_idx=0)
        clean_ms = clean_cube - clean_cube.mean(axis=(1, 2), keepdims=True)
        
        # Only show a single, clean colorbar on the last plot column panel (iz == 2)
        show_bar = (iz == 2)
        
        # Row 0: Clean Signal
        plotter1.subplot(0, iz)
        grid_c = make_pv_grid(clean_ms)
        plotter1.add_volume(
            grid_c, scalars="dT_b", cmap=cmap, opacity=opacity_transfer, 
            clim=clean_clim, blending="composite", show_scalar_bar=show_bar,
            scalar_bar_args={
                "title": "Clean Signal dT_b [mK]",
                "position_x": 0.93, "position_y": 0.55, "height": 0.38, "width": 0.025,
                "vertical": True, "font_family": "courier", "title_font_size": 10, "label_font_size": 8
            }
        )
        plotter1.add_bounding_box(color="black", line_width=1)
        plotter1.show_axes()
        plotter1.camera = master_camera1
        
        # Row 1: Thermal Noise
        plotter1.subplot(1, iz)
        grid_n = make_pv_grid(noise_cube)
        plotter1.add_volume(
            grid_n, scalars="dT_b", cmap=cmap, opacity=opacity_transfer, 
            clim=noise_clim, blending="composite", show_scalar_bar=show_bar,
            scalar_bar_args={
                "title": "Thermal Noise [mK]",
                "position_x": 0.93, "position_y": 0.08, "height": 0.38, "width": 0.05,
                "vertical": True, "font_family": "courier", "title_font_size": 13, "label_font_size": 10
            }
        )
        plotter1.add_bounding_box(color="black", line_width=1)
        plotter1.show_axes()
        plotter1.camera = master_camera1

    plotter1.camera_position = 'iso'
    plotter1.camera.azimuth -= 25
    plotter1.camera.elevation += 12
    plotter1.enable_parallel_projection()
    plotter1.camera.zoom(0.95)
    
    out1 = os.path.join(OUT_DIR, "raycasted_3d_components.png")
    plotter1.screenshot(out1)
    plotter1.close()
    print(f"Exported Component Matrix -> {out1}")

    # ============================================================
    # PLOT 2: STANDALONE NOISED DATA OBSERVATION GRID (1 Row x 3 Cols)
    # ============================================================
    plotter2 = pv.Plotter(shape=(1, 3), window_size=[1650, 550], border=False, off_screen=True)
    master_camera2 = pv.Camera()
    
    for iz, z in enumerate(REDSHIFTS):
        clean_cube, noise_cube = read_cube_components(z, sim_idx=0)
        noised_ms = (clean_cube + noise_cube) - (clean_cube + noise_cube).mean(axis=(1, 2), keepdims=True)
        
        show_bar = (iz == 2)
        
        plotter2.subplot(0, iz)
        grid_m = make_pv_grid(noised_ms)
        plotter2.add_volume(
            grid_m, scalars="dT_b", cmap=cmap, opacity=opacity_transfer, 
            clim=noised_clim, blending="composite", show_scalar_bar=show_bar,
            scalar_bar_args={
                "title": "dT_b [mK]",
                "position_x": 0.88, "position_y": 0.15, "height": 0.8, "width": 0.05,
                "vertical": True, "title_font_size": 14, "label_font_size": 10
            }
        )
        plotter2.add_bounding_box(color="black", line_width=1)
        plotter2.show_axes()
        plotter2.camera = master_camera2

    plotter2.camera_position = 'iso'
    plotter2.camera.azimuth -= 25
    plotter2.camera.elevation += 12
    plotter2.enable_parallel_projection()
    plotter2.camera.zoom(0.90)
    
    out2 = os.path.join(OUT_DIR, "raycasted_3d_combined.png")
    plotter2.screenshot(out2, scale=2)
    plotter2.close()
    print(f"Exported Noised Observation Matrix -> {out2}")

if __name__ == "__main__":
    main()