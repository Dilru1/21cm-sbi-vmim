"""
Isometric version of cube_c.py  --  TRUE 32x32x32 CUBES

Each redshift is drawn as a real cube (equal extent in x, y, z). The three
visible faces are the actual outer faces of the 32^3 volume:

    top face    = V[:, :, -1]     (the z = 31 plane)
    left  wall  = V[0, :, :]      (the x =  0 plane)
    right wall  = V[:, 0, :]      (the y =  0 plane)

so nothing on screen is invented -- the walls are data, not extrusion.
Only ONE noise level is shown (NOISE_SCALE below).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.cm as cm_mpl
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# =========================================================================
# 0. ISOMETRIC GEOMETRY KNOBS
# =========================================================================
ISO_ANGLE = 30.0          # 30 deg = true isometric
CUBE_L    = 1.0           # cube edge -- x, y and z all get this. It's a cube.
GAP       = 0.22          # empty space between consecutive cubes (units of L)

# How consecutive cubes are displaced. (dx, dy, dz) in units of L.
#   (0, 0, 1 + GAP)      -> vertical stack            (default)
#   (1 + GAP, 0, 0)      -> row receding to the right
#   (0, 0, -(1 + GAP))   -> vertical stack, order flipped
OFFSET = (0.0, 0.0, 1.0 + GAP)

SHADE_L   = 0.62          # brightness of the left  wall (x = 0 face)
SHADE_R   = 0.42          # brightness of the right wall (y = 0 face)
EDGE_LW   = 0.45          # outline weight on the cube silhouettes
NOISE_SCALE = 2.0         # <-- the single noise case (was scales = [1, 2, 3])

_C = np.cos(np.deg2rad(ISO_ANGLE))
_S = np.sin(np.deg2rad(ISO_ANGLE))


def iso(x, y, z):
    """Project a 3D point (x, y, z) onto the 2D isometric plane."""
    return (x - y) * _C, (x + y) * _S + z


def darken(cmap, f):
    """Return a copy of cmap with every colour multiplied by f."""
    cols = cmap(np.linspace(0, 1, 256))
    cols[:, :3] = np.clip(cols[:, :3] * f, 0, 1)
    return ListedColormap(cols)


# =========================================================================
# 1. HELPER: COLORBAR EXPORT   (unchanged)
# =========================================================================
# =========================================================================
# 1. HELPER: COLORBAR EXPORT   (Updated for Horizontal)
# =========================================================================
def export_colorbar(cmap, norm, label, out_file):
    # Swapped width and height for a horizontal layout
    fig_cb = plt.figure(figsize=(12 , 1.2), dpi=300) 
    
    # Adjusted axes [left, bottom, width, height] to fit horizontally
    ax_cb = fig_cb.add_axes([0.05, 0.4, 0.9, 0.25]) 
    
    sm = cm_mpl.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Changed orientation to horizontal
    cb = fig_cb.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    
    fig_cb.savefig(f"{out_file}.pdf", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig_cb.savefig(f"{out_file}.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_cb)
    print(f"Exported standalone horizontal colorbar: {out_file}")

# =========================================================================
# 2. LOAD REAL DATA -- FULL CUBES, NOT SINGLE SLICES
# =========================================================================
N = 32
clean_tpl = "/data/ddehiwalage-don/data/dtb_data/clean_cubes_z={z}.dat"
noise_tpl = "/data/ddehiwalage-don/data/dtb_data/noise_cubes_100h_z={z}.dat"
sim_ids_path = "/data/ddehiwalage-don/data/original_sim_ids_masked_from_original.npy"

# Order from Back to Front (Highest Z is furthest away)
redshifts = ["12.06", "10.32", "8.18"]

clean_cubes = []
noise_cubes = []

if os.path.exists(sim_ids_path):
    sim_ids = np.load(sim_ids_path).astype(int)
    for z in redshifts:
        c_mm = np.memmap(clean_tpl.format(z=z), dtype=np.float64, mode="r").reshape(-1, N, N, N)
        n_mm = np.memmap(noise_tpl.format(z=z), dtype=np.float64, mode="r").reshape(-1, N, N, N)
        clean_cubes.append(np.asarray(c_mm[sim_ids[0]], dtype=np.float32))   # (N, N, N)
        noise_cubes.append(np.asarray(n_mm[0], dtype=np.float32))            # (N, N, N)
else:  # ---- offline preview so the layout can be checked without the cluster
    print("[warn] data files not found - using synthetic stand-in volumes")
    rng = np.random.default_rng(3)
    f = np.fft.fftfreq(N)
    kk = np.sqrt(sum(a ** 2 for a in np.meshgrid(f, f, f, indexing="ij")))
    spec = np.exp(-(kk / 0.09) ** 2)
    for k in range(3):
        g = np.real(np.fft.ifftn(np.fft.fftn(rng.normal(size=(N, N, N))) * spec))
        clean_cubes.append(((g - g.mean()) / g.std() * 6 + 12 * (k + 1)).astype(np.float32))
        noise_cubes.append(rng.normal(0, 2.0, (N, N, N)).astype(np.float32))

try:
    import cmastro
    cmap = plt.get_cmap("cma:emph_r")
except ImportError:
    cmap = plt.get_cmap("magma")

# ---------------------------------------------------------
# DUAL NORMALIZATION & COLORBAR GENERATION
# ---------------------------------------------------------
# Percentiles are taken over the visible faces only, so the stretch matches
# what the reader actually sees rather than the hidden interior.
def visible_faces(V):
    return np.concatenate([V[:, :, -1].ravel(), V[0, :, :].ravel(), V[:, 0, :].ravel()])


vis_clean = np.concatenate([visible_faces(V) for V in clean_cubes])
vmin_sig, vmax_sig = np.percentile(vis_clean, [0.5, 99.5])
norm_sig = Normalize(vmin=vmin_sig, vmax=vmax_sig)
export_colorbar(cmap, norm_sig, r"$\delta T_b$ [mK]", "colorbar_signal")

noise_only = [V * NOISE_SCALE for V in noise_cubes]
sig_plus_noise = [c + n for c, n in zip(clean_cubes, noise_only)]

vis_noise = np.concatenate([visible_faces(V) for V in noise_only])
vmin_noise, vmax_noise = np.percentile(vis_noise, [0.5, 99.5])
norm_noise = Normalize(vmin=vmin_noise, vmax=vmax_noise)
export_colorbar(cmap, norm_noise, r"$\delta T_b$ [mK] (Noise)", "colorbar_noise")


# =========================================================================
# 3. THE ISOMETRIC CUBE RENDERER
# =========================================================================
def cube_origins(n):
    """World-space origin of each cube, index 0 = last drawn = top of stack."""
    dx, dy, dz = OFFSET
    return [(k * dx * CUBE_L, k * dy * CUBE_L, k * dz * CUBE_L) for k in range(n)]


def stack_bbox(n):
    """Projected bounding box of the whole stack, from all 8 corners of each cube."""
    xs, ys = [], []
    L = CUBE_L
    for ox, oy, oz in cube_origins(n):
        for cx in (0, L):
            for cy in (0, L):
                for cz in (0, L):
                    px, py = iso(ox + cx, oy + cy, oz + cz)
                    xs.append(px)
                    ys.append(py)
    return min(xs), max(xs), min(ys), max(ys)


def draw_stack(ax, cubes, current_cmap, current_norm):
    """Draw a stack of true cubes, furthest/lowest first."""
    L = CUBE_L
    cL = darken(current_cmap, SHADE_L)
    cR = darken(current_cmap, SHADE_R)
    e = np.linspace(0, L, N + 1)
    A, B = np.meshgrid(e, e, indexing="ij")
    ones = np.ones_like(A)
    origins = cube_origins(len(cubes))

    # cubes are ordered back-to-front, so the last one sits at the bottom
    order = sorted(range(len(cubes)), key=lambda k: origins[len(cubes) - 1 - k][2])

    for draw_i, k in enumerate(order):
        V = np.asarray(cubes[len(cubes) - 1 - k])
        ox, oy, oz = origins[k]
        zo = 10 * draw_i

        mesh = dict(shading="flat", rasterized=True, linewidth=0,
                    antialiased=False, norm=current_norm)

        # ---- top face  z = L, parametrised by (x, y) ----------------------
        Px, Py = iso(ox + A, oy + B, oz + L * ones)
        ax.pcolormesh(Px, Py, V[:, :, -1], cmap=current_cmap, zorder=zo + 3, **mesh)

        # ---- left wall  x = 0, parametrised by (y, z) ---------------------
        Px, Py = iso(ox + 0 * ones, oy + A, oz + B)
        ax.pcolormesh(Px, Py, V[0, :, :], cmap=cL, zorder=zo + 2, **mesh)

        # ---- right wall y = 0, parametrised by (x, z) ---------------------
        Px, Py = iso(ox + A, oy + 0 * ones, oz + B)
        ax.pcolormesh(Px, Py, V[:, 0, :], cmap=cR, zorder=zo + 2, **mesh)

        # ---- silhouette: outer hexagon + the three near edges -------------
        def P(x, y, z):
            return iso(ox + x, oy + y, oz + z)

        hexagon = [P(0, L, L), P(L, L, L), P(L, 0, L),
                   P(L, 0, 0), P(0, 0, 0), P(0, L, 0)]
        near = [P(0, L, L), P(0, 0, L), P(L, 0, L)]
        vert = [P(0, 0, L), P(0, 0, 0)]
        for pts, closed in ((hexagon, True), (near, False), (vert, False)):
            p = np.array(pts + ([pts[0]] if closed else []))
            ax.plot(p[:, 0], p[:, 1], color="0.25", lw=EDGE_LW,
                    solid_capstyle="round", solid_joinstyle="round", zorder=zo + 6)

    x0, x1, y0, y1 = stack_bbox(len(cubes))
    m = 0.02 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_axis_off()


# =========================================================================
# 4. LAYOUT GEOMETRY
# =========================================================================
PAD_L, PAD_R, PAD_T, PAD_B = 0.16, 0.16, 0.46, 0.16
DIVIDER_GAP = 0.24
GAP_X = 0.14

BIG_H = 4.10                       # drawing height of the clean stack
SMALL_H = 3.30                     # drawing height of the two noise stacks

_x0, _x1, _y0, _y1 = stack_bbox(3)
ASPECT = (_x1 - _x0) / (_y1 - _y0)

BIG_W = BIG_H * ASPECT
SMALL_W = SMALL_H * ASPECT

grid_w = 2 * SMALL_W + GAP_X
FW = PAD_L + BIG_W + 2 * DIVIDER_GAP + grid_w + PAD_R
FH = PAD_B + max(BIG_H, SMALL_H) + PAD_T

fig = plt.figure(figsize=(FW, FH), dpi=600)
fig.patch.set_facecolor("white")

fig.patches.append(FancyBboxPatch(
    (0.006, 0.009), 0.988, 0.982, transform=fig.transFigure,
    boxstyle="round,pad=0,rounding_size=0.012",
    linewidth=1.1, edgecolor="#33383d", facecolor="#fbfbfc", zorder=-5))

body_bot = PAD_B
body_top = FH - PAD_T


def stack_axes(x_in, y_in, w_in, h_in):
    return fig.add_axes([x_in / FW, y_in / FH, w_in / FW, h_in / FH])


# ---- left: clean signal -------------------------------------------------
ax_big = stack_axes(PAD_L, body_bot, BIG_W, BIG_H)
draw_stack(ax_big, clean_cubes, cmap, norm_sig)
fig.text((PAD_L + BIG_W / 2) / FW, (body_bot + BIG_H + 0.20) / FH,
         "(a)  Clean signal", ha="center", va="bottom",
         fontsize=10.5, fontweight="bold", color="#1b1f23")

# ---- divider ------------------------------------------------------------
xd = (PAD_L + BIG_W + DIVIDER_GAP) / FW
fig.add_artist(plt.Line2D([xd, xd], [body_bot / FH, (body_top - 0.02) / FH],
                          color="#c9ced4", lw=0.8, linestyle=(0, (3, 2.4))))

# ---- right: the single noise case ---------------------------------------
gx0 = PAD_L + BIG_W + 2 * DIVIDER_GAP
y_small = body_bot + (BIG_H - SMALL_H) * 0.5
col_titles = ["Noise only", "Signal + noise"]
col_data = [noise_only, sig_plus_noise]
col_norm = [norm_noise, norm_sig]

for c in range(2):
    x = gx0 + c * (SMALL_W + GAP_X)
    ax = stack_axes(x, y_small, SMALL_W, SMALL_H)
    draw_stack(ax, col_data[c], cmap, col_norm[c])
    fig.text((x + SMALL_W / 2) / FW, (y_small + SMALL_H + 0.05) / FH,
             col_titles[c], ha="center", va="bottom",
             fontsize=8.2, fontweight="bold", color="#4b5563")

fig.text((gx0 + grid_w / 2) / FW, (body_bot + BIG_H + 0.20) / FH,
         "(b)  Noise realisation", ha="center", va="bottom",
         fontsize=10.5, fontweight="bold", color="#1b1f23")

for ext in ("pdf", "png"):
    fig.savefig(f"cube_iso_grid.{ext}", dpi=600, facecolor="white")

print(f"Generated isometric cubes. Dimensions: {FW:.2f} x {FH:.2f} inches.")