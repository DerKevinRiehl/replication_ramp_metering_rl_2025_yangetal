import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from config import DATA_DIR
from plot_utils import save_figure

SCENARIOS = {
    "no_control": "No Control",
    "pi_alinea":  "PI-ALINEA",
    "hero":       "HERO",
    "rl":         "RL Agent",
}

bounds = [0, 4, 8, 12, 16, 20, 24, 28, 32]

colors = [
    "#2F4AA0",  # 0-4:   dark blue
    "#3562AF",  # 4-8:   bright blue
    "#2AC8F6",  # 8-12:  cyan
    "#83CCA2",  # 12-16: cyan-green
    "#BDD637",  # 16-20: yellow-green
    "#FFD400",  # 20-24: yellow-orange
    "#F45F00",  # 24-28: red-orange
    "#CB0017",  # 28-32: dark red
]

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

ramp_positions = [933, 3972, 4856, 6442]
peak_times = [600, 3600]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes_flat = axes.flatten()

for ax, (key, label) in zip(axes_flat, SCENARIOS.items()):
    speed_grid = np.load(os.path.join(DATA_DIR, f"fig_15_speed_grid_{key}.npy"))

    im = ax.imshow(speed_grid,
                cmap=cmap,
                norm=norm,
                origin='lower',
                aspect='auto',
                extent=[0, 5300, 0, 8000])

    for pos in ramp_positions:
        ax.axhline(y=pos, color='black', linestyle='--', alpha=0.6)
    for t in peak_times:
        ax.axvline(x=t, color='black', linestyle='--', alpha=0.6)

    ax.set_title(label, fontsize=14)
    ax.set_xlabel('Simulation time (s)', fontsize=16)
    ax.set_ylabel('Position (m)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 7000)

    cbar = fig.colorbar(im, ax=ax, ticks=bounds)
    cbar.set_label('Average Speed (m/s)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

plt.tight_layout()

save_figure(fig, "fig_15_rep.pdf")

plt.show()
