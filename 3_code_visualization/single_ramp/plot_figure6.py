import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR

data_prod_dir = os.path.join("..", "..", "2_data_produced")
with open(os.path.join(data_prod_dir, "figure_6_data.pkl"), "rb") as f:
    modified_curves = pickle.load(f)

# =========================
# PLOT
# =========================

plt.figure(figsize=(8, 6))
skip_steps = 400
marker_interval = 15

styles = {
    "+920 m (upstream)":     {'color': '#d62728', 'marker': '^', 'markersize': 5, 'linewidth': 1},
    "+1225 m (downstream1)": {'color': '#ff7f0e', 'marker': '.', 'markersize': 5, 'linewidth': 1},
    "+1475 m (downstream2)": {'color': '#1f77b4', 'marker': 'x', 'markersize': 5, 'linewidth': 1}
}

for loc, (t, data) in modified_curves.items():
    mask = t > skip_steps
    plt.plot(t[mask][::marker_interval],
            data[mask][::marker_interval],
            label=loc.split(' ')[0],
            **styles[loc])

# Vertical dashed lines for capacity drop phases (approx 600s and 950s)
plt.axvline(x=600, color='lightgray', linestyle='--')
plt.axvline(x=950, color='lightgray', linestyle='--')

plt.xlabel("Simulation time (s)", fontsize=16)
plt.ylabel("$N'(x,t) = N(x,t) - q_0 \\times t$", fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig_6_rep.pdf"), bbox_inches="tight")
plt.show()