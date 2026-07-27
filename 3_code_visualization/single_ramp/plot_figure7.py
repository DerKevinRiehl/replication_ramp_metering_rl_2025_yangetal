import os
import traci
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import PLOTS_DIR

MAINLINE_FLOWS = [7400, 7800, 8200, 8600, 9000, 9400]
RAMP_FLOWS = [600, 800, 1000, 1200, 1400, 1600]

# MAINLINE_FLOWS = [5800+(i*100) for i in range(6)]
# RAMP_FLOWS = [600+(i*100) for i in range(6)]

def generate_heatmap():
    data_prod_dir = os.path.join("..", "..", "2_data_produced")
    results = np.load(os.path.join(data_prod_dir, "figure_7_matrix.npy"))

    df = pd.DataFrame(results, index=RAMP_FLOWS, columns=MAINLINE_FLOWS)
    df = df.iloc[::-1]

    plt.figure(figsize=(8,7))

    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu_r", cbar_kws={'label': 'Occupancy (%)'},
                        vmin=12, vmax=14.5, linewidths=.5, annot_kws={"size": 14})

    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Mainline flow (veh/h)", fontsize=16)
    ax.set_ylabel("On-ramp flow (veh/h)", fontsize=16)
    ax.xaxis.set_ticks_position('bottom')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Occupancy (%)", fontsize=16)

    plt.tight_layout()
    fig_path = os.path.join(PLOTS_DIR, "fig_7_rep.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    generate_heatmap()