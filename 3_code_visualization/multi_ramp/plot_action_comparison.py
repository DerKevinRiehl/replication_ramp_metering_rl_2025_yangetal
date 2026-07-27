import os

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, PLOTS_DIR
from plot_utils import load_pickle

NUM_RAMPS = 4

CONTROLLERS = [
    ("alinea_actions",      "ALINEA",          "#4C72B0"),
    ("baseline_actions",    "RL-Baseline",      "#DD8452"),
    ("replacement_actions", "RL+Replacement",   "#55A868"),
]

BINS = np.linspace(0.0, 1.0, 31)  # 30 equal-width bins over [0, 1]


def plot_action_comparison(data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes_flat = axes.flatten()

    for ramp_idx, ax in enumerate(axes_flat):
        ramp_id = ramp_idx + 1
        for key, label, color in CONTROLLERS:
            actions = data[key][ramp_id]

            # Calculate weights so the sum of all bins equals 1.0
            weights = np.ones_like(actions) / len(actions)

            ax.hist(
                actions,
                bins=BINS,
                label=label,
                color=color,
                alpha=0.55,
                edgecolor="none",
                weights=weights,  # Replaces density=True
            )

        ax.set_title(f"Ramp {ramp_id}", fontsize=16)
        ax.set_xlabel("Action ratio", fontsize=16)
        ax.set_ylabel("Proportion of Actions", fontsize=16)
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, axis="y", alpha=0.25)
        if ramp_idx == 0:
            ax.legend(fontsize=14, framealpha=0.9)

    # fig.suptitle(f"Action distribution — seed {seed}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Saved action comparison plot to {save_path}")


if __name__ == "__main__":
    data_path = os.path.join(DATA_DIR, "multi_ramp_action_comparison.pkl")
    data = load_pickle(data_path)

    fig_path = os.path.join(PLOTS_DIR, "multi_ramp_action_comparison.pdf")
    plot_action_comparison(data, fig_path)
