import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

DEMAND_INDEX = 1

data_prod_dir = os.path.join("..", "..", "2_data_produced")
plots_dir = os.path.join("..", "..", "3_data_visualization")

def plot_action_comparison(baseline_actions, replacement_actions, alinea_actions, seed, save_path):
    control_steps = np.arange(len(baseline_actions))

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(control_steps, baseline_actions, label="RL-Baseline", linewidth=1.5, linestyle="-")
    ax.plot(control_steps, replacement_actions, label="RL+Replacement", linewidth=1.5, linestyle="-", alpha=0.9)
    ax.plot(control_steps, alinea_actions, label="ALINEA", linewidth=1.5, linestyle="-", alpha=0.9)

    ax.set_xlabel("Control step", fontsize=16)
    ax.set_ylabel("Action value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    ax.legend(fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    data_path = os.path.join(data_prod_dir, f"action_comparison_demand{DEMAND_INDEX}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    fig_path = os.path.join(plots_dir, f"action_comparison_demand{DEMAND_INDEX}.pdf")
    plot_action_comparison(
        data["baseline_actions"],
        data["replacement_actions"],
        data["alinea_actions"],
        data["seed"],
        fig_path,
    )
    print(f"Saved action comparison plot to {fig_path}")
