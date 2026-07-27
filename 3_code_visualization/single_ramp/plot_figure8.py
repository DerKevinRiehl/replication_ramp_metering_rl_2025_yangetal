import matplotlib.pyplot as plt

from config import BASELINE_HISTORY_PATH, REPLACEMENT_HISTORY_PATH
from plot_utils import add_bottom_legend, load_history, save_figure, style_training_axis

def plot_episode_lengths():
    baseline_data = load_history(BASELINE_HISTORY_PATH)
    replacement_data = load_history(REPLACEMENT_HISTORY_PATH)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot baseline (orange)
    ax.plot(
        baseline_data["steps"],
        baseline_data["lengths"],
        color="darkorange",
        linewidth=1.2,
        alpha=0.8,
        label="Without lower bound constraint"
    )

    # Plot replacement (blue)
    ax.plot(
        replacement_data["steps"],
        replacement_data["lengths"],
        color="steelblue",
        linewidth=1.2,
        alpha=0.8,
        label="With lower bound constraint"
    )

    # Format exactly like Figure 8
    ax.set_ylim(-5, 250)
    # ax.set_xlim(-2000, 105000)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    # ax.set_xticks([0, 50000, 100000])
    # ax.set_xticklabels(["0", "50 k", "100 k"])

    style_training_axis(ax, "Episode length")
    add_bottom_legend(ax)
    save_figure(fig, "fig_8_rep.pdf")
    plt.show()

if __name__ == "__main__":
    plot_episode_lengths()