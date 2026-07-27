import matplotlib.pyplot as plt

from config import BASELINE_HISTORY_PATH, REPLACEMENT_HISTORY_PATH
from plot_utils import add_bottom_legend, load_history, save_figure, style_training_axis

def plot_network_tts():
    baseline_data = load_history(BASELINE_HISTORY_PATH)
    replacement_data = load_history(REPLACEMENT_HISTORY_PATH)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot baseline (orange)
    ax.plot(
        baseline_data["steps"],
        baseline_data["tts"],
        color="darkorange",
        linewidth=1.2,
        alpha=0.8,
        label="Without lower bound constraint"
    )

    # Plot replacement (blue)
    ax.plot(
        replacement_data["steps"],
        replacement_data["tts"],
        color="steelblue",
        linewidth=1.2,
        alpha=0.8,
        label="With lower bound constraint"
    )

    # Formatting
    ax.set_xlim(-2000, 105000)
    ax.set_xticks([0, 50000, 100000])
    ax.set_xticklabels(["0", "50 k", "100 k"])

    # Format y-axis to scientific notation like the paper (e.g., 1e6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(16)

    style_training_axis(ax, "Total time spent (s)")
    add_bottom_legend(ax)
    save_figure(fig, "fig_9_rep.pdf")
    plt.show()

if __name__ == "__main__":
    plot_network_tts()
