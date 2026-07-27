import matplotlib.pyplot as plt

from config import REPLACEMENT_HISTORY_PATH
from plot_utils import load_history, save_figure, style_training_axis

def plot_replacement_percentage():
    data = load_history(REPLACEMENT_HISTORY_PATH)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the replacement percentage
    ax.plot(
        data["steps"],
        data["replacement_pct"],
        color="firebrick",
        linewidth=1.2,
        alpha=0.9
    )

    # Formatting matching Figure 10
    ax.set_ylim(-1, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])

    max_step = max(data["steps"])
    ax.set_xlim(-5000, max_step + 5000)

    style_training_axis(ax, "Action replacement percentage (%)")
    save_figure(fig, "fig_10_rep.pdf")
    plt.show()

if __name__ == "__main__":
    plot_replacement_percentage()