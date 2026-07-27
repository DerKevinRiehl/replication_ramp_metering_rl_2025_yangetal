import matplotlib.pyplot as plt

from config import BASELINE_HISTORY_PATH, REPLACEMENT_HISTORY_PATH
from plot_utils import add_bottom_legend, load_pickle, save_figure, style_training_axis

XLIM = 150_000


def plot_episode_lengths():
    baseline_data = load_pickle(BASELINE_HISTORY_PATH)
    replacement_data = load_pickle(REPLACEMENT_HISTORY_PATH)

    # Clip both series to the first XLIM simulation steps
    base_steps   = [s for s in baseline_data["steps"]    if s <= XLIM]
    base_lengths = baseline_data["lengths"][:len(base_steps)]

    repl_steps   = [s for s in replacement_data["steps"]   if s <= XLIM]
    repl_lengths = replacement_data["lengths"][:len(repl_steps)]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        base_steps,
        base_lengths,
        color="darkorange",
        linewidth=1.0,
        alpha=0.85,
        label="Without lower bound constraint",
    )

    ax.plot(
        repl_steps,
        repl_lengths,
        color="steelblue",
        linewidth=1.0,
        alpha=0.85,
        label="With lower bound constraint",
    )

    ax.set_xlim(0, XLIM)
    ax.set_xticks([0, 50_000, 100_000, 150_000])
    ax.set_xticklabels(["0", "50 k", "100 k", "150 k"])

    ax.set_ylim(-5, 330)
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])

    style_training_axis(ax, "Episode length")
    add_bottom_legend(ax)
    fig_path = save_figure(fig, "fig_16_rep.pdf")
    print(f"Saved → {fig_path}")
    plt.show()


if __name__ == "__main__":
    plot_episode_lengths()
