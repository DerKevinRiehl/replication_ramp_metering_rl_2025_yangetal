import matplotlib.pyplot as plt

from config import BASELINE_HISTORY_PATH, REPLACEMENT_HISTORY_PATH
from plot_utils import add_bottom_legend, load_pickle, save_figure, style_training_axis

XLIM = 150_000


def plot_network_tts():
    baseline_data = load_pickle(BASELINE_HISTORY_PATH)
    replacement_data = load_pickle(REPLACEMENT_HISTORY_PATH)

    base_steps = [s for s in baseline_data["steps"]      if s <= XLIM]
    base_tts   = baseline_data["tts"][:len(base_steps)]

    repl_steps = [s for s in replacement_data["steps"]   if s <= XLIM]
    repl_tts   = replacement_data["tts"][:len(repl_steps)]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        base_steps,
        base_tts,
        color="darkorange",
        linewidth=1.0,
        alpha=0.85,
        label="Without lower bound constraint",
    )

    ax.plot(
        repl_steps,
        repl_tts,
        color="steelblue",
        linewidth=1.0,
        alpha=0.85,
        label="With lower bound constraint",
    )

    ax.set_xlim(0, XLIM)
    ax.set_xticks([0, 50_000, 100_000, 150_000])
    ax.set_xticklabels(["0", "50 k", "100 k", "150 k"])

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(12)

    style_training_axis(ax, "Total time spent (s)")
    add_bottom_legend(ax)
    fig_path = save_figure(fig, "fig_17_rep.pdf")
    print(f"Saved → {fig_path}")
    plt.show()


if __name__ == "__main__":
    plot_network_tts()
