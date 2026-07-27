import numpy as np
import matplotlib.pyplot as plt

from config import REPLACEMENT_HISTORY_PATH
from plot_utils import load_pickle, save_figure, style_training_axis


def plot_replacement_percentage():
    data = load_pickle(REPLACEMENT_HISTORY_PATH)

    # replacement_pct is a list of per-ramp dicts; average across the 4 ramps
    avg_pcts = [np.mean(list(d.values())) for d in data["replacement_pct"]]

    steps    = data["steps"]
    max_step = steps[-1]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        steps,
        avg_pcts,
        color="firebrick",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.set_xlim(0, max_step + 2_000)
    tick_step = 50_000
    ticks = list(range(0, max_step + tick_step, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000} k" if t > 0 else "0" for t in ticks])

    ax.set_ylim(-1, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])

    style_training_axis(ax, "Action replacement percentage (%)")
    fig_path = save_figure(fig, "fig_18_rep.pdf")
    print(f"Saved → {fig_path}")
    plt.show()


if __name__ == "__main__":
    plot_replacement_percentage()
