import os

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR
from plot_utils import load_pickle, save_figure

LINE_STYLES = {
    "(4) Inner Ring South": {
        "color": "#d62728", "marker": "x", "markersize": 5, "linewidth": 1.2
    },
    "(3) Shiyang": {
        "color": "#ff7f0e", "marker": "+", "markersize": 5, "linewidth": 1.2
    },
    "(1) Maquan": {
        "color": "#2ca02c", "marker": "o", "markersize": 5, "linewidth": 1.2
    },
    "(2) Shaungqi": {
        "color": "#1f77b4", "marker": "^", "markersize": 4, "linewidth": 1.2
    },
}
MARKER_EVERY = 300


def rolling_flow_vehph(raw_counts, window):
    raw_counts = np.asarray(raw_counts, dtype=float)
    cumulative = np.zeros(len(raw_counts) + 1)
    cumulative[1:] = np.cumsum(raw_counts)
    flow = np.zeros(len(raw_counts))
    half_window = window // 2

    for index in range(len(raw_counts)):
        start = max(0, index - half_window)
        end = min(len(raw_counts), index + half_window + 1)
        flow[index] = (
            (cumulative[end] - cumulative[start]) / (end - start) * 3600
        )
    return flow


def main():
    data = load_pickle(os.path.join(DATA_DIR, "figure_12_data.pkl"))
    times = data["times"]
    raw_counts = data["raw_counts"]
    flow_window = data["flow_window"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, counts in raw_counts.items():
        flow = rolling_flow_vehph(counts, flow_window)
        mask = times >= 600
        ax.plot(
            times[mask][::MARKER_EVERY],
            flow[mask][::MARKER_EVERY],
            label=name,
            **LINE_STYLES[name],
        )

    ax.set_xlabel("Simulation time (s)", fontsize=11)
    ax.set_ylabel("On-ramp downstream flow (veh/h)", fontsize=11)
    ax.legend(
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=10,
    )

    save_figure(fig, "fig_12_rep.pdf")
    plt.show()


if __name__ == "__main__":
    main()
