import os

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR
from plot_utils import load_pickle, save_figure

MARKER_EVERY = 15
X_PADDING = 100
DETECTOR_STYLES = [
    {"color": "#d62728", "marker": "^", "linewidth": 1},
    {"color": "#ffbf0e", "marker": ".", "linewidth": 1},
    {"color": "#1f77b4", "marker": "x", "linewidth": 1},
]


def compute_modified_counts(raw_counts, times, shift, lane_count, background_flow):
    flow_per_second_per_lane = (background_flow / 3600.0) / lane_count
    modified_counts = (
        np.asarray(raw_counts) / lane_count
        - flow_per_second_per_lane * times
    )
    return times - shift, modified_counts


def plot_ramp(ax, ramp_name, config, times, history, lane_count):
    x_min, x_max = config["x_range"]

    for index, (label, detector) in enumerate(config["detectors"].items()):
        shifted_times, modified_counts = compute_modified_counts(
            history[ramp_name][label],
            times,
            detector["shift"],
            lane_count,
            config["q0"],
        )
        in_range = (shifted_times >= x_min) & (shifted_times <= x_max)
        ax.plot(
            shifted_times[in_range][::MARKER_EVERY],
            modified_counts[in_range][::MARKER_EVERY],
            label=label,
            **DETECTOR_STYLES[index],
        )

    ax.set_xlim(x_min - X_PADDING, x_max + X_PADDING)
    ax.margins(y=0.05)
    ax.set_xlabel("Simulation time (s)", fontsize=11)
    ax.set_ylabel("$N'(x,t) = N(x,t) - q_0 \\times t$", fontsize=11)
    ax.set_title(ramp_name, y=-0.2, fontsize=12)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.4))


def main():
    data = load_pickle(os.path.join(DATA_DIR, "figure_13_data.pkl"))
    times = np.asarray(data["times"])
    history = data["history"]
    ramps = data["ramps"]
    lane_count = data["num_lanes"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ramp_name, config in ramps.items():
        row, column = config["loc"]
        plot_ramp(
            axes[row, column],
            ramp_name,
            config,
            times,
            history,
            lane_count,
        )

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    save_figure(fig, "fig_13_rep.pdf")
    plt.show()


if __name__ == "__main__":
    main()
