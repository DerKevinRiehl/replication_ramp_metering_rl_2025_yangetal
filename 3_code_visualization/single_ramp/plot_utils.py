import os
import pickle

from config import PLOTS_DIR

LABEL_SIZE = 16
TICK_SIZE = 14


def load_history(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def style_training_axis(ax, ylabel):
    ax.set_xlabel("Simulation step", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)


def add_bottom_legend(ax):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        ncol=1,
        fontsize=LABEL_SIZE,
    )


def save_figure(fig, filename):
    fig.tight_layout()
    output_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(output_path, bbox_inches="tight")
    return output_path
