import matplotlib.pyplot as plt

from config import BASELINE_HISTORY_PATH
from plot_utils import load_pickle, save_figure

REWARD_COLOR    = "steelblue"
TTS_COLOR = "firebrick"


def plot_tts_and_reward():
    data = load_pickle(BASELINE_HISTORY_PATH)

    steps   = data["steps"]
    tts     = data["tts"]
    scores  = data["scores"]
    max_step = steps[-1]

    fig, ax_tts = plt.subplots(figsize=(8, 6))
    ax_reward   = ax_tts.twinx()

    line_tts, = ax_tts.plot(
        steps, tts,
        color=TTS_COLOR,
        linewidth=1.0,
        alpha=0.85,
        label="Total time spent",
    )

    line_reward, = ax_reward.plot(
        steps, scores,
        color=REWARD_COLOR,
        linewidth=1.0,
        alpha=0.85,
        label="Episode reward",
    )

    # X-axis
    ax_tts.set_xlim(0, max_step + 2_000)
    tick_step = 50_000
    ticks = list(range(0, max_step + tick_step, tick_step))
    ax_tts.set_xticks(ticks)
    ax_tts.set_xticklabels([f"{t // 1000} k" if t > 0 else "0" for t in ticks])
    ax_tts.set_xlabel("Simulation step", fontsize=16)
    ax_tts.tick_params(axis='both', labelsize=14)

    # Left y-axis: TTS
    ax_tts.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_tts.yaxis.get_offset_text().set_fontsize(16)
    ax_tts.set_ylabel("Total time spent (s)", fontsize=16, color=TTS_COLOR)
    ax_tts.tick_params(axis="y", labelcolor=TTS_COLOR)

    # Right y-axis: episode reward
    ax_reward.set_ylabel("Episode reward", fontsize=16, color=REWARD_COLOR)
    ax_reward.tick_params(axis="y", labelcolor=REWARD_COLOR)

    ax_tts.grid(True, linestyle="--", alpha=0.4)

    # Combined legend centred below the plot
    lines  = [line_tts, line_reward]
    labels = [l.get_label() for l in lines]
    ax_tts.legend(
        lines, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        ncol=2,
        fontsize=16,
    )

    fig_path = save_figure(fig, "fig_14_rep.pdf")
    print(f"Saved → {fig_path}")
    plt.show()


if __name__ == "__main__":
    plot_tts_and_reward()
