"""Calibration script for multi-ramp RL training constants.

Runs several full episodes under NoControlBaseline (all ramps fully open)
to measure worst-case per-step TTS and state statistics.  Using no-control
as the reference ensures that RL rewards are clearly positive when the agent
outperforms doing nothing and negative when it makes things worse.

Outputs:
    MAX_TTS     -- maximum single-step TTS across all no-control seeds
    AVG_TTS     -- mean single-step TTS across all no-control seeds
                   (reward normalisation constant)
    STATE_MEANS / STATE_STDS -- paste into config.py

Usage:
    python calibration/calibrate.py --seeds 3
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    SIMULATION_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL, NUM_RAMPS,
)
from env import MultiRampMeterEnv
from controllers import NoControlBaseline
from utils import format_state_vector


def run_calibration_episode(seed):
    sumo_cmd = [
        "sumo", "-c", SIMULATION_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(seed),
    ]

    env = MultiRampMeterEnv(sumo_cmd=sumo_cmd)
    controller = NoControlBaseline()

    env.start()
    tts_per_step = []
    state_vectors = []

    print(f"  Seed {seed}: running no-control episode...")

    # Bootstrap: one fully-open interval so detectors have data
    initial_greens = {i: int(SIM_STEPS_PER_CONTROL) for i in range(1, NUM_RAMPS + 1)}
    env.apply_actions_and_get_tts(initial_greens, SIM_STEPS_PER_CONTROL)
    last_ratios = {i: 1.0 for i in range(1, NUM_RAMPS + 1)}

    for step in range(CONTROL_STEPS_PER_EPISODE):
        # 1. Observe state
        state_dict = env.get_traffic_state(SIM_STEPS_PER_CONTROL)
        raw_state = format_state_vector(state_dict, last_ratios)

        # 2. No-control action (all ramps fully open)
        action_ratios, *_ = controller.execute_control()

        # 3. Apply actions and collect TTS
        green_durations = {
            i: int(action_ratios[i] * SIM_STEPS_PER_CONTROL)
            for i in range(1, NUM_RAMPS + 1)
        }
        step_tts, _, _ = env.apply_actions_and_get_tts(
            green_durations, SIM_STEPS_PER_CONTROL
        )

        tts_per_step.append(step_tts)
        state_vectors.append(raw_state)
        last_ratios = action_ratios

        if (step + 1) % 60 == 0:
            print(f"    Step {step + 1}/{CONTROL_STEPS_PER_EPISODE} | TTS: {step_tts:.0f}")

    env.close()
    return np.array(tts_per_step), np.array(state_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds to average over (default: 3)")
    args = parser.parse_args()

    all_tts = []
    all_states = []

    print(f"Running {args.seeds} no-control calibration episode(s)...")
    for seed in range(args.seeds):
        tts_data, state_data = run_calibration_episode(seed)
        all_tts.append(tts_data)
        all_states.append(state_data)
        print(f"  Seed {seed}: max={tts_data.max():.0f}  avg={tts_data.mean():.0f}")

    all_tts = np.concatenate(all_tts)
    all_states = np.concatenate(all_states, axis=0)

    max_tts = float(np.max(all_tts))
    avg_tts = float(np.mean(all_tts))

    means = np.mean(all_states, axis=0)
    stds = np.std(all_states, axis=0)
    stds[stds == 0] = 1e-8

    print("\n--- Calibration Results ---")
    print(f"MAX_TTS (maximum,  no-control): {max_tts:.2f}")
    print(f"AVG_TTS (mean,     no-control): {avg_tts:.2f}")
    print(f"Ratio MAX/AVG: {max_tts/avg_tts:.3f}  (single-ramp reference: 1.643)")
    print(f"\nSTATE_MEANS ({len(means)}-dim):")
    print(f"  {means.round(4).tolist()}")
    print(f"\nSTATE_STDS ({len(stds)}-dim):")
    print(f"  {stds.round(4).tolist()}")
    print("\nPaste these into multi_ramp/config.py:")
    print(f"  MAX_TTS = {max_tts:.2f}")
    print(f"  AVG_TTS = {avg_tts:.2f}")
    print(f"  STATE_MEANS = {means.round(4).tolist()}")
    print(f"  STATE_STDS  = {stds.round(4).tolist()}")
