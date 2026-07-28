"""Calibration script for single-ramp RL training constants.

Runs several full episodes under NoControlBaseline (ramp fully open)
to measure worst-case per-step TTS and state statistics.  Using no-control
as the reference ensures that the RL reward is clearly positive when the
agent outperforms doing nothing, and negative when it makes things worse.

Outputs:
    MAX_TTS     -- maximum single-step TTS across all no-control seeds
    AVG_TTS     -- mean single-step TTS across all no-control seeds
                   (reward normalisation constant)
    STATE_MEANS / STATE_STDS -- paste into config.py

Usage:
    python calibrate.py [--seeds 3]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import (
    RAMP_DETS, SIMULATION_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)
from env import RampMeterEnv
from utils import format_state_vector
from controllers import NoControlBaseline


def run_calibration_episode(seed):
    sumo_cmd = [
        "sumo", "-c", SIMULATION_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(seed),
    ]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS, ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS, ramp_detector=RAMP_DETS
    )

    controller = NoControlBaseline()

    env.start()
    tts_per_step = []
    state_vectors = []

    print(f"  Seed {seed}: running no-control episode...")

    # Bootstrap: one fully-open interval so detectors have data
    env.apply_action_and_get_tts(SIM_STEPS_PER_CONTROL, 0)
    last_green = SIM_STEPS_PER_CONTROL

    for step in range(CONTROL_STEPS_PER_EPISODE):
        state_dict = env.get_traffic_state(SIM_STEPS_PER_CONTROL)
        raw_state_list = format_state_vector(state_dict, last_green)

        action_ratio, *_ = controller.execute_control(raw_state_list)

        green_duration = action_ratio * SIM_STEPS_PER_CONTROL
        red_duration = SIM_STEPS_PER_CONTROL - green_duration

        current_step_tts = env.apply_action_and_get_tts(green_duration, red_duration)

        tts_per_step.append(current_step_tts)
        state_vectors.append(raw_state_list)

        last_green = green_duration

        if (step + 1) % 60 == 0:
            print(f"    Step {step + 1}/{CONTROL_STEPS_PER_EPISODE} | TTS: {current_step_tts:.0f}")

    env.close()
    return tts_per_step, state_vectors


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds to average over (default: 3)")
    args = parser.parse_args()

    all_tts = []
    all_states = []

    print(f"Running {args.seeds} no-control calibration episode(s)...")
    for seed in range(args.seeds):
        tts_data, state_data = run_calibration_episode(seed)
        tts_arr = np.array(tts_data)
        all_tts.append(tts_arr)
        all_states.append(np.array(state_data))
        print(f"  Seed {seed}: max={tts_arr.max():.0f}  avg={tts_arr.mean():.0f}")

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
    print(f"Ratio MAX/AVG: {max_tts/avg_tts:.3f}")
    print(f"\nSTATE_MEANS ({len(means)}-dim):")
    print(f"  {means.round(4).tolist()}")
    print(f"\nSTATE_STDS ({len(stds)}-dim):")
    print(f"  {stds.round(4).tolist()}")
    print("\nPaste these into single_ramp/config.py:")
    print(f"MAX_TTS = {max_tts:.2f}")
    print(f"AVG_TTS = {avg_tts:.2f}")
    print(f"STATE_MEANS = {means.round(4).tolist()}")
    print(f"STATE_STDS  = {stds.round(4).tolist()}")
