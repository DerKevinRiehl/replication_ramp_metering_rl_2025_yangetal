import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from action_replacement import calculate_penalty
from utils import format_state_vector
from config import (
    ALPHA, MAX_TTS, AVG_TTS, NUM_RAMPS, SIM_STEPS_PER_CONTROL,
)
from controllers import RAMP_CONFIG


def run_episode(env, controller, control_steps, sim_steps_per_control,
                is_training=False, verbose=False, progress_desc=None):
    trajectory = {
        "states": [], "actions": [], "log_probs": [],
        "values": [], "rewards": [], "dones": [],
    }
    history = {
        "green_times": [],
        "queues": {i: [] for i in range(1, NUM_RAMPS + 1)},
        "pending": 0,
        "speed": 0,
        "tts_total": 0,
        "lower_bounds": {i: [] for i in range(1, NUM_RAMPS + 1)},
    }
    num_replacements = {i: 0 for i in range(1, NUM_RAMPS + 1)}

    # Bootstrap initial state (all-green for one interval)
    initial_greens = {i: int(sim_steps_per_control) for i in range(1, NUM_RAMPS + 1)}
    env.apply_actions_and_get_tts(initial_greens, sim_steps_per_control)
    last_ratios = {i: 1.0 for i in range(1, NUM_RAMPS + 1)}

    raw_state_dict = env.get_traffic_state(sim_steps_per_control)
    raw_state = format_state_vector(raw_state_dict, last_ratios)

    steps = range(control_steps)
    if verbose:
        steps = tqdm(
            steps,
            desc=progress_desc or "episode",
            leave=False,
            unit="step",
        )

    for step in steps:
        # 1. Get actions from controller
        action_ratios, log_prob, value, raw_actions, state_tensor, extras = \
            controller.execute_control(raw_state, is_training)
        replaced_dict, lower_bounds = extras

        green_durations = {
            i: int(action_ratios[i] * sim_steps_per_control)
            for i in range(1, NUM_RAMPS + 1)
        }

        # 2. Step environment
        step_tts, pending, speed = env.apply_actions_and_get_tts(green_durations, sim_steps_per_control)
        reward = (MAX_TTS - step_tts) / AVG_TTS

        # Per-ramp replacement penalty
        use_replacement = getattr(controller, 'use_replacement', False)
        # if use_replacement:
        for i in range(1, NUM_RAMPS + 1):
            if replaced_dict.get(i, 0):
                ramp_offset = (i - 1) * 7
                curr_queue = raw_state[ramp_offset + 3]
                demand = raw_state[ramp_offset + 4]
                penalty = calculate_penalty(
                    curr_queue, demand, action_ratios[i],
                    RAMP_CONFIG[i]['ramp_capacity'] / 3600.0,  # convert veh/h → veh/s
                    SIM_STEPS_PER_CONTROL,
                    RAMP_CONFIG[i]['max_queue'], ALPHA,
                )
                reward -= penalty
                num_replacements[i] += 1

        # 3. Get next state
        next_state_dict = env.get_traffic_state(sim_steps_per_control)
        next_raw_state = format_state_vector(next_state_dict, action_ratios)

        # 4. Check termination -- spillback on any ramp
        spillback = False
        # if use_replacement:
        for i in range(1, NUM_RAMPS + 1):
            queue = next_state_dict[i]["queue"]
            if queue > ALPHA * RAMP_CONFIG[i]['max_queue']:
                spillback = True
                break
        done = spillback or (step == control_steps - 1)

        # 5. Record data
        history["tts_total"] += step_tts
        history["pending"] += pending
        history["speed"] += speed
        history["green_times"].append(action_ratios)
        for i in range(1, NUM_RAMPS + 1):
            history["queues"][i].append(next_state_dict[i]["queue"])
            history["lower_bounds"][i].append(lower_bounds.get(i, 0.0))

        if is_training:
            trajectory["states"].append(state_tensor)
            trajectory["actions"].append(raw_actions)
            trajectory["log_probs"].append(log_prob)
            trajectory["values"].append(value)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

        raw_state = next_raw_state

        if done and is_training:
            break

    num_steps = len(history["green_times"])
    replacement_pcts = {
        i: (num_replacements[i] / num_steps) * 100 for i in range(1, NUM_RAMPS + 1)
    }
    return trajectory, history, raw_state, replacement_pcts
