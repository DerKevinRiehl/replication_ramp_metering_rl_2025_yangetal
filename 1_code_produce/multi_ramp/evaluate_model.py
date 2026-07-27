import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS_DIR, STATE_DIM, ACTION_DIM, SIMULATION_PATH,
    CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL, NUM_RAMPS, ALPHA,
)
from env import MultiRampMeterEnv
from controllers import RLController, NoControlBaseline, PiAlineaController, HeroController, RAMP_CONFIG
from runner import run_episode
from utils import normalize_static, format_state_vector
from model import SharedActorCritic

EVAL_SEEDS = np.arange(10)

RL_MODEL_PATH = os.path.join(MODELS_DIR, "model_baseline.pth")
RL_TRACKER_PATH = os.path.join(MODELS_DIR, "state_tracker_baseline.pkl")

RL_2_MODEL_PATH = os.path.join(MODELS_DIR, "model_replacement.pth")
RL_2_TRACKER_PATH = os.path.join(MODELS_DIR, "state_tracker_replacement.pkl")


def run_evaluation(seed, model_path=None, tracker_path=None,
                    use_replacement=False, no_control=False,
                    alinea=False, hero=False, verbose=True,
                    return_history=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sumo_cmd = [
        "sumo", "-c", SIMULATION_PATH,
        "--no-step-log", "true", "--no-warnings",
        "--seed", str(seed),
    ]

    env = MultiRampMeterEnv(sumo_cmd=sumo_cmd)

    if no_control:
        controller = NoControlBaseline()
    elif alinea:
        controller = PiAlineaController()
    elif hero:
        controller = HeroController()
    else:
        agent = SharedActorCritic(STATE_DIM, action_dim=ACTION_DIM)
        agent.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.eval()

        with open(tracker_path, "rb") as f:
            state_tracker = pickle.load(f)

        controller = RLController(
            agent=agent,
            state_tracker=state_tracker,
            normalize_fnc=normalize_static,
            use_replacement=use_replacement,
        )

    if no_control:
        progress_desc = f"No-Control seed {seed}"
    elif alinea:
        progress_desc = f"ALINEA seed {seed}"
    elif hero:
        progress_desc = f"HERO seed {seed}"
    else:
        progress_desc = f"RL seed {seed}"

    env.start()
    _, history, _, _ = run_episode(
        env=env,
        controller=controller,
        control_steps=CONTROL_STEPS_PER_EPISODE,
        sim_steps_per_control=SIM_STEPS_PER_CONTROL,
        is_training=False,
        verbose=verbose,
        progress_desc=progress_desc,
    )
    env.close()

    tts_hours = history["tts_total"] / 3600.0
    pending_hours = history["pending"] / 3600.0
    avg_speed = history["speed"] / CONTROL_STEPS_PER_EPISODE

    # Per-ramp max queue and spillback check
    max_queues = {
        i: max(history["queues"][i]) if history["queues"][i] else 0.0
        for i in range(1, NUM_RAMPS + 1)
    }
    spillback_occurred = any(
        max_queues[i] > ALPHA * RAMP_CONFIG[i]["max_queue"]
        for i in range(1, NUM_RAMPS + 1)
    )

    # Average green-time ratio across ramps and steps
    all_ratios = [
        ratio
        for step_ratios in history["green_times"]
        for ratio in step_ratios.values()
    ]
    avg_ratio = float(np.mean(all_ratios)) if all_ratios else float("nan")

    if return_history:
        return tts_hours, pending_hours, avg_speed, max_queues, spillback_occurred, avg_ratio, history
    return tts_hours, pending_hours, avg_speed, max_queues, spillback_occurred, avg_ratio


def _extract_per_ramp_actions(green_times):
    """Convert list-of-dicts green_times into a dict of per-ramp action arrays."""
    per_ramp = {i: [] for i in range(1, NUM_RAMPS + 1)}
    for step_ratios in green_times:
        for i in range(1, NUM_RAMPS + 1):
            per_ramp[i].append(step_ratios[i])
    return {i: np.array(v) for i, v in per_ramp.items()}


def run_evaluation_multi_seed(label, capture_first_seed_actions=False, **kwargs):
    tts_list, pending_list, speed_list, sb_list = [], [], [], []
    max_queue_lists = {i: [] for i in range(1, NUM_RAMPS + 1)}
    first_seed_actions = None

    for idx, seed in enumerate(EVAL_SEEDS):
        if capture_first_seed_actions and idx == 0:
            tts, pending, speed, max_queues, sb, avg_ratio, history = run_evaluation(
                seed=seed, return_history=True, **kwargs
            )
            first_seed_actions = _extract_per_ramp_actions(history["green_times"])
        else:
            tts, pending, speed, max_queues, sb, avg_ratio = run_evaluation(seed=seed, **kwargs)
        tts_list.append(tts)
        pending_list.append(pending)
        speed_list.append(speed)
        sb_list.append(sb)
        for i in range(1, NUM_RAMPS + 1):
            max_queue_lists[i].append(max_queues[i])

        mq_str = "  ".join(f"R{i}:{max_queues[i]:.0f}" for i in range(1, NUM_RAMPS + 1))
        print(
            f"  seed {seed:2d} -> TTS: {tts:.2f} h | {mq_str} | "
            f"Spillback: {sb} | Avg Ratio: {avg_ratio:.3f} | Pending: {pending:.1f} h | Speed: {speed:.1f} m/s"
        )

    tts_arr = np.array(tts_list)
    pending_arr = np.array(pending_list)
    sb_rate = np.mean(sb_list) * 100
    speed_arr = np.array(speed_list)

    mq_summary = "  ".join(
        f"R{i}:{np.mean(max_queue_lists[i]):.1f}±{np.std(max_queue_lists[i]):.1f}"
        for i in range(1, NUM_RAMPS + 1)
    )
    print(
        f"{label} | TTS: {tts_arr.mean():.2f} ± {tts_arr.std():.2f} h | "
        f"Max Queue — {mq_summary} | Spillback rate: {sb_rate:.0f}% | Pending: {pending_arr.mean():.1f} ± {pending_arr.std():.1f} h | Speed: {speed_arr.mean():.1f} ± {speed_arr.std():.1f} m/s"
    )
    if capture_first_seed_actions:
        return tts_arr, pending_arr, speed_arr, max_queue_lists, sb_list, first_seed_actions
    return tts_arr, pending_arr, speed_arr, max_queue_lists, sb_list


if __name__ == "__main__":
    print(f"--- Multi-Ramp Evaluation ({len(EVAL_SEEDS)} seeds) ---\n")

    # print("[No-Control]")
    # run_evaluation_multi_seed("No-Control", no_control=True)

    print("\n[ALINEA]")
    _, _, _, _, _, alinea_actions = run_evaluation_multi_seed(
        "ALINEA", alinea=True, capture_first_seed_actions=True
    )

    print("\n[HERO]")
    _, _, _, _, _, hero_actions = run_evaluation_multi_seed(
        "HERO", hero=True, capture_first_seed_actions=True
    )

    print("\n[RL]")
    _, _, _, _, _, baseline_actions = run_evaluation_multi_seed(
        "RL",
        model_path=RL_MODEL_PATH,
        tracker_path=RL_TRACKER_PATH,
        capture_first_seed_actions=True,
    )

    print("\n[RL+Replacement]")
    _, _, _, _, _, replacement_actions = run_evaluation_multi_seed(
        "RL+Replacement",
        model_path=RL_2_MODEL_PATH,
        tracker_path=RL_2_TRACKER_PATH,
        use_replacement=True,
        capture_first_seed_actions=True,
    )

    data_prod_dir = os.path.join("..", "..", "2_data_produced")
    os.makedirs(data_prod_dir, exist_ok=True)
    action_data = {
        "seed": int(EVAL_SEEDS[0]),
        "alinea_actions": alinea_actions,
        "hero_actions": hero_actions,
        "baseline_actions": baseline_actions,
        "replacement_actions": replacement_actions,
    }
    data_path = os.path.join(data_prod_dir, "multi_ramp_action_comparison.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(action_data, f)
    print(f"\nSaved action comparison data to {data_path}")
