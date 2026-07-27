import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS_DIR, RAMP_DETS, STATE_DIM, SIMULATION_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)

from env import RampMeterEnv
from controllers import RLController, NoControlBaseline, PiAlineaController
from runner import run_episode
from utils import normalize_static
from model import SharedActorCritic

EVAL_SEEDS = np.arange(10)

def run_evaluation(seed, model_path=None, tracker_path=None, use_replacement=False, no_control=False, alinea=False, return_history=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sumo_cmd = ["sumo", "-c", SIMULATION_PATH, "--no-step-log", "true", "--no-warnings",
                "--seed", str(seed)]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS, ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS, ramp_detector=RAMP_DETS
    )

    if no_control:
        controller = NoControlBaseline()
    elif alinea:
        controller = PiAlineaController()
    else:
        agent = SharedActorCritic(STATE_DIM)
        agent.load_state_dict(torch.load(model_path))
        agent.eval()

        with open(tracker_path, "rb") as f:
            state_tracker = pickle.load(f)

        controller = RLController(
            agent=agent, state_tracker=state_tracker,
            normalize_fnc=normalize_static, use_replacement=use_replacement
        )

    env.start()
    _, history, _, _ = run_episode(
        env=env,
        controller=controller,
        control_steps=CONTROL_STEPS_PER_EPISODE,
        sim_steps_per_control=SIM_STEPS_PER_CONTROL,
        is_training=False
    )
    env.close()

    tts_hours = history["tts_total"] / 3600.0
    max_queue = max(history["queues"])
    spillback_occurred = max_queue > (42 * 0.9)
    avg_action = np.mean(history["green_times"])
    min_action = min(history["green_times"])

    if return_history:
        return tts_hours, max_queue, spillback_occurred, avg_action, min_action, history
    return tts_hours, max_queue, spillback_occurred, avg_action, min_action


def run_evaluation_multi_seed(label, capture_first_seed_actions=False, **kwargs):
    tts_list, mq_list, sb_list = [], [], []
    first_seed_actions = None
    for i, seed in enumerate(EVAL_SEEDS):
        if capture_first_seed_actions and i == 0:
            tts, mq, sb, avg_action, min_action, history = run_evaluation(
                seed=seed, return_history=True, **kwargs
            )
            first_seed_actions = history["green_times"]
        else:
            tts, mq, sb, avg_action, min_action = run_evaluation(seed=seed, **kwargs)
        tts_list.append(tts)
        mq_list.append(mq)
        sb_list.append(sb)
        print(f"  seed {seed:2d} -> TTS: {tts:.2f} h | Max Queue: {mq:.1f} | Spillback: {sb} | Avg Action: {avg_action:.2f} | Min Action: {min_action:.2f}")

    tts_arr = np.array(tts_list)
    mq_arr = np.array(mq_list)
    sb_rate = np.mean(sb_list) * 100

    print(
        f"{label} | TTS: {tts_arr.mean():.2f} ± {tts_arr.std():.2f} h | "
        f"Max Queue: {mq_arr.mean():.1f} ± {mq_arr.std():.1f} | "
        f"Spillback rate: {sb_rate:.0f}%"
    )
    if capture_first_seed_actions:
        return tts_arr, mq_arr, sb_list, first_seed_actions
    return tts_arr, mq_arr, sb_list


if __name__ == "__main__":
    print(f"--- Single Ramp Evaluation ({len(EVAL_SEEDS)} seeds) ---\n")

    # print("[No-Control]")
    run_evaluation_multi_seed("No-Control", no_control=True)

    print("\n[ALINEA]")
    _, _, _, alinea_actions = run_evaluation_multi_seed(
        "ALINEA", alinea=True, capture_first_seed_actions=True,
    )

    base_model = os.path.join(MODELS_DIR, "model_baseline.pth")
    base_tracker = os.path.join(MODELS_DIR, "state_tracker_baseline.pkl")
    print("\n[RL-Baseline]")
    _, _, _, baseline_actions = run_evaluation_multi_seed(
        "RL-Baseline", model_path=base_model, tracker_path=base_tracker,
        capture_first_seed_actions=True,
    )

    rep_model = os.path.join(MODELS_DIR, "model_replacement.pth")
    rep_tracker = os.path.join(MODELS_DIR, "state_tracker_replacement.pkl")
    print("\n[RL+Replacement]")
    _, _, _, replacement_actions = run_evaluation_multi_seed(
        "RL+Replacement", model_path=rep_model, tracker_path=rep_tracker,
        use_replacement=True, capture_first_seed_actions=True,
    )

    data_prod_dir = os.path.join("..", "..", "2_data_produced")
    os.makedirs(data_prod_dir, exist_ok=True)
    action_data = {
        "seed": int(EVAL_SEEDS[0]),
        "baseline_actions": baseline_actions,
        "replacement_actions": replacement_actions,
        "alinea_actions": alinea_actions,
    }
    data_path = os.path.join(data_prod_dir, "action_comparison.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(action_data, f)
    print(f"\nSaved action comparison data to {data_path}")
