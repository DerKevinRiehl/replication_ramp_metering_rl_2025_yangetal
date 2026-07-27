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
from controllers import PiAlineaController
from runner import run_episode

EVAL_SEEDS = np.arange(10)

def run_evaluation(seed, target_occ=13.0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sumo_cmd = ["sumo", "-c", SIMULATION_PATH, "--no-step-log", "true", "--no-warnings",
                "--seed", str(seed)]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS, ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS, ramp_detector=RAMP_DETS
    )

    controller = PiAlineaController(target_occ=target_occ)

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

    return tts_hours, max_queue, spillback_occurred, avg_action, min_action


def run_evaluation_multi_seed(label, **kwargs):
    tts_list, mq_list, sb_list = [], [], []
    for seed in EVAL_SEEDS:
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
    return tts_arr, mq_arr, sb_list


if __name__ == "__main__":
    print(f"--- Single Ramp Evaluation ({len(EVAL_SEEDS)} seeds) ---\n")

    target_occs = [11.0, 12.0, 12.5, 13.0, 14.0, 15.0, 16.0]
    for target_occ in target_occs:
        print(f"\n[ALINEA (target_occ={target_occ})]")
        run_evaluation_multi_seed(f"ALINEA (target_occ={target_occ})", target_occ=target_occ)
