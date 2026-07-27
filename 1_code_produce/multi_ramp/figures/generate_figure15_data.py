import re
import os
import sys
import pickle

import traci
import torch
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
CODE_DIR = os.path.dirname(MULTI_RAMP_DIR)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, MULTI_RAMP_DIR)

from utils import insertVehicles, format_state_vector, normalize_static, normalize_dynamic
from config import SIMULATION_PATH, NUM_RAMPS, STATE_DIM, ACTION_DIM, MODELS_DIR, SIM_STEPS_PER_CONTROL
from controllers import NoControlBaseline, PiAlineaController, HeroController, RLController, RAMP_CONFIG
from model import SharedActorCritic

# -----------------------------------------------------------------------
# Scenario configuration
# -----------------------------------------------------------------------

SUMO_BINARY = "sumo"
SIM_END = 5300
CONTROL_STEP = 15

SPATIAL_RESOLUTION = 50  # metres per bin
TOTAL_DISTANCE = 8000
N_BINS = TOTAL_DISTANCE // SPATIAL_RESOLUTION

TLS_IDS = {
    1: "junction_ramp_1",
    2: "junction_ramp_2",
    3: "junction_ramp_3",
    4: "junction_ramp_4",
}

ramp_demands = {
    1: [1400] * 12,
    2: [1200] * 12,
    3: [900]  * 12,
    4: [1600] * 12,
}

main_demands = [8000] * 12

warm_up   = 0.9
cool_down = 0.6

main_demands = [
    val * warm_up   if i <= 2 else
    val * cool_down if i >= 10 else
    val
    for i, val in enumerate(main_demands)
]

ramp_demands = {
    i: [
        val * warm_up   if j <= 1 else
        val * cool_down if j >= 10 else
        val
        for j, val in enumerate(demands)
    ]
    for i, demands in ramp_demands.items()
}

# Checkpoint to use for RL scenarios
RL_MODEL_PATH = os.path.join(MULTI_RAMP_DIR, MODELS_DIR, "model_baseline.pth")
RL_TRACKER_PATH = os.path.join(
    MULTI_RAMP_DIR, MODELS_DIR, "state_tracker_baseline.pkl"
)

BASELINE_SCENARIOS = {
    "no_control": NoControlBaseline,
    "pi_alinea":  PiAlineaController,
    "hero":       HeroController,
}

# -----------------------------------------------------------------------
# State reading helper (mirrors MultiRampMeterEnv.get_traffic_state)
# -----------------------------------------------------------------------

MAX_SPEED = 27.78

def _get_agg(det_ids, interval_steps):
    occ = float(np.mean([traci.inductionloop.getLastIntervalOccupancy(d) for d in det_ids]))
    raw_speeds = [traci.inductionloop.getLastIntervalMeanSpeed(d) for d in det_ids]
    speed = float(np.mean([s if s >= 0 else MAX_SPEED for s in raw_speeds]))
    veh = float(np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in det_ids])) / interval_steps
    return occ, speed, veh

def read_traffic_state(interval_steps):
    state = {}
    for i in range(1, NUM_RAMPS + 1):
        cfg = RAMP_CONFIG[i]
        dn_occ, dn_speed, dn_veh = _get_agg(cfg['dn_dets'], interval_steps)
        ramp_arr = float(np.sum([
            traci.inductionloop.getLastIntervalVehicleNumber(d)
            for d in cfg['ramp_arr_dets']
        ])) / interval_steps
        ramp_dep = float(np.sum([
            traci.inductionloop.getLastIntervalVehicleNumber(d)
            for d in cfg['ramp_dep_dets']
        ])) / interval_steps
        queue = float(sum(
            traci.lanearea.getJamLengthVehicle(d) for d in cfg['queue_dets']
        ))
        state[i] = {
            "dn_occ": dn_occ, "dn_speed": dn_speed, "dn_veh": dn_veh,
            "queue": queue, "ramp_arr": ramp_arr, "ramp_dep": ramp_dep,
        }
    return state

# -----------------------------------------------------------------------
# RL controller loader
# -----------------------------------------------------------------------

def load_rl_controller(model_path, tracker_path, use_dynamic_norm=False):
    agent = SharedActorCritic(STATE_DIM, action_dim=ACTION_DIM)
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()

    with open(tracker_path, "rb") as f:
        state_tracker = pickle.load(f)

    normalize_fnc = normalize_dynamic if use_dynamic_norm else normalize_static
    return RLController(agent=agent, state_tracker=state_tracker,
                        normalize_fnc=normalize_fnc, use_replacement=False)

# -----------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------

def run_scenario(scenario_name, controller, seed=42):
    np.random.seed(seed)
    speed_grid = np.full((N_BINS, SIM_END), 0.0)
    tts_total = 0
    pending = 0

    action_ratios = {i: 1.0 for i in range(1, 5)}
    green_steps   = {i: CONTROL_STEP for i in range(1, 5)}
    last_ratios   = {i: 1.0 for i in range(1, 5)}
    is_rl = isinstance(controller, RLController)

    traci.start([SUMO_BINARY, "-c", SIMULATION_PATH,
                 "--no-step-log", "true", "--no-warnings", "--seed", str(seed)])

    _DET_RE = re.compile(r"^det_dist_(\d+)_L\d+$")
    dist_detectors: dict[int, list[str]] = {}
    for det_id in traci.inductionloop.getIDList():
        m = _DET_RE.match(det_id)
        if m:
            dist = int(m.group(1))
            dist_detectors.setdefault(dist, []).append(det_id)

    dist_to_bin = {d: d // SPATIAL_RESOLUTION for d in dist_detectors}
    print(f"[{scenario_name}] Tracking {len(dist_detectors)} spatial detector groups "
          f"(max distance {max(dist_detectors, default=0)}m)")

    for step in tqdm(range(SIM_END), desc=scenario_name):
        if step > 0 and step % CONTROL_STEP == 0:
            if is_rl:
                state_dict = read_traffic_state(CONTROL_STEP)
                raw_state = format_state_vector(state_dict, last_ratios)
                action_ratios, *_ = controller.execute_control(raw_state, is_training=False)
            else:
                action_ratios, *_ = controller.execute_control()

            green_steps = {i: int(action_ratios[i] * CONTROL_STEP) for i in range(1, 5)}
            last_ratios = action_ratios

        step_in_control = step % CONTROL_STEP
        for ramp_idx in range(1, 5):
            state = "GG" if step_in_control < green_steps[ramp_idx] else "rr"
            traci.trafficlight.setRedYellowGreenState(TLS_IDS[ramp_idx], state)

        insertVehicles(main_demands, ramp_demands)

        for dist, det_ids in dist_detectors.items():
            speeds = [
                traci.inductionloop.getLastStepMeanSpeed(d)
                for d in det_ids
                if traci.inductionloop.getLastStepMeanSpeed(d) >= 0
            ]
            bin_idx = dist_to_bin[dist]
            if 0 <= bin_idx < N_BINS and speeds:
                speed_grid[bin_idx][step] = np.mean(speeds)

        traci.simulationStep()
        tts_total += traci.vehicle.getIDCount()
        pending += len(traci.simulation.getPendingVehicles())

    traci.close()
    print(f"[{scenario_name}] TTS = {tts_total / 3600:.2f} hours, Delay on Adjacent Networks = {pending / 3600:.2f}")
    return speed_grid


# -----------------------------------------------------------------------
# Run all scenarios
# -----------------------------------------------------------------------

def main():
    out_dir = os.path.join(REPO_ROOT, "2_data_produced")
    os.makedirs(out_dir, exist_ok=True)

    for name, controller_cls in BASELINE_SCENARIOS.items():
        print(f"\n{'='*40}\n  Running scenario: {name}\n{'='*40}")
        grid = run_scenario(name, controller_cls())
        output_path = os.path.join(out_dir, f"fig_15_speed_grid_{name}.npy")
        np.save(output_path, grid)
        print(f"Saved → {output_path}")

    print(f"\n{'='*40}\n  Running scenario: rl\n{'='*40}")
    rl_controller = load_rl_controller(RL_MODEL_PATH, RL_TRACKER_PATH)
    grid = run_scenario("rl", rl_controller)
    output_path = os.path.join(out_dir, "fig_15_speed_grid_rl.npy")
    np.save(output_path, grid)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
