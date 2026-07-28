import pickle

import traci
import numpy as np
import os
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(MULTI_RAMP_DIR, "..", ".."))
sys.path.insert(0, MULTI_RAMP_DIR)

from config import SIMULATION_PATH, ramp_times
from controllers import RAMP_CONFIG
from utils import insertVehicles

# ------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------

SUMO_BINARY = "sumo"
SIM_END = 3600
NUM_SIMS = 10

n_demands = 6
demand_step = 50
base_demand = 1800

main_demands = [0]*12

all_ramp_demands = {
    i:
        {j: [base_demand+(i*demand_step)]*len(ramp_times) for j in range(1, 5)}
    for i in range(n_demands)
}
def run_simulation(ramp_demands, seed):
    history_queue = {f"Ramp {i}": [] for i in range(1, 5)}
    sumo_cmd = [
        SUMO_BINARY, "-c", SIMULATION_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(seed),
    ]

    np.random.seed(seed)
    traci.start(sumo_cmd)
    try:
        for _ in range(SIM_END):
            insertVehicles(main_demands, ramp_demands)
            for ramp_idx in range(1, 5):
                queue = sum(
                    traci.lanearea.getLastStepVehicleNumber(detector)
                    for detector in RAMP_CONFIG[ramp_idx]["queue_dets"]
                )
                history_queue[f"Ramp {ramp_idx}"].append(queue)
            traci.simulationStep()
    finally:
        traci.close()

    return history_queue


def main():
    train_history = {
        f"Demand {demand_idx}": {
            "queue": {f"Ramp {ramp_idx}": {} for ramp_idx in range(1, 5)}
        }
        for demand_idx in range(n_demands)
    }

    total_runs = n_demands * NUM_SIMS
    with tqdm(total=total_runs, desc="Running simulations") as progress:
        for demand_idx, ramp_demands in all_ramp_demands.items():
            for seed in range(NUM_SIMS):
                queues = run_simulation(ramp_demands, seed)
                for ramp, queue_history in queues.items():
                    train_history[f"Demand {demand_idx}"]["queue"][ramp][seed] = queue_history
                progress.update(1)

    output_dir = os.path.join(REPO_ROOT, "2_data_produced")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multi_ramp_queue_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(train_history, f)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()