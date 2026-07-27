import argparse
import json
import os
import sys

import numpy as np
import traci

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(MULTI_RAMP_DIR, "..", ".."))
sys.path.insert(0, MULTI_RAMP_DIR)

from config import SIMULATION_PATH, TLS_IDS, main_demands, ramp_demands
from controllers import RAMP_CONFIG
from utils import insertVehicles

SUMO_CONFIG_PATH = os.path.abspath(os.path.join(MULTI_RAMP_DIR, SIMULATION_PATH))


def run_simulation(seed, simulation_steps):
    np.random.seed(seed)
    maximum_queues = {ramp_idx: 0.0 for ramp_idx in RAMP_CONFIG}
    sumo_cmd = [
        "sumo", "-c", SUMO_CONFIG_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(seed),
    ]

    traci.start(sumo_cmd)
    try:
        for _ in range(simulation_steps):
            for tls_id in TLS_IDS.values():
                traci.trafficlight.setRedYellowGreenState(tls_id, "rr")

            insertVehicles(main_demands, ramp_demands)
            traci.simulationStep()

            for ramp_idx, config in RAMP_CONFIG.items():
                queue = sum(
                    traci.lanearea.getJamLengthVehicle(detector)
                    for detector in config["queue_dets"]
                )
                maximum_queues[ramp_idx] = max(maximum_queues[ramp_idx], queue)
    finally:
        traci.close()

    return maximum_queues


def main():
    parser = argparse.ArgumentParser(
        description="Estimate maximum ramp queues with all meters held red."
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--simulation-steps", type=int, default=3600)
    args = parser.parse_args()

    results = {
        seed: run_simulation(seed, args.simulation_steps)
        for seed in range(args.seeds)
    }
    estimates = {
        ramp_idx: max(seed_result[ramp_idx] for seed_result in results.values())
        for ramp_idx in RAMP_CONFIG
    }

    output_dir = os.path.join(REPO_ROOT, "2_data_produced")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multi_ramp_max_queue_estimates.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "simulation_steps": args.simulation_steps,
                "seeds": args.seeds,
                "per_seed": results,
                "estimates": estimates,
            },
            f,
            indent=2,
        )

    for ramp_idx, estimate in estimates.items():
        print(f"Ramp {ramp_idx}: maximum observed queue = {estimate:.0f} vehicles")
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
