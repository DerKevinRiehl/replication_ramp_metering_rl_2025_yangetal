import os
import pickle
import sys

import numpy as np
import traci

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(MULTI_RAMP_DIR, "..", ".."))
sys.path.insert(0, MULTI_RAMP_DIR)

from config import SIMULATION_PATH
from env import MultiRampMeterEnv

SIMULATION_STEPS = 3600
SEED = 42
SUMO_CONFIG_PATH = os.path.abspath(os.path.join(MULTI_RAMP_DIR, SIMULATION_PATH))

RAMP_DETECTORS = {
    "(4) Inner Ring South": [f"det_a1_dn1_{i}" for i in range(4)],
    "(3) Shiyang": [f"det_a2_dn1_{i}" for i in range(1, 5)],
    "(1) Maquan": [f"det_a3_dn1_{i}" for i in range(4)],
    "(2) Shaungqi": [f"det_a4_dn1_{i}" for i in range(4)],
}


def main():
    np.random.seed(SEED)
    env = MultiRampMeterEnv([
        "sumo", "-c", SUMO_CONFIG_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(SEED),
    ])
    times = []
    raw_counts = {name: [] for name in RAMP_DETECTORS}

    env.start()
    try:
        for _ in range(SIMULATION_STEPS):
            env.insert_vehicles()
            traci.simulationStep()
            times.append(traci.simulation.getTime())
            for name, detector_ids in RAMP_DETECTORS.items():
                raw_counts[name].append(sum(
                    traci.inductionloop.getLastStepVehicleNumber(detector)
                    for detector in detector_ids
                ))
    finally:
        env.close()

    output_dir = os.path.join(REPO_ROOT, "2_data_produced")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure_12_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "times": np.asarray(times),
                "raw_counts": raw_counts,
                "flow_window": 300,
                "seed": SEED,
            },
            f,
        )
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
