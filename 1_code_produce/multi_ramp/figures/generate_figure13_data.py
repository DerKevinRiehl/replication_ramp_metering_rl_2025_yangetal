import traci
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(MULTI_RAMP_DIR, "..", ".."))
sys.path.insert(0, MULTI_RAMP_DIR)

from config import SIMULATION_PATH, ramp_times, ramp_demands, main_times

SUMO_BINARY = "sumo"
SIM_END = 4500
SEED = 42
SUMO_CONFIG_PATH = os.path.abspath(os.path.join(MULTI_RAMP_DIR, SIMULATION_PATH))

NUM_LANES = 4
RAMPS = {
    "Ramp 1": {
        "loc": (1, 1),
        "x_range": (1400, 3600),
        "q0": 8196,
        "detectors": {
            "up":  {"ids": [f"det_a1_up_{i}"  for i in range(4)], "shift": 0},
            "down1": {"ids": [f"det_a1_dn1_{i}" for i in range(4)],   "shift": 30},
            "down2": {"ids": [f"det_a1_dn2_{i}" for i in range(4)],   "shift": 58},
        },
    },
    "Ramp 2": {
        "loc": (0, 1),
        "x_range": (2100, 4600),
        "q0": 7584,
        "detectors": {
            "up": {"ids": [f"det_a2_up_{i}"  for i in range(4)], "shift": 0},
            "down1": {"ids": [f"det_a2_dn1_{i}" for i in range(1, 5)], "shift": 29},
            "down2": {"ids": [f"det_a2_dn2_{i}" for i in range(4)],   "shift": 51},
        },
    },
    "Ramp 3": {
        "loc": (1, 0),
        "x_range": (1500, 3900),
        "q0": 7476,
        "detectors": {
            "up": {"ids": [f"det_a3_up_{i}"  for i in range(4)]
                        , "shift": 0},
            "down1": {"ids": [f"det_a3_dn1_{i}" for i in range(4)],   "shift": 22},
            "down2": {"ids": [f"det_a3_dn2_{i}" for i in range(1, 5)], "shift": 43},
        },
    },
    "Ramp 4": {
        "loc": (0, 0),
        "x_range": (1000, 3900),
        "q0": 6792,
        "detectors": {
            "up": {"ids": [f"det_a4_up_{i}"  for i in range(4)], "shift": 0},
            "down1": {"ids": [f"det_a4_dn1_{i}" for i in range(4)],   "shift": 39},
            "down2": {"ids": [f"det_a4_dn2_{i}" for i in range(4)],   "shift": 62},
        },
    },
}

def getAllVehCounts(history):
    for ramp_name, item in RAMPS.items():
        up_ids = item['detectors']['up']['ids']
        down1_ids = item['detectors']['down1']['ids']
        down2_ids = item['detectors']['down2']['ids']

        up_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in up_ids])
        down1_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in down1_ids])
        down2_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in down2_ids])

        history[ramp_name]['up'].append(up_count)
        history[ramp_name]['down1'].append(down1_count)
        history[ramp_name]['down2'].append(down2_count)

    return history

ramp_routes = {
    1:  ["ramp1_to_end", "ramp1_to_off1", "ramp1_to_off2", "ramp1_to_off3", "ramp1_to_off4"],
    2:  ["ramp2_to_end", "ramp2_to_off3", "ramp2_to_off4"],
    3:  ["ramp3_to_end", "ramp3_to_off4"],
    4:  ["ramp4_to_end"]
}

ramp_probs = {
    1:  [0.5, 0.2, 0.1, 0.1, 0.1],
    2:  [0.6, 0.2, 0.2],
    3:  [0.7, 0.3],
    4:  [1.0]
}

def get_ramp_flow(rampIndex, t):
    return np.interp(t, ramp_times, ramp_demands[rampIndex])

def insertRampVehicles(rampIndex, t):
    V = get_ramp_flow(rampIndex, t)
    p = V / (3600 * 2)

    for lane in range(2):
        if np.random.random() < p:
            route = np.random.choice(ramp_routes[rampIndex], p=ramp_probs[rampIndex])

            traci.vehicle.add(f"ramp{rampIndex}_{t}_{lane}", route, typeID="car_ramp",
                departLane="best", departPos="free", departSpeed="random")

main_demands = [0]*len(main_times)
main_routes =  ["main_to_end", "main_to_off1", "main_to_off2", "main_to_off3", "main_to_off4"]
main_probs = [0.6, 0.1, 0.1, 0.1, 0.1]

def get_main_flow(t):
    return np.interp(t, main_times, main_demands)

def insertMainVehicles(t):
    V = get_main_flow(t)
    p = V / (3600 * 2)

    for lane in range(4):
        if np.random.random() < p:
            route = np.random.choice(main_routes, p=main_probs)

            traci.vehicle.add(f"main_{t}_{lane}", route, typeID="car_main",
                departLane="best", departPos="free", departSpeed="random")

def insertVehicles():
    t = traci.simulation.getTime()

    insertMainVehicles(t)

    for i in range(4):
        insertRampVehicles(i+1, t)

def main():
    np.random.seed(SEED)
    history = {
        ramp_name: {"up": [], "down1": [], "down2": []}
        for ramp_name in RAMPS
    }
    times = []

    traci.start([
        SUMO_BINARY, "-c", SUMO_CONFIG_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(SEED),
    ])
    try:
        for step in tqdm(range(SIM_END)):
            insertVehicles()
            history = getAllVehCounts(history)
            times.append(step)
            traci.simulationStep()
    finally:
        traci.close()

    output_dir = os.path.join(REPO_ROOT, "2_data_produced")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure_13_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "times": np.asarray(times),
                "history": history,
                "ramps": RAMPS,
                "num_lanes": NUM_LANES,
                "seed": SEED,
            },
            f,
        )
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
