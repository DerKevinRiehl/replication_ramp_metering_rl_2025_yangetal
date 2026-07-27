import os

import numpy as np

SIM_END = 3600

SUMO_PATH = os.path.join("..", "..", "1_data_source", "multi_ramp", "sumo_network")
SIMULATION_PATH = os.path.join(SUMO_PATH, "data", "simulation.sumocfg")

# RL training constants
NUM_RAMPS = 4
STATE_DIM = 28
ACTION_DIM = 4
CONTROL_STEPS_PER_EPISODE = 320
SIM_STEPS_PER_CONTROL = 15.0

ALPHA = 0.9
R_MIN_RATIO = 0.1
RAMP_CAPACITY_VEH_S = 1930.0 / 3600.0

MAX_TTS = 24372.40
AVG_TTS = 18296.09

STATE_MEANS = [17.4871, 15.273, 2.3467, 53.6, 0.1892, 0.1758, 0.3325, 11.4817, 18.0356, 2.1481, 8.9833, 0.2875, 0.3319, 0.67, 14.6207, 17.0733, 2.1056, 20.8583, 0.2294, 0.2242, 0.46, 11.9513, 20.6646, 2.1611, 46.1667, 0.3406, 0.3378, 0.6175]
STATE_STDS = [2.4204, 1.9738, 0.2413, 22.0153, 0.1589, 0.1769, 0.3939, 2.9193, 2.6961, 0.4298, 12.5346, 0.1122, 0.203, 0.4337, 3.9553, 3.0985, 0.4578, 17.9693, 0.1766, 0.2019, 0.4409, 3.4765, 3.2518, 0.4849, 33.1902, 0.1406, 0.2187, 0.4449]

MODELS_DIR = "models"

TLS_IDS = {
    1: "junction_ramp_1",
    2: "junction_ramp_2",
    3: "junction_ramp_3",
    4: "junction_ramp_4",
}

os.makedirs(os.path.join(SUMO_PATH, 'out'), exist_ok=True)

ramp_demands = {
    1:  [1400]*12,
    2:  [1200]*12,
    3:  [900]*12,
    4:  [1600]*12
}

ramp_times = [0] + list(np.arange(600, 3601, step=300))

main_demands = [8000]*12
main_times = [0] + list(np.arange(600, 3601, step=300))

ramp_routes = {
    1:  ["ramp1_to_end", "ramp1_to_off1", "ramp1_to_off2", "ramp1_to_off3", "ramp1_to_off4"],
    2:  ["ramp2_to_end", "ramp2_to_off3", "ramp2_to_off4"],
    3:  ["ramp3_to_end", "ramp3_to_off4"],
    4:  ["ramp4_to_end"]
}

ramp_probs = {
    1:  [1.0, 0.0, 0.0, 0.0, 0.0],
    2:  [1.0, 0.0, 0.0],
    3:  [1.0, 0.0],
    4:  [1.0]
}


main_routes =  ["main_to_end", "main_to_off1", "main_to_off2", "main_to_off3", "main_to_off4"]
main_probs = [0.6, 0.1, 0.1, 0.1, 0.1]

NUM_LANES = 4
RAMPS = {
    "Ramp 1": {
        "loc": (0,0),
        "x_range": (1400, 3600),
        "detectors": {
            "up":  {"ids": [f"det_a1_ramp_dep_{i}"  for i in range(2)], "shift": 0},
            "down1": {"ids": [f"det_a1_dn1_{i}" for i in range(4)],   "shift": 30},
            "down2": {"ids": [f"det_a1_dn2_{i}" for i in range(4)],   "shift": 59},
        },
    },
    "Ramp 2": {
        "loc": (0, 1),
        "x_range": (2100, 4600),
        "detectors": {
            "up": {"ids": [f"det_a2_ramp_dep_{i}"  for i in range(2)], "shift": 0},
            "down1": {"ids": [f"det_a2_dn1_{i}" for i in range(1, 5)], "shift": 32},
            "down2": {"ids": [f"det_a2_dn2_{i}" for i in range(4)],   "shift": 55},
        },
    },
    "Ramp 3": {
        "loc": (1, 0),
        "x_range": (1500, 3900),
        "detectors": {
            "up": {"ids": [f"det_a3_ramp_dep_{i}"  for i in range(2)], "shift": 0},
            "down1": {"ids": [f"det_a3_dn1_{i}" for i in range(4)],   "shift": 20},
            "down2": {"ids": [f"det_a3_dn2_{i}" for i in range(1, 5)], "shift": 41},
        },
    },
    "Ramp 4": {
        "loc": (1, 1),
        "x_range": (1000, 3900),
        "detectors": {
            "up": {"ids": [f"det_a4_ramp_dep_{i}"  for i in range(2)], "shift": 0},
            "down1": {"ids": [f"det_a4_dn1_{i}" for i in range(4)],   "shift": 40},
            "down2": {"ids": [f"det_a4_dn2_{i}" for i in range(4)],   "shift": 64},
        },
    },
}