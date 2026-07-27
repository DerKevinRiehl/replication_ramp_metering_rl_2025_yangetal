import traci
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SINGLE_RAMP_DIR, "..", ".."))
sys.path.insert(0, SINGLE_RAMP_DIR)

from config import SIMULATION_PATH, TLS_ID
from env import RampMeterEnv

SUMO_CONFIG_PATH = os.path.abspath(os.path.join(SINGLE_RAMP_DIR, SIMULATION_PATH))

# =========================
# CONFIG
# =========================

SUMO_BINARY = "sumo"

STEP_LENGTH = 1
SIM_END = 2300

q0 = 8160 / 3600.0   # veh/sec

# detector groups
detector_groups = {
    "+920 m (upstream)": [
        "det_pre_merge_0","det_pre_merge_1","det_pre_merge_2","det_pre_merge_3"
    ],
    "+1225 m (downstream1)": [
        "det_loc2_0","det_loc2_1","det_loc2_2","det_loc2_3"
    ],
    "+1475 m (downstream2)": [
        "det_loc3_0","det_loc3_1","det_loc3_2","det_loc3_3"
    ]
}

# downstream travel time shifts from paper
time_shift = {
    "+920 m (upstream)": 0,
    "+1225 m (downstream1)": 16,
    "+1475 m (downstream2)": 25
}


# =========================
# insert logic
# =========================

MAIN_T = [0, 600, 600.1, 3300, 3600, 4200]
MAIN_VEH = [7400, 7400, 7900, 7900, 4000, 4000]

RAMP_T = [0, 600, 600.1, 3300, 3600, 4200]
RAMP_VEH = [600, 1000, 1300, 1300, 500, 500]

def get_mainline_flow(t):
    return np.interp(t, MAIN_T, MAIN_VEH)

def get_ramp_flow(t):
    return np.interp(t, RAMP_T, RAMP_VEH)

def insertVehicles():
    t = traci.simulation.getTime()

    V_main = get_mainline_flow(t)
    p_main = V_main / (3600 * 4)

    for lane in range(4):
        if np.random.random() < p_main:
            traci.vehicle.add(f"main_{t}_{lane}", "route_main", typeID="car_main",
                departLane="best", departPos="free", departSpeed="random")

    # Process Ramp
    V_ramp = get_ramp_flow(t)
    p_ramp = V_ramp / (3600 * 2)

    for lane in range(2):
        if np.random.random() < p_ramp:
            traci.vehicle.add(f"ramp_{t}_{lane}", "route_ramp", typeID="car_ramp",
                departLane="best", departPos="free", departSpeed="random")

# =========================
# START SUMO
# =========================

traci.start([SUMO_BINARY, "-c", SUMO_CONFIG_PATH])

times = []
cumulative = {k: 0 for k in detector_groups}
records = {k: [] for k in detector_groups}

# =========================
# SIM LOOP
# =========================

cumulative = {k: 0 for k in detector_groups}
records = {k: [] for k in detector_groups}
times = []

while traci.simulation.getTime() < SIM_END:
    insertVehicles()
    traci.simulationStep()
    t = traci.simulation.getTime()
    times.append(t)

    for loc, dets in detector_groups.items():
        step_count = 0
        # before = step_count
        for det in dets:
            # FIX: Only count vehicles that have completely passed to avoid double-counting
            step_count += traci.inductionloop.getLastStepVehicleNumber(det)
            # Note: If values are still high, use: len(traci.inductionloop.getPassedVehiclesID(det))

        cumulative[loc] += step_count
        records[loc].append(cumulative[loc])

traci.close()

# =========================
# POST PROCESS
# =========================

times = np.array(times)

modified_curves = {}

lane_count = {
    "+920 m (upstream)": 4,
    "+1225 m (downstream1)": 4,
    "+1475 m (downstream2)": 4
}

for loc in records:

    N = np.array(records[loc])

    # convert to per-lane cumulative
    N = N / lane_count[loc]

    # subtract background flow PER LANE
    q0_lane = (8160/3600.0) / 4.0   # assuming 4 mainline lanes
    N_mod = N - q0_lane * times

    # shift downstream curves left
    shift_steps = int(time_shift[loc] / STEP_LENGTH)

    if shift_steps > 0:
        N_mod = N_mod[shift_steps:]
        t_mod = times[:-shift_steps]
    else:
        t_mod = times

    modified_curves[loc] = (t_mod, N_mod)

data_prod_dir = os.path.join(REPO_ROOT, "2_data_produced")
os.makedirs(data_prod_dir, exist_ok=True)
with open(os.path.join(data_prod_dir, "figure_6_data.pkl"), "wb") as f:
    pickle.dump(modified_curves, f)

