import pickle

import traci
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SIMULATION_PATH, RAMP_ARR_DETS, RAMP_DEP_DETS, RAMP_DETS

# ------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------
SUMO_BINARY = "sumo"
SIM_END = 3600
CONTROL_STEP = 15

NUM_LANES = 4

NUM_SIMS = 10

ramp_times = [0,3600]

n_demands = 15
demand_step = 50
base_demand = 1600

all_ramp_demands = {
    i: [base_demand+(i*demand_step)]*len(ramp_times) for i in range(n_demands)
}

train_history = {}

for i in range(n_demands):
    train_history[f"Demand {i}"] = {"queue": {}}

# ------------------------------------------------------------------------
# FUNCTIONS FOR SIMULATION
# ------------------------------------------------------------------------

def get_mainline_flow(t):
    return np.interp(t, [0,3600], [0, 0])

def get_ramp_flow(t, ramp_times, ramp_demand):
    return np.interp(t, ramp_times, ramp_demand)

def insertVehicles(ramp_demand):
    t = traci.simulation.getTime()

    V_main = get_mainline_flow(t)
    p_main = V_main / (3600 * 4)

    for lane in range(4):
        if np.random.random() < p_main:
            traci.vehicle.add(f"main_{t}_{lane}", "route_main", typeID="car_main",
                departLane="best", departPos="free", departSpeed="random")

    # Process Ramp
    V_ramp = get_ramp_flow(t, ramp_times, ramp_demand)
    p_ramp = V_ramp / (3600 * 2)

    for lane in range(2):
        if np.random.random() < p_ramp:
            traci.vehicle.add(f"ramp_{t}_{lane}", "route_ramp", typeID="car_ramp",
                departLane="best", departPos="free", departSpeed="random")

def getQueueCount(history_queue):
    history_queue.append(np.sum([traci.lanearea.getLastStepVehicleNumber(d) for d in RAMP_DETS]))

    return history_queue

# ------------------------------------------------------------------------
# RUN SIMULATION
# ------------------------------------------------------------------------

def runSim(demand_num, ramp_demands, episode):
    seed = episode

    history_queue = []

    sumo_cmd = [SUMO_BINARY, "-c", SIMULATION_PATH, "--no-step-log", "true", "--no-warnings", "--seed", str(seed)]
    traci.start(sumo_cmd)

    np.random.seed(seed)

    times = []

    for i in (range(SIM_END)):
        insertVehicles(ramp_demands)

        history_queue = getQueueCount(history_queue)

        times.append(i)

        traci.simulationStep()

    traci.close()

    train_history[f"Demand {demand_num}"]['queue'][episode] = history_queue

for i in tqdm(range(n_demands)):
    ramp_demands = all_ramp_demands[i]
    for j in (range(NUM_SIMS)):
        runSim(i, ramp_demands, j)

# ------------------------------------------------------------------------
# SAVING DATA
# ------------------------------------------------------------------------

data_prod_dir = os.path.join("..", "..", "2_data_produced")
with open(os.path.join(data_prod_dir, "ramp_queue_data.pkl"), "wb") as f:
    pickle.dump(train_history, f)