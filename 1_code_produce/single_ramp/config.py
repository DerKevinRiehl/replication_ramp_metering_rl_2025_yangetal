import os
import numpy as np

# Constants for the Store-and-Forward Model
W_MAX = 42.0
ALPHA = 0.9
R_MIN_RATIO = 0.1
RAMP_CAPACITY_VEH_S = 1930.0 / 3600.0

MAX_SPEED = 27.78

MAX_TTS = 23285.70
AVG_TTS = 13029.78
STATE_MEANS = [25.4845, 10.0435, 1.9713, 11.0691, 22.9742, 4.6212, 5.6806, 0.4259, 0.4217, 7.5]
STATE_STDS  = [5.545, 3.7435, 0.1753, 1.8837, 2.0336, 0.6329, 8.5927, 0.1456, 0.1438, 7.5]

TLS_ID = "junction_ramp"
SUMO_PATH = os.path.join("..", "..", "1_data_source", "single_ramp", "sumo_network")
SIMULATION_PATH = os.path.join(SUMO_PATH, "data", "simulation.sumocfg")

STATE_DIM = 10
CONTROL_STEPS_PER_EPISODE = 240
SIM_STEPS_PER_CONTROL = 15.0

UPSTREAM_DETS = [f"det_upstream_{i}" for i in range(4)]
DOWNSTREAM_DETS = [f"det_loc2_{i}" for i in range(4)] + [f"det_loc3_{i}" for i in range(4)]
RAMP_ARR_DETS = [f"det_ramp_arr_{i}" for i in range(2)]
RAMP_DEP_DETS = [f"det_ramp_dep_{i}" for i in range(2)]
RAMP_DETS = ["det_ramp_queue_0", "det_ramp_queue_1"]

MAIN_T = [0, 600, 600.1, 3300, 3301, 4200]
RAMP_T = [0, 600, 600.1, 3300, 3600, 4200]

# COPY DEMAND PROFILE FROM PAPER
MAIN_VEH = [7400, 7400, 7900, 7900, 4000, 4000]
RAMP_VEH = [600, 1000, 1300, 1300, 500, 500]

os.makedirs(os.path.join(SUMO_PATH, 'out'), exist_ok=True)

MODELS_DIR = "models"

REPLACEMENT_HISTORY_PATH = os.path.join("..", MODELS_DIR, "training_history_replacement_seed42.pkl")
BASELINE_HISTORY_PATH = os.path.join("..", MODELS_DIR, "v1_training_history_baseline_seed42.pkl")
