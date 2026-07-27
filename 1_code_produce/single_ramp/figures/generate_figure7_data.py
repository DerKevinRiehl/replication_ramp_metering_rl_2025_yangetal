import numpy as np

import traci
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_RAMP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SINGLE_RAMP_DIR, "..", ".."))
sys.path.insert(0, SINGLE_RAMP_DIR)

from config import SUMO_PATH

SUMO_NETWORK_DIR = os.path.abspath(os.path.join(SINGLE_RAMP_DIR, SUMO_PATH))
SUMO_NET_FILE = os.path.join(SUMO_NETWORK_DIR, "data", "network.net.xml")
SUMO_CFG_FILE = os.path.join(SUMO_NETWORK_DIR, "data", "simulation.sumocfg")
SUMO_ROU_FILE = os.path.join(SUMO_NETWORK_DIR, "data", "dynamic_routes.rou.xml")

# Grid Search Parameters (from the paper)
MAINLINE_FLOWS = [7400, 7800, 8200, 8600, 9000, 9400]
RAMP_FLOWS = [600, 800, 1000, 1200, 1400, 1600]

# MAINLINE_FLOWS = [5800+(i*100) for i in range(6)]
# RAMP_FLOWS = [600+(i*100) for i in range(6)]

# Simulation parameters
SIM_STEPS = 3600
WARMUP_STEPS = 1800
TLS_ID = "junction_ramp"
DOWNSTREAM_DETS = [f"det_loc2_{i}" for i in range(4)]

def generate_route_file(main_flow, ramp_flow):
    vtype = """<vType id="paper_car" length="5.0" minGap="2.0" accel="2.6" decel="4.5" sigma="0.3" tau="1.1"
                    lcCooperative="1.0" lcSpeedGain="2.5" lcImpatience="1.0"
                    lcOvertakeRight="0.3" lcLookaheadLeft="0.5" lcAssertive="3.0" lcStrategic="0.8"/>"""

    # Define the physical routes through your updated network
    routes = """
    <route id="route_main" edges="edge_virtual_main edge_mainline edge_merge edge_downstream"/>
    <route id="route_ramp" edges="edge_virtual_ramp edge_ramp_2 edge_ramp_out edge_merge edge_downstream"/>
    """

    # Define continuous flows for 1 hour
    flows = f"""
    <flow id="flow_main" type="paper_car" route="route_main" begin="0" end="{SIM_STEPS}" vehsPerHour="{main_flow}" departLane="best" departSpeed="max"/>
    <flow id="flow_ramp" type="paper_car" route="route_ramp" begin="0" end="{SIM_STEPS}" vehsPerHour="{ramp_flow}" departLane="best" departSpeed="max"/>
    """

    with open(SUMO_ROU_FILE, "w") as f:
        f.write(f'<routes>\n{vtype}\n{routes}\n{flows}\n</routes>')


def run_simulation_and_get_occupancy():
    sumo_cmd = [
        "sumo",
        "-c", SUMO_CFG_FILE,
        "-r", SUMO_ROU_FILE,
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--collision.action", "none",
        "--no-warnings"
    ]

    traci.start(sumo_cmd)

    occupancies = []

    for step in range(SIM_STEPS):
        traci.trafficlight.setRedYellowGreenState(TLS_ID, "GG")
        traci.simulationStep()

        if step > WARMUP_STEPS and step % 15 == 0:
            occ = np.mean([traci.inductionloop.getLastIntervalOccupancy(d) for d in DOWNSTREAM_DETS])
            if occ >= 0:
                occupancies.append(occ)

    traci.close()

    # Return average occupancy across the steady-state window
    return np.mean(occupancies) if occupancies else 0.0

if __name__ == "__main__":
    results = np.zeros((len(RAMP_FLOWS), len(MAINLINE_FLOWS)))

    print("Starting Grid Search for Figure 7...")
    for i, r_flow in enumerate(RAMP_FLOWS):
        for j, m_flow in enumerate(MAINLINE_FLOWS):
            print(f"Testing: Mainline={m_flow} veh/h, Ramp={r_flow} veh/h", end="", flush=True)

            generate_route_file(m_flow, r_flow)
            avg_occ = run_simulation_and_get_occupancy()

            results[i, j] = avg_occ
            print(f" -> Occupancy: {avg_occ:.2f}%")

    data_prod_dir = os.path.join(REPO_ROOT, "2_data_produced")
    os.makedirs(data_prod_dir, exist_ok=True)
    np.save(os.path.join(data_prod_dir, "figure_7_matrix.npy"), results)