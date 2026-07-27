import os
import sys
import traci

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import (RAMP_DETS, SIM_STEPS_PER_CONTROL, SIMULATION_PATH, TLS_ID)
from env import RampMeterEnv

def evaluate_max_queue():
    sumo_cmd = ["sumo", "-c", SIMULATION_PATH]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=None,
        downstream_dets=None, ramp_arr_dets=None,
        ramp_dep_dets=None, ramp_detector=None
    )
    env.start()

    max_queue = 0

    for step in range(400):
        # Advance the simulation
        red_duration = SIM_STEPS_PER_CONTROL

        _ = env.apply_action_and_get_tts(0, red_duration)

        if step % SIM_STEPS_PER_CONTROL == 0:
            current_queue = sum(traci.lanearea.getJamLengthVehicle(det) for det in RAMP_DETS)

            # Update max queue
            if current_queue > max_queue:
                max_queue = current_queue

            print(f"Sim Step {step:04d} | Current Queue: {current_queue:02d} | Max Queue so far: {max_queue:02d}")

    traci.close()
    print("-" * 40)
    print(f"Test Complete. Absolute Maximum Queue Reached: {max_queue}")

if __name__ == "__main__":
    evaluate_max_queue()