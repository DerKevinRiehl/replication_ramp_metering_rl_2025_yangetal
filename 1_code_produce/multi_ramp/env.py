import numpy as np
import traci

from config import (
    ramp_demands, main_demands, TLS_IDS, NUM_RAMPS,
    ramp_times, ramp_routes, ramp_probs,
    main_times, main_routes, main_probs,
)
from controllers import RAMP_CONFIG


class MultiRampMeterEnv:
    def __init__(self, sumo_cmd, max_speed=27.78):
        self.sumo_cmd = sumo_cmd
        self.max_speed = max_speed

    def start(self):
        traci.start(self.sumo_cmd)

    def close(self):
        traci.close()

    # ------------------------------------------------------------------
    # Vehicle insertion
    # ------------------------------------------------------------------

    def _get_ramp_flow(self, ramp_idx, t):
        return np.interp(t, ramp_times, ramp_demands[ramp_idx])

    def _get_main_flow(self, t):
        return np.interp(t, main_times, main_demands)

    def insert_vehicles(self):
        t = traci.simulation.getTime()

        V_main = self._get_main_flow(t)
        p_main = V_main / (3600 * 4)
        for lane in range(4):
            if np.random.random() < p_main:
                route = np.random.choice(main_routes, p=main_probs)
                traci.vehicle.add(
                    f"main_{t}_{lane}", route, typeID="car_main",
                    departLane="best", departPos="free", departSpeed="random",
                )

        for ramp_idx in range(1, NUM_RAMPS + 1):
            V_ramp = self._get_ramp_flow(ramp_idx, t)
            p_ramp = V_ramp / (3600 * 2)
            for lane in range(2):
                if np.random.random() < p_ramp:
                    route = np.random.choice(
                        ramp_routes[ramp_idx], p=ramp_probs[ramp_idx],
                    )
                    traci.vehicle.add(
                        f"ramp{ramp_idx}_{t}_{lane}", route, typeID="car_ramp",
                        departLane="best", departPos="free", departSpeed="random",
                    )

    # ------------------------------------------------------------------
    # State observation
    # ------------------------------------------------------------------

    def _get_aggregate(self, detectors, interval_steps):
        if not detectors:
            return 0.0, 0.0, 0.0

        occ = np.mean([
            traci.inductionloop.getLastIntervalOccupancy(d) for d in detectors
        ])
        raw_speeds = [
            traci.inductionloop.getLastIntervalMeanSpeed(d) for d in detectors
        ]
        speeds = [s if s >= 0 else self.max_speed for s in raw_speeds]
        speed = np.mean(speeds)

        veh_total = np.sum([
            traci.inductionloop.getLastIntervalVehicleNumber(d) for d in detectors
        ])
        veh_per_sec = veh_total / interval_steps

        return occ, speed, veh_per_sec

    def get_traffic_state(self, interval_steps):
        """Return per-ramp traffic measurements.

        Returns a dict keyed by ramp index (1..4), each containing:
            dn_occ, dn_speed, dn_veh, queue, ramp_arr, ramp_dep
        """
        state = {}
        for i in range(1, NUM_RAMPS + 1):
            cfg = RAMP_CONFIG[i]

            dn_occ, dn_speed, dn_veh = self._get_aggregate(
                cfg['dn_dets'], interval_steps,
            )

            arr_total = np.sum([
                traci.inductionloop.getLastIntervalVehicleNumber(d)
                for d in cfg['ramp_arr_dets']
            ])
            dep_total = np.sum([
                traci.inductionloop.getLastIntervalVehicleNumber(d)
                for d in cfg['ramp_dep_dets']
            ])
            ramp_arr = arr_total / interval_steps
            ramp_dep = dep_total / interval_steps

            queue = sum(
                traci.lanearea.getJamLengthVehicle(d) for d in cfg['queue_dets']
            )

            state[i] = {
                "dn_occ": dn_occ,
                "dn_speed": dn_speed,
                "dn_veh": dn_veh,
                "queue": queue,
                "ramp_arr": ramp_arr,
                "ramp_dep": ramp_dep,
            }
        return state

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def apply_actions_and_get_tts(self, green_durations, sim_steps_per_control):
        """Run the full control interval, applying per-ramp green-then-red.

        Args:
            green_durations: dict {ramp_idx: green_steps}
            sim_steps_per_control: total sub-steps in one control interval
        Returns:
            Total Time Spent (vehicle-seconds) accumulated during the interval.
        """
        tts = 0
        pending = 0
        speed = 0
        total_steps = int(sim_steps_per_control)

        for sub_step in range(total_steps):
            for ramp_idx in range(1, NUM_RAMPS + 1):
                if sub_step < green_durations[ramp_idx]:
                    traci.trafficlight.setRedYellowGreenState(
                        TLS_IDS[ramp_idx], "GG",
                    )
                else:
                    traci.trafficlight.setRedYellowGreenState(
                        TLS_IDS[ramp_idx], "rr",
                    )

            self.insert_vehicles()
            traci.simulationStep()
            pending += len(traci.simulation.getPendingVehicles())
            vehicle_ids = traci.vehicle.getIDList()
            if vehicle_ids:
                speed += np.mean([traci.vehicle.getSpeed(v) for v in vehicle_ids])
            tts += traci.vehicle.getIDCount()

        return tts, pending, speed / total_steps