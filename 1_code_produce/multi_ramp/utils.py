import math
import xml.etree.ElementTree as ET

import numpy as np
import traci

from config import (ramp_times, ramp_probs, ramp_routes,
                    main_demands, main_times, main_probs, main_routes,
                    NUM_RAMPS, STATE_MEANS, STATE_STDS)

# Ordered sequence of edges along the main highway from start to end
_MAIN_ROUTE_EDGES = [
    "edge_mainline_1", "edge_merge_1",
    "edge_mainline_2", "edge_out_1",
    "edge_mainline_3", "edge_out_2",
    "edge_mainline_4", "edge_merge_2",
    "edge_mainline_5", "edge_merge_3",
    "edge_mainline_6", "edge_out_3",
    "edge_mainline_7", "edge_merge_4",
    "edge_mainline_8",
]


def _shape_length(shape_str):
    pts = [tuple(map(float, p.split(","))) for p in shape_str.strip().split()]
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def getDetectorPositions(net_xml_path, det_xml_path):
    # --- 1. Parse edge lengths -------------------------------------------
    edge_lengths = {}
    for edge in ET.parse(net_xml_path).findall("edge"):
        eid = edge.get("id")
        if eid.startswith(":"):          # skip internal junction edges
            continue
        length = edge.get("length")
        if length:
            edge_lengths[eid] = float(length)
        else:
            shape = edge.get("shape")
            if shape:
                edge_lengths[eid] = _shape_length(shape)

    # --- 2. Accumulate start positions along the main route ---------------
    edge_start = {}
    cumulative = 0.0
    for eid in _MAIN_ROUTE_EDGES:
        edge_start[eid] = cumulative
        cumulative += edge_lengths.get(eid, 0.0)

    # --- 3. Map each induction loop to its absolute highway position ------
    detector_positions = {}
    for loop in ET.parse(det_xml_path).findall("inductionLoop"):
        det_id  = loop.get("id")
        lane_id = loop.get("lane")
        pos     = float(loop.get("pos"))

        # lane id format: "<edge_id>_<lane_index>"  e.g. "edge_mainline_2_0"
        edge_id = "_".join(lane_id.split("_")[:-1])

        if edge_id not in edge_start:
            continue   # ramp or off-ramp detector — not on the main route

        edge_len   = edge_lengths.get(edge_id, 0.0)
        local_pos  = edge_len + pos if pos < 0 else pos   # SUMO: negative = from end
        detector_positions[det_id] = edge_start[edge_id] + local_pos

    return detector_positions

def get_ramp_flow(ramp_demands, rampIndex, t):
    return np.interp(t, ramp_times, ramp_demands[rampIndex])

def insertRampVehicles(ramp_demands, rampIndex, t):
    V = get_ramp_flow(ramp_demands, rampIndex, t)
    p = V / (3600 * 2)

    for lane in range(2):
        if np.random.random() < p:
            route = np.random.choice(ramp_routes[rampIndex], p=ramp_probs[rampIndex])

            traci.vehicle.add(f"ramp{rampIndex}_{t}_{lane}", route, typeID="car_ramp",
                departLane="best", departPos="free", departSpeed="max")

def get_main_flow(main_demands, t):
    return np.interp(t, main_times, main_demands)

def insertMainVehicles(main_demands, t):
    V = get_main_flow(main_demands, t)
    p = V / (3600 * 4)

    for lane in range(4):
        if np.random.random() < p:
            route = np.random.choice(main_routes, p=main_probs)

            traci.vehicle.add(f"main_{t}_{lane}", route, typeID="car_main",
                departLane="best", departPos="free", departSpeed="max")

def insertVehicles(main_demands, ramp_demands):
    t = traci.simulation.getTime()

    insertMainVehicles(main_demands, t)

    for i in range(4):
        insertRampVehicles(ramp_demands, i+1, t)

def getAllVehCounts(ramps, history, history_speed, history_queue):
    for ramp_name, item in ramps.items():
        up_ids = item['detectors']['up']['ids']
        down1_ids = item['detectors']['down1']['ids']
        # down2_ids = item['detectors']['down2']['ids']

        up_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in up_ids])
        down1_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in down1_ids])
        # down2_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in down2_ids])/4

        up_speed = [traci.inductionloop.getLastStepMeanSpeed(d) if traci.inductionloop.getLastStepMeanSpeed(d) > 0 else 27.78 for d in up_ids]
        down1_speed = [traci.inductionloop.getLastStepMeanSpeed(d) if traci.inductionloop.getLastStepMeanSpeed(d) > 0 else 27.78 for d in down1_ids]

        up_speed = np.mean(up_speed)
        down1_speed = np.mean(down1_speed)

        history[ramp_name]['up'].append(up_count)
        history[ramp_name]['down1'].append(down1_count)

        history_speed[ramp_name]['up'].append(up_speed)
        history_speed[ramp_name]['down1'].append(down1_speed)

        history_queue[ramp_name].append(traci.lanearea.getLastStepVehicleNumber(f"det_{ramp_name.lower().replace(' ', '_')}_queue_0") + \
                                        traci.lanearea.getLastStepVehicleNumber(f"det_{ramp_name.lower().replace(' ', '_')}_queue_1"))

    return history, history_speed, history_queue


# ------------------------------------------------------------------
# RL state formatting and normalization
# ------------------------------------------------------------------

def format_state_vector(state_dict, last_ratios):
    """Convert per-ramp traffic state into a flat 28-element list.

    Per ramp (7 features x 4 ramps = 28):
        dn_occ, dn_speed, dn_veh, queue, ramp_arr, ramp_dep, last_ratio
    """
    vec = []
    for i in range(1, NUM_RAMPS + 1):
        s = state_dict[i]
        vec.extend([
            s["dn_occ"], s["dn_speed"], s["dn_veh"],
            s["queue"], s["ramp_arr"], s["ramp_dep"],
            last_ratios[i],
        ])
    return vec


def normalize_static(raw_state, tracker=None):
    raw_state = np.asarray(raw_state)
    return (raw_state - STATE_MEANS) / STATE_STDS


def normalize_dynamic(raw_state, tracker):
    raw_state = np.array(raw_state)
    tracker.push(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8
    return (raw_state - tracker.mean) / std