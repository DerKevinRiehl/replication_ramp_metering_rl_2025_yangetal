import os
import sys

import numpy as np
import torch
import traci

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from action_replacement import calculate_lower_bound
from config import (
    NUM_RAMPS, SIM_STEPS_PER_CONTROL, ALPHA, R_MIN_RATIO,
)

MIN_RATIO = 0.1
MAX_RATIO = 1.0

# Paper Section 4.2: optimal (k_r, k_p) and critical occupancy per ramp,
# found via grid search.  Detector IDs and max queue capacities derived from
# the SUMO network definition.
RAMP_CONFIG = {
    1: {
        'k_r': 110.0, 'k_p': 20.0, 'target_occ': 12.3,
        'dn_dets': [f"det_a1_dn1_{i}" for i in range(4)],
        'ramp_arr_dets': ["det_a1_ramp_arr_0", "det_a1_ramp_arr_1"],
        'ramp_dep_dets': ["det_a1_ramp_dep_0", "det_a1_ramp_dep_1"],
        'queue_dets': ["det_ramp_1_queue_0", "det_ramp_1_queue_1"],
        'max_queue': 84, 'ramp_capacity': 2540
    },
    2: {
        'k_r': 100.0, 'k_p': 0.0, 'target_occ': 12.5,
        'dn_dets': [f"det_a2_dn1_{i}" for i in range(5)],
        'ramp_arr_dets': ["det_a2_ramp_arr_0", "det_a2_ramp_arr_1"],
        'ramp_dep_dets': ["det_a2_ramp_dep_0", "det_a2_ramp_dep_1"],
        'queue_dets': ["det_ramp_2_queue_0", "det_ramp_2_queue_1"],
        'max_queue': 112, 'ramp_capacity': 2650
    },
    3: {
        'k_r': 100.0, 'k_p': 0.0, 'target_occ': 13.0,
        'dn_dets': [f"det_a3_dn1_{i}" for i in range(4)],
        'ramp_arr_dets': ["det_a3_ramp_arr_0", "det_a3_ramp_arr_1"],
        'ramp_dep_dets': ["det_a3_ramp_dep_0", "det_a3_ramp_dep_1"],
        'queue_dets': ["det_ramp_3_queue_0", "det_ramp_3_queue_1"],
        'max_queue': 58, 'ramp_capacity': 1930
    },
    4: {
        'k_r': 100.0, 'k_p': 20.0, 'target_occ': 13.5,
        'dn_dets': [f"det_a4_dn1_{i}" for i in range(4)],
        'ramp_arr_dets': ["det_a4_ramp_arr_0", "det_a4_ramp_arr_1"],
        'ramp_dep_dets': ["det_a4_ramp_dep_0", "det_a4_ramp_dep_1"],
        'queue_dets': ["det_ramp_4_queue_0", "det_ramp_4_queue_1"],
        'max_queue': 164, 'ramp_capacity': 2300
    },
}


class BaseController:
    def execute_control(self, raw_state=None, is_training=False):
        raise NotImplementedError


class NoControlBaseline(BaseController):
    def execute_control(self, raw_state=None, is_training=False):
        ratios = {i: 1.0 for i in range(1, NUM_RAMPS + 1)}
        return ratios, None, None, None, None, ({}, {})


class PiAlineaController(BaseController):
    """Independent PI-ALINEA controller for each ramp (Eq. 16 of paper)."""

    def __init__(self, min_ratio=MIN_RATIO, max_ratio=MAX_RATIO):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prev_dn_occ = {i: None for i in range(1, NUM_RAMPS + 1)}
        self.prev_ratio = {i: 1.0 for i in range(1, NUM_RAMPS + 1)}

    def _get_dn_occ(self, ramp_idx):
        dets = RAMP_CONFIG[ramp_idx]['dn_dets']
        return np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in dets])

    def _compute_alinea(self, ramp_idx):
        params = RAMP_CONFIG[ramp_idx]
        dn_occ = self._get_dn_occ(ramp_idx)

        if self.prev_dn_occ[ramp_idx] is None:
            self.prev_dn_occ[ramp_idx] = dn_occ

        ratio = (self.prev_ratio[ramp_idx] +
                 params['k_r']/params['ramp_capacity'] * (params['target_occ'] - dn_occ) -
                 params['k_p']/params['ramp_capacity'] * (dn_occ - self.prev_dn_occ[ramp_idx]))

        self.prev_dn_occ[ramp_idx] = dn_occ
        return max(self.min_ratio, min(self.max_ratio, ratio))

    def execute_control(self, raw_state=None, is_training=False):
        action_ratios = {}
        for ramp_idx in range(1, NUM_RAMPS + 1):
            ratio = self._compute_alinea(ramp_idx)
            self.prev_ratio[ramp_idx] = ratio
            action_ratios[ramp_idx] = ratio
        return action_ratios, None, None, None, None, ({}, {})


class HeroController(BaseController):
    """
    HERO: Heuristic ramp-metering coordination (Papamichail & Papageorgiou, 2008).

    Each ramp runs PI-ALINEA independently.  When a ramp's queue exceeds
    QUEUE_ON it becomes a *master* bottleneck and begins recruiting upstream
    *slave* ramps.  Slaves are recruited sequentially (S1 = master-1,
    S2 = master-2, …) one per consecutive stressed control step, forming an
    expanding cluster.  A slave's ALINEA rate is overridden entirely: if its
    current queue is below SLAVE_TARGET_QUEUE_PCT the controller forces the
    minimum metering ratio to build a holding queue on the ramp; once the
    target is met ALINEA resumes.  The cluster dissolves when the master's
    queue falls below QUEUE_OFF *or* its downstream mainstream occupancy
    falls below 85 % of its critical value.
    """

    # ASSUMED VALUES
    QUEUE_ON = 0.5
    QUEUE_OFF = 0.3
    # Target minimum queue (as fraction of max_queue) imposed on slave ramps.
    SLAVE_TARGET_QUEUE_PCT = 0.2

    def __init__(self, min_ratio=MIN_RATIO, max_ratio=MAX_RATIO):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prev_dn_occ = {i: None for i in range(1, NUM_RAMPS + 1)}
        self._alinea_ratio = {i: 1.0 for i in range(1, NUM_RAMPS + 1)}
        # Number of consecutive control steps each ramp has been a stressed master.
        # 0 means the ramp is not currently a master.
        self._master_stress_steps = {i: 0 for i in range(1, NUM_RAMPS + 1)}

    def _get_dn_occ(self, ramp_idx):
        dets = RAMP_CONFIG[ramp_idx]['dn_dets']
        return np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in dets])

    def _get_queue(self, ramp_idx):
        dets = RAMP_CONFIG[ramp_idx]['queue_dets']
        return sum(traci.lanearea.getLastStepVehicleNumber(d) for d in dets)

    def _get_queue_pct(self, ramp_idx):
        return self._get_queue(ramp_idx) / RAMP_CONFIG[ramp_idx]['max_queue']

    def _compute_alinea(self, ramp_idx):
        params = RAMP_CONFIG[ramp_idx]
        dn_occ = self._get_dn_occ(ramp_idx)

        if self.prev_dn_occ[ramp_idx] is None:
            self.prev_dn_occ[ramp_idx] = dn_occ

        ratio = (self._alinea_ratio[ramp_idx] +
                 params['k_r']/params['ramp_capacity'] * (params['target_occ'] - dn_occ) -
                 params['k_p']/params['ramp_capacity'] * (dn_occ - self.prev_dn_occ[ramp_idx]))

        self.prev_dn_occ[ramp_idx] = dn_occ
        return max(self.min_ratio, min(self.max_ratio, ratio))

    def execute_control(self, raw_state=None, is_training=False):
        alinea_rates = {}
        queue_pcts = {}
        dn_occs = {}

        for i in range(1, NUM_RAMPS + 1):
            alinea_rates[i] = self._compute_alinea(i)
            self._alinea_ratio[i] = alinea_rates[i]
            queue_pcts[i] = self._get_queue_pct(i)
            dn_occs[i] = self.prev_dn_occ[i]

        action_ratios = dict(alinea_rates)

        # slave_assignments: slave ramp index -> target minimum queue fraction.
        # Built from downstream masters first; a ramp already claimed as slave
        # (by a more downstream master) or active as its own master is skipped.
        slave_assignments = {}

        for i in range(NUM_RAMPS, 1, -1):
            target_occ = RAMP_CONFIG[i]['target_occ']
            is_master = self._master_stress_steps[i] > 0

            # Activate coordination when the ramp queue becomes stressed.
            if not is_master:
                if queue_pcts[i] > self.QUEUE_ON:
                    self._master_stress_steps[i] = 1
            else:
                # Dissolve the cluster when queue or downstream occupancy recovers.
                if queue_pcts[i] < self.QUEUE_OFF or dn_occs[i] < target_occ * 0.85:
                    self._master_stress_steps[i] = 0
                else:
                    self._master_stress_steps[i] += 1

            # Recruit one additional upstream slave per consecutive stressed
            # step (S1 on step 1, S2 added on step 2, …).
            if self._master_stress_steps[i] > 0:
                num_slaves = self._master_stress_steps[i]
                for k in range(1, num_slaves + 1):
                    slave_idx = i - k
                    if slave_idx < 1:
                        break
                    # A ramp that is itself a master keeps its own coordination role.
                    if slave_idx in slave_assignments or self._master_stress_steps[slave_idx] > 0:
                        continue
                    slave_assignments[slave_idx] = self.SLAVE_TARGET_QUEUE_PCT

        # Force min_ratio to build the slave's holding queue; once the target
        # queue is reached, let ALINEA govern normally.
        for slave_idx, target_queue_pct in slave_assignments.items():
            if queue_pcts[slave_idx] < target_queue_pct:
                action_ratios[slave_idx] = self.min_ratio

        return action_ratios, None, None, None, None, ({}, {})


class RLController(BaseController):
    def __init__(self, agent, state_tracker, normalize_fnc, use_replacement=False):
        self.agent = agent
        self.state_tracker = state_tracker
        self.normalize_fnc = normalize_fnc
        self.use_replacement = use_replacement

    def execute_control(self, raw_state, is_training=False):
        """Return (action_ratios, log_prob, value, actions, state_tensor, extras).

        extras = (replaced_dict, lower_bounds_dict)

        actions are raw samples from the policy distribution and are stored in the
        PPO buffer. action_ratios are clamped/replaced ratios used only to step
        the environment.
        """
        state = self.normalize_fnc(raw_state, self.state_tracker)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            dist, value = self.agent(state_tensor)
            actions = dist.sample() if is_training else dist.mean
            action_ratios_t = torch.clamp(actions, 0.0, 1.0).squeeze(0)

        action_ratios = {}
        replaced_dict = {}
        lower_bounds = {}

        for i in range(1, NUM_RAMPS + 1):
            cfg = RAMP_CONFIG[i]
            ratio = action_ratios_t[i - 1].item()

            ramp_offset = (i - 1) * 7
            prev_demand = raw_state[ramp_offset + 4]   # ramp_arr
            curr_queue = raw_state[ramp_offset + 3]     # queue

            lb = calculate_lower_bound(
                prev_demand, curr_queue,
                cfg['ramp_capacity'] / 3600.0,  # convert stored veh/h → veh/s
                SIM_STEPS_PER_CONTROL,
                cfg['max_queue'], ALPHA, R_MIN_RATIO,
            )
            replaced = int(ratio < lb)

            if self.use_replacement and replaced:
                ratio = min(1.0, max(ratio, lb))

            action_ratios[i] = ratio
            replaced_dict[i] = replaced
            lower_bounds[i] = lb

        # Calculate log_prob on the RAW sampled actions to preserve PPO math.
        log_prob = dist.log_prob(actions).sum(dim=-1)

        # Return raw actions for trajectory storage and action_ratios for env step.
        return action_ratios, log_prob, value.item(), actions, state_tensor, (replaced_dict, lower_bounds)
