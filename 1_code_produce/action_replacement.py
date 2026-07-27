def calculate_lower_bound(demand_prev_veh_s, current_queue,
                            ramp_capacity_veh_s, sim_steps, w_max, alpha, r_min):
    max_allowed_queue = alpha * w_max
    available_storage = max_allowed_queue - current_queue
    absorption_rate_veh_s = available_storage / sim_steps
    required_discharge_veh_s = demand_prev_veh_s - absorption_rate_veh_s
    r_lb_raw = required_discharge_veh_s / ramp_capacity_veh_s
    r_lb = min(1.0, max(r_min, r_lb_raw))
    return r_lb


def calculate_penalty(current_queue, demand_current, action_ratio,
                        ramp_capacity_veh_s, sim_steps, w_max, alpha,
                        penalty_scaling_factor=0.1):
    discharge_current = action_ratio * ramp_capacity_veh_s
    net_accumulation_veh_s = demand_current - discharge_current
    net_accumulation_per_step = net_accumulation_veh_s * sim_steps

    if net_accumulation_per_step <= 0:
        return 0.0

    remaining_capacity = (alpha * w_max) - current_queue
    steps_to_spillback = max(1.0, remaining_capacity / net_accumulation_per_step)
    penalty = current_queue / (steps_to_spillback + 1.0)
    # print("Steps to spillback:", steps_to_spillback, "Current queue:", current_queue, "Action ratio:", action_ratio)
    return penalty_scaling_factor * penalty
