import os
from matplotlib import pyplot as plt
import numpy as np

from config import DATA_DIR
from plot_utils import load_pickle, save_figure

# ------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------

NUM_SIMS = 10

demand_step = 100
base_demand = 1600


train_history = load_pickle(os.path.join(DATA_DIR, "multi_ramp_queue_data.pkl"))


# ------------------------------------------------------------------------
# POST-PROCESSING
# ------------------------------------------------------------------------
def isBreakdown(history, threshold, persistence):
    consecutive = 0
    for q in history:
        if q > threshold:
            consecutive += 1
            if consecutive >= persistence:
                return 1
        else:
            consecutive = 0

    return 0

thresholds = [35, 45, 25, 70]
persistence = 300
breakdowns_per_demand = []
for demand, hist in train_history.items():
    num_breakdowns = [0]*4
    for ramp_name, history in hist['queue'].items():
        ramp_num = int(ramp_name.split()[1])-1
        for ep in history.values():
            num_breakdowns[ramp_num] += isBreakdown(ep, thresholds[ramp_num], persistence)
    breakdowns_per_demand.append(num_breakdowns)


demand_values = [
    base_demand + i * demand_step
    for i in range(len(train_history))
]
breakdown_array = np.array(breakdowns_per_demand) / NUM_SIMS  # shape: (n_demands, 4)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i in range(4):
    axes[i].plot(demand_values, breakdown_array[:, i], marker='o')
    axes[i].axhline(0.5, color='r', linestyle='--', label='50% threshold')
    axes[i].set_title(f"Ramp {i+1}")
    axes[i].set_xlabel("Demand (veh/hr)")
    axes[i].set_ylabel("Breakdown Probability")
    axes[i].set_ylim(0, 1)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.tight_layout()

save_figure(plt.gcf(), f"{base_demand}_capacity_calculations.pdf")

plt.show()

# Capacity estimate per ramp
for i in range(4):
    for d, p in zip(demand_values, breakdown_array[:, i]):
        if p >= 0.5:
            print(f"Ramp {i+1} capacity ≈ {d} veh/hr (first demand with ≥50% breakdown)")
            break