import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

# ------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------

NUM_SIMS = 10

n_demands = 15
demand_step = 50
base_demand = 1600

data_prod_dir = os.path.join("..", "..", "2_data_produced")
with open(os.path.join(data_prod_dir, "ramp_queue_data.pkl"), "rb") as f:
    train_history = pickle.load(f)

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

threshold = 15
persistence = 300
breakdowns_per_demand = []
for demand, hist in train_history.items():
    num_breakdowns = 0
    for episode, history in hist['queue'].items():
        num_breakdowns += isBreakdown(history, threshold, persistence)
    breakdowns_per_demand.append(num_breakdowns)

# ------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------

demand_values = [base_demand + i * demand_step for i in range(n_demands)]
breakdown_probs = np.array(breakdowns_per_demand) / NUM_SIMS

plt.plot(demand_values, breakdown_probs, marker='o')
plt.axhline(0.5, color='r', linestyle='--', label='50% threshold')
plt.title(f"Ramp Capacity")
plt.xlabel("Demand (veh/hr)")
plt.ylabel("Breakdown Probability")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plots_dir = os.path.join("..", "..", "3_data_visualization")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, f"{base_demand}_capacity_calculations.pdf"), bbox_inches="tight")

plt.show()

# Capacity estimate per ramp
for d, p in zip(demand_values, breakdown_probs):
    if p >= 0.5:
        print(f"Capacity ≈ {d} veh/hr (first demand with ≥50% breakdown)")
        break