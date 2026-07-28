import subprocess
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

MULTI_RAMP_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(MULTI_RAMP_DIR)
sys.path.insert(0, MULTI_RAMP_DIR)
sys.path.insert(0, CODE_DIR)

from config import MODELS_DIR

SEEDS = np.arange(10)

def run_single_experiment(args):
    seed, use_replacement, dynamic_norm = args
    variant = "replacement" if use_replacement else "baseline"
    norm_tag = "dynamic" if dynamic_norm else "static"
    print(f"Starting {variant} ({norm_tag} norm) | seed {seed}...")

    cmd = [
        sys.executable,
        os.path.join(MULTI_RAMP_DIR, "train.py"),
        "--seed", str(seed),
        "--num_episodes", "500",
    ]
    if use_replacement:
        cmd.append("--use_replacement")
    if dynamic_norm:
        cmd.append("--dynamic_norm")

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=MULTI_RAMP_DIR,
    )
    print(f"---- Finished {variant} ({norm_tag} norm) | seed {seed}.")
    return True

def load_and_plot(use_replacement, color):
    variant = "replacement" if use_replacement else "baseline"
    all_scores = []

    for seed in SEEDS:
        file_path = os.path.join(MODELS_DIR, f"training_history_{variant}_seed{seed}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                all_scores.append(data["scores"])
        else:
            print(f"Warning: {file_path} not found.")

    if not all_scores:
        return

    scores_array = np.array(all_scores)
    mean_scores = np.mean(scores_array, axis=0)
    std_scores = np.std(scores_array, axis=0)
    episodes = np.arange(1, len(mean_scores) + 1)

    plt.plot(episodes, mean_scores, label=f"{variant.capitalize()} Mean", color=color)
    plt.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, color=color, alpha=0.2)

if __name__ == "__main__":
    # 1. Setup jobs (Baseline and Replacement for all seeds)
    jobs = [(seed, False, False) for seed in SEEDS] + [(seed, True, False) for seed in SEEDS]

    # Leave one CPU core free so your computer remains usable
    max_workers = max(1, os.cpu_count() - 1)
    print(f"Executing {len(jobs)} jobs across {max_workers} parallel workers...\n")

    # 2. Run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_single_experiment, jobs)

    print("\nAll training runs complete. Generating plots...")

    # 3. Aggregate and plot
    plt.figure(figsize=(10, 6))

    load_and_plot(use_replacement=False, color="blue")
    load_and_plot(use_replacement=True, color="orange")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Training Performance Across Multiple Seeds (Multi-Ramp)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("multi_seed_results.png", dpi=300)
    print("Saved plot to multi_seed_results.png")
