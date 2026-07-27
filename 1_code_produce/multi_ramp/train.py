import os
import sys
import pickle
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODELS_DIR, STATE_DIM, ACTION_DIM, SIMULATION_PATH,
    CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL, NUM_RAMPS,
)
from env import MultiRampMeterEnv
from controllers import RLController
from runner import run_episode
from utils import normalize_static, normalize_dynamic
from model import SharedActorCritic
from ppo_loss import compute_gae, ppo_update
from stats import RunningStat
from live_plot import init_plot, update_live_plot

def train(use_replacement, seed, num_episodes, dynamic_norm):
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_prefix = "replacement" if use_replacement else "baseline"

    sumo_cmd = [
        "sumo", "-c", SIMULATION_PATH,
        "--no-step-log", "true", "--no-warnings", "--seed", str(seed),
    ]

    env = MultiRampMeterEnv(sumo_cmd=sumo_cmd)

    agent = SharedActorCritic(STATE_DIM, action_dim=ACTION_DIM, log_std=-1.0)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    state_tracker = RunningStat(shape=(STATE_DIM,))

    normalize_fnc = normalize_static
    if dynamic_norm:
        normalize_fnc = normalize_dynamic

    controller = RLController(
        agent=agent,
        state_tracker=state_tracker,
        normalize_fnc=normalize_fnc,
        use_replacement=use_replacement,
    )

    line, ax, fig = init_plot(use_replacement)
    all_scores, history_steps, history_lengths, history_tts = [], [], [], []
    history_replacement, history_lb = [], []
    cumulative_steps = 0

    for episode in range(1, num_episodes + 1):
        env.start()

        trajectory, history, _, replacement_pcts = run_episode(
            env=env,
            controller=controller,
            control_steps=CONTROL_STEPS_PER_EPISODE,
            sim_steps_per_control=SIM_STEPS_PER_CONTROL,
            is_training=True,
        )

        env.close()

        states = torch.cat(trajectory["states"])
        actions = torch.cat(trajectory["actions"])
        log_probs = torch.cat(trajectory["log_probs"])
        values = trajectory["values"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]

        with torch.no_grad():
            _, next_value = agent(states[-1].unsqueeze(0))

        returns, advantages = compute_gae(rewards, values, next_value.item(), dones)
        ppo_update(agent, optimizer, states, actions, log_probs, returns, advantages)

        total_reward = sum(rewards)
        all_scores.append(total_reward)
        cumulative_steps += len(rewards)

        history_steps.append(cumulative_steps)
        history_lengths.append(len(rewards))
        history_tts.append(history["tts_total"])
        history_lb.append(history["lower_bounds"])
        history_replacement.append(replacement_pcts)

        update_live_plot(all_scores, line, ax, fig)

        max_queues = {
            i: max(history["queues"][i]) if history["queues"][i] else 0
            for i in range(1, NUM_RAMPS + 1)
        }

        print(
            f"Episode {episode} | Reward: {total_reward:.2f} | "
            f"Steps: {len(rewards)} | TTS: {history['tts_total']:.0f} | "
            f"Max Queues: {max_queues}"
        )

        if episode % 10 == 0:
            torch.save(
                agent.state_dict(),
                os.path.join(MODELS_DIR, f"model_{file_prefix}_ep{episode}.pth"),
            )
            with open(os.path.join(MODELS_DIR, f"state_tracker_{file_prefix}_ep{episode}.pkl"), "wb") as f:
                pickle.dump(state_tracker, f)

    torch.save(agent.state_dict(), os.path.join(MODELS_DIR, f"model_{file_prefix}.pth"))
    with open(os.path.join(MODELS_DIR, f"state_tracker_{file_prefix}.pkl"), "wb") as f:
        pickle.dump(state_tracker, f)

    with open(os.path.join(MODELS_DIR, f"training_history_{file_prefix}_seed{seed}.pkl"), "wb") as f:
        pickle.dump({
            "scores": all_scores, "steps": history_steps,
            "lengths": history_lengths, "tts": history_tts,
            "replacement_pct": history_replacement, "lower_bounds": history_lb,
        }, f)

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_replacement", action="store_true")
    parser.add_argument("--dynamic_norm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args.use_replacement, args.seed, args.num_episodes, args.dynamic_norm)
