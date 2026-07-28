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
    MODELS_DIR, RAMP_DETS, STATE_DIM, SIMULATION_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)
from env import RampMeterEnv
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

    sumo_cmd = ["sumo", "-c", SIMULATION_PATH, "--no-step-log", "true", "--no-warnings", "--seed", str(seed)]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd,
        tls_id=TLS_ID,
        upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS,
        ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS,
        ramp_detector=RAMP_DETS
    )

    agent = SharedActorCritic(STATE_DIM)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    state_tracker = RunningStat(shape=(STATE_DIM,))

    normalize_fnc = normalize_static
    if dynamic_norm:
        normalize_fnc = normalize_dynamic

    controller = RLController(
        agent=agent,
        state_tracker=state_tracker,
        normalize_fnc=normalize_fnc,
        use_replacement=use_replacement
    )

    line, ax, fig = init_plot(use_replacement)
    all_scores, history_steps, history_lengths, history_tts = [], [], [], []
    history_actions, history_replacement, history_lb = [], [], []
    reward_history = {}
    cumulative_steps = 0
    total_expected_steps = num_episodes * CONTROL_STEPS_PER_EPISODE

    for episode in range(1, num_episodes+1):
        env.start()

        # Execute the episode using the runner
        trajectory, history, _, replacement_pct = run_episode(
            env=env,
            controller=controller,
            control_steps=CONTROL_STEPS_PER_EPISODE,
            sim_steps_per_control=SIM_STEPS_PER_CONTROL,
            is_training=True
        )

        env.close()

        # Extract trajectory for PPO update
        states = torch.cat(trajectory["states"])
        actions = torch.cat(trajectory["actions"])
        log_probs = torch.cat(trajectory["log_probs"])
        values = trajectory["values"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]

        # Calculate advantages and update policy
        frac = max(0.0, 1.0 - (cumulative_steps / total_expected_steps))
        current_lr = 3e-4 * frac
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Calculate advantages and update policy
        with torch.no_grad():
            _, next_value = agent(states[-1].unsqueeze(0))

        returns, advantages = compute_gae(rewards, values, next_value.item(), dones)
        ppo_update(agent, optimizer, states, actions, log_probs, returns, advantages)

        # Logging
        total_reward = sum(rewards)
        all_scores.append(total_reward)
        cumulative_steps += len(rewards)

        history_steps.append(cumulative_steps)
        history_lengths.append(len(rewards))
        history_tts.append(history["tts_total"])
        history_actions.append(history["green_times"])
        history_lb.append(history["lower_bound"])
        history_replacement.append(replacement_pct)

        update_live_plot(all_scores, line, ax, fig)
        action_avg = np.mean(history["green_times"])
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Steps: {len(rewards)} | TTS: {history['tts_total']:.0f} | Max Queue: {max(history['queues'])} | Replacement %: {replacement_pct:.2f} | Action Avg: {action_avg:.2f}")

        # Save checkpoints
        if episode % 100 == 0:
            reward_history[episode] = rewards.copy()

            fig_reward, ax_reward = plt.subplots()

            ax_reward.plot(np.arange(len(rewards)), rewards, label=f"Episode {episode}")

            ax_reward.legend()
            ax_reward.axhline(0, linestyle="--", color="r", alpha=0.5)
            ax_reward.set_xlabel("Step")
            ax_reward.set_ylabel("Reward")
            ax_reward.set_title("Rewards Plot")
            os.makedirs("plots", exist_ok=True)
            fig_reward.savefig(os.path.join("plots", f"rewards_episode{episode}.png"))
            plt.close(fig_reward)

            torch.save(agent.state_dict(), os.path.join(MODELS_DIR, f"model_{file_prefix}_seed{seed}_ep{episode}.pth"))
            with open(os.path.join(MODELS_DIR, f"state_tracker_{file_prefix}_seed{seed}_ep{episode}.pkl"), "wb") as f:
                pickle.dump(state_tracker, f)

    # Save seed-specific artifacts so parallel multi-seed runs do not overwrite each other.
    torch.save(agent.state_dict(), os.path.join(MODELS_DIR, f"model_{file_prefix}_seed{seed}.pth"))
    with open(os.path.join(MODELS_DIR, f"state_tracker_{file_prefix}_seed{seed}.pkl"), "wb") as f:
        pickle.dump(state_tracker, f)
    if seed == 42:
        torch.save(agent.state_dict(), os.path.join(MODELS_DIR, f"model_{file_prefix}.pth"))
        with open(os.path.join(MODELS_DIR, f"state_tracker_{file_prefix}.pkl"), "wb") as f:
            pickle.dump(state_tracker, f)

    # Save final metrics
    with open(os.path.join(MODELS_DIR, f"training_history_{file_prefix}_seed{seed}.pkl"), "wb") as f:
        pickle.dump({
            "scores": all_scores, "steps": history_steps,
            "lengths": history_lengths, "tts": history_tts,
            "replacement_pct": history_replacement, "lower_bounds": history_lb
        }, f)

    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_replacement", action="store_true")
    parser.add_argument("--dynamic_norm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=500)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args.use_replacement, args.seed, args.num_episodes, args.dynamic_norm)
