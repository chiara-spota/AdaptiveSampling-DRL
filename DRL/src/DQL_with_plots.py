
import os
import time
import random
import copy
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Agent import Agent
from Config import Config
from Environment import Environment
from EpsilonGreedyStrategy import Epsilon_Greedy_Exploration


# ---------------- Loss helpers ----------------
def huber_loss(x):
    return torch.where(torch.abs(x) < 1.0, 0.5 * x ** 2, torch.abs(x) - 0.5)

def mse_loss(x):
    return x ** 2


def plot_simple(x, fig_name, y_label):
    plt.figure(figsize=(20, 5))
    plt.plot(x)
    plt.ylabel(y_label)
    plt.title(fig_name)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()


# ---------------- Plot helpers ----------------

def plot_learning_curve_steps(
    step_reward_history,
    out_path,
    window=2000,        
    linewidth=1.2,       
):
    """
    Plots a step-level learning curve using a moving average over `window` samples.
    - No epsilon overlay
    - No grid
    - Thinner line
    """

    if len(step_reward_history) == 0:
        return

    steps = np.array([s for s, _ in step_reward_history], dtype=np.int64)
    r = np.array([v for _, v in step_reward_history], dtype=np.float64)

    # moving average smoothing
    N = min(window, len(r))
    if N <= 1:
        steps_smooth, r_smooth = steps, r
    else:
        kernel = np.ones(N, dtype=np.float64) / N
        r_smooth = np.convolve(r, kernel, mode="valid")
        steps_smooth = steps[N - 1 :]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps_smooth, r_smooth, linewidth=linewidth)

    ax.set_xlabel("Training steps (ECG segments)")
    ax.set_ylabel("Average Reward")
    ax.set_title("Learning Dynamics of DDQN Agent")

    ax.grid(False)                 # <- remove grid
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close()



# ---------------- Main training script ----------------
config_file_path = "configs/DQL.yaml"
config = Config(config_file_path)

environment = Environment(config, "train")
environment.case = "DRL"

exploration_strategy = Epsilon_Greedy_Exploration(config)
agent = Agent(config, exploration_strategy)

agent.policy_net.to(config.device)
agent.target_net.to(config.device)
agent.target_net.eval()

action_to_decimation = {0: "8", 1: "4", 2: "2", 3: "1"}

total_time = 0.0
avg_reward_list = []
trial_mean_rewards = []
trial_std_rewards = []
eps_history = []

# step-level histories 
global_step = 0
step_reward_history = []  # list of (global_step, reward)
step_eps_history = []     # list of (global_step, epsilon)

losses = []

if config.use_PER:
    config_string = f"{config.name}_PER_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}"
else:
    config_string = f"{config.name}_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}"

print(f"name: {config.name}")
print(f"device: {config.device}")
print(f"num_trials: {config.num_trials}")
print(f"num_subjects_train: {config.num_subjects_train}")
print(f"replay_batch_size: {config.replay_batch_size}")
print(f"epsilon_decay_rate: {config.epsilon_decay_rate}")
print(f"max_epsilon: {config.max_epsilon}")
print(f"min_epsilon: {config.min_epsilon}")
print(f"learning_rate_min: {config.learning_rate_min}")
print(f"learning_rate_max: {config.learning_rate_max}")
print(f"gamma: {config.gamma}")
print(f"hard_update: {config.hard_update}")
print(f"target_network_update_rate: {config.target_network_hard_update_rate}")
print(f"tau: {config.tau}")
print(f"replay_memory_size: {config.replay_memory_size}")

torch.set_printoptions(profile="full")

optimal_actions = []
eval_actions = []

start_time = time.time()

for trial in range(config.num_trials + 1):
    print(f"Trial {trial + 1}")

    environment = Environment(config, "train")
    environment.reset()
    environment.case = "DRL"

    prev_action = 3


    if trial == config.num_trials:
        agent.epsilon = 0.0

    eps_history.append(float(agent.epsilon))

    for subject in range(config.num_subjects_train):
        environment.done = 0
        environment.subject_index = subject

        current_state = environment.get_state(False)
        per_subject_reward = []
        counter = 0

        while environment.done == 0:
            if trial == 0:
                optimal_actions.append(np.argmax(environment.data_train[subject][2][counter]))

            action = agent.select_action(current_state, prev_action)

            if trial == config.num_trials:
                eval_actions.append(action)

            reward, next_state = environment.step(action)
            per_subject_reward.append(reward)

            step_reward_history.append((global_step, float(reward)))
            step_eps_history.append((global_step, float(agent.epsilon)))
            global_step += 1


            if config.use_PER:
                current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(config.device)
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(config.device)
                prev_action_tensor = torch.tensor([[prev_action]], dtype=torch.float, device=config.device)
                done = environment.done

                with torch.no_grad():
                    max_action = agent.policy_net(next_state_tensor, prev_action_tensor).detach().argmax(dim=1).unsqueeze(1)
                    next_q_value = agent.target_net(next_state_tensor, prev_action_tensor).gather(dim=1, index=max_action).squeeze(1)
                    target_q_value = reward + (1 - done) * (config.gamma * next_q_value)
                    current_q_value = agent.policy_net(current_state_tensor, prev_action_tensor)[0][action]

                error = torch.abs(current_q_value - target_q_value).item()
                agent.replay_buffer.add_experience(
                    current_state, prev_action, action, reward, next_state, error, int(environment.done == 1)
                )
            else:
                agent.replay_buffer.add_experience(
                    current_state, prev_action, action, reward, next_state, int(environment.done == 1)
                )

            # --------- sample + DDQN update ----------
            if len(agent.replay_buffer.memory) >= config.batch_delay * config.replay_batch_size:
                if config.use_PER:
                    current_states, prev_actions, actions, rewards, next_states, dones, sampled_idxs, is_weights = \
                        agent.replay_buffer.sample(config.replay_batch_size)
                else:
                    current_states, prev_actions, actions, rewards, next_states, dones = \
                        agent.replay_buffer.sample(config.replay_batch_size)

                # DDQN target: online selects, target evaluates
                with torch.no_grad():
                    max_actions = agent.policy_net(next_states, actions).detach().argmax(dim=1).unsqueeze(1)
                    next_q_values = agent.target_net(next_states, actions).gather(dim=1, index=max_actions).squeeze(1)
                    target_q_values = rewards.squeeze(1) + np.multiply(
                        (1 - dones).squeeze(1),
                        (config.gamma * next_q_values),
                    )

                current_q_values = agent.policy_net(current_states, prev_actions).gather(dim=1, index=actions).squeeze(1)

                if config.use_PER:
                    errors = torch.abs(current_q_values - target_q_values)
                    for j in range(len(sampled_idxs)):
                        agent.replay_buffer.update(sampled_idxs[j], errors[j].item())
                    loss = torch.sum(mse_loss(errors) * torch.FloatTensor(is_weights))
                else:
                    loss = nn.MSELoss()(current_q_values, target_q_values)

                losses.append(float(loss.item()))

                agent.optimizer.zero_grad()
                loss.backward()

                if config.gradient_clip:
                    for param in agent.policy_net.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-1, 1)

                agent.optimizer.step()

                # target net update
                if config.hard_update:
                    if (counter + 1) % config.target_network_hard_update_rate == 0:
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())
                else:
                    for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
                        target_param.data.copy_(config.tau * policy_param.data + (1 - config.tau) * target_param.data)

            prev_action = action
            current_state = next_state
            counter += 1

        # per-subject mean reward 
        avg_reward_list.append(sum(per_subject_reward) / environment.num_data_segments_train)

  
    n = config.num_subjects_train
    this_trial_rewards = np.array(avg_reward_list[-n:], dtype=np.float64)  # per-subject rewards this trial
    this_trial_mean = float(this_trial_rewards.mean())
    this_trial_std = float(this_trial_rewards.std())

    trial_mean_rewards.append(this_trial_mean)
    trial_std_rewards.append(this_trial_std)

    print(f"trial_mean_reward: {this_trial_mean:.4f}  (std across subjects: {this_trial_std:.4f})")


    if trial < config.num_trials:
        agent.epsilon = agent.epsilon_decay(trial + 1, config.num_trials)

end_time = time.time()
total_time += (end_time - start_time)

# ---------------- Save model ----------------
model_save_path = config.policy_model_save_path
file_name = config_string.replace(".", "_") + ".pkl"
full_path = model_save_path + file_name

os.makedirs(model_save_path, exist_ok=True)
torch.save({"model_state_dict": agent.policy_net.state_dict()}, full_path)

print(f"total_time: {total_time}, average_trial_time: {total_time / max(config.num_trials, 1)}")

# ---------------- Diagnostics ----------------
optimal_action_count = len(optimal_actions)
optimal_action_counts = Counter(optimal_actions)
print(f"optimal_action_counts: {optimal_action_counts}")
print(f"optimal_action_ratios: {np.round(np.asarray(list(optimal_action_counts.values()))/max(optimal_action_count,1)*100, 2)}")

eval_action_count = len(eval_actions)
eval_action_counts = Counter(eval_actions)
print(f"eval_action_counts: {eval_action_counts}")
print(f"eval_action_ratios: {np.round(np.asarray(list(eval_action_counts.values()))/max(eval_action_count,1)*100, 2)}")

action_accuracy = 0
for idx in range(min(optimal_action_count, eval_action_count)):
    if eval_actions[idx] == optimal_actions[idx]:
        action_accuracy += 1
action_accuracy = round(action_accuracy / max(optimal_action_count, 1) * 100, 2)
print(f"action_accuracy: {action_accuracy}")


# ---------------- Plots ----------------
os.makedirs("results/train/", exist_ok=True)
base = full_path.split("/")[-1].split(".")[0]


plot_simple(avg_reward_list, f"results/train/{base}_rewards.png", "reward")
plot_simple(losses, f"results/train/{base}_losses.png", "loss")


plot_learning_curve_steps(
    step_reward_history=step_reward_history,
    out_path=f"results/train/{base}_learning_curve_steps.png",
    window=500, 
)

print("Saved plots:")
print(f"  results/train/{base}_learning_curve_trial_pretty.png")
print(f"  results/train/{base}_learning_curve_steps.png")
print(f"  results/train/{base}_rewards.png")
print(f"  results/train/{base}_losses.png")
