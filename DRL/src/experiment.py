import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from Replay_Buffer import Replay_Buffer
from Agent import Agent
from Config import Config
from Environment import Environment
from EpsilonGreedyStrategy import Epsilon_Greedy_Exploration
import time
import csv
import matplotlib
from matplotlib.cm import Blues

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import copy
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from matplotlib.collections import LineCollection


config_file_path = "configs/DQL.yaml"

config = Config(config_file_path)

environment = Environment(config, 'eval')

action_to_decimation = {0: '8', 1: '4', 2: '2', 3: '1'}

print(f'name: {config.name}')
print(f'device: {config.device}')
print(f'num_trials: {config.num_trials}')
print(f'num_subjects_train: {config.num_subjects_train}')
print(f'num_subjects_eval: {environment.num_subjects_eval}')
print(f'replay_batch_size: {config.replay_batch_size}')
print(f'epsilon_decay_rate: {config.epsilon_decay_rate}')
print(f'max_epsilon: {config.max_epsilon}')
print(f'min_epsilon: {config.min_epsilon}')
print(f'learning_rate_min: {config.learning_rate_min}')
print(f'learning_rate_max: {config.learning_rate_max}')
print(f'gamma: {config.gamma}')
print(f'hard_update: {config.hard_update}')
print(f'target_network_update_rate: {config.target_network_hard_update_rate}')
print(f'tau: {config.tau}')
print(f'replay_memory_size: {config.replay_memory_size}')

exploration_strategy = Epsilon_Greedy_Exploration(config)

agent = Agent(config, exploration_strategy)

if config.use_PER:
    config_string = f'{config.name}_PER_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}'
else:
    config_string = f'{config.name}_lr{config.learning_rate_max}_bs{config.replay_batch_size}_trial{config.num_trials}'

model_save_path = config.policy_model_save_path + config_string.replace('.', '_') + '.pkl'
model_name = model_save_path.split('/')[-1].split('.')[0]

if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    agent = Agent(config, exploration_strategy)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    print(f'model path {model_save_path}')
    print(f'DRL model {model_name} loaded')
else:
    print(f'DRL model path {model_save_path} not found')
    exit(-1)

agent.policy_net.to(config.device)
agent.policy_net.eval()

policy_actions = []
patient_wise_policy_actions = {}
patient_wise_q_values = {}  


optimal_actions = []
patient_wise_optimal_actions = []



total_time = 0
start_time = time.time()


for subject in range(environment.num_subjects_eval):
    if len(environment.data_eval[subject][0]) == 0:
        print(f"Skipping subject {subject} (0 segments)")
        continue
    environment.done = 0
    environment.subject_index = subject
    current_state = environment.get_state(False)
    prev_action = 3
    patient_wise_list = []
    patient_wise_q_list = []
    counter = 0
    while environment.done == 0:
        optimal_actions.append(np.argmax(environment.data_eval[subject][2][counter]))
        with torch.no_grad():
            current_state = torch.tensor([current_state], dtype=torch.float, device=config.device)
            prev_action = torch.tensor([[prev_action]], dtype=torch.float, device=config.device)
            action_values = agent.policy_net(current_state, prev_action)
            # log raw Q-values (convert to 1D numpy array length 4)
            q_vec = action_values.squeeze(0).detach().cpu().numpy()  # (4,)

            best_action = torch.argmax(action_values)
            _, next_state = environment.step(best_action)
        policy_actions.append(best_action.item())
        patient_wise_list.append(best_action.item())
        patient_wise_q_list.append(q_vec)
        prev_action = best_action
        current_state = next_state
        counter += 1
    patient_wise_policy_actions[subject] = patient_wise_list
    patient_wise_q_values[subject] = np.stack(patient_wise_q_list, axis=0)
end_time = time.time()
elapsed_time = end_time - start_time
total_time += elapsed_time

action_count = len(policy_actions)

print(f'total_time: {total_time}, average_step_time: {total_time / action_count}')

optimal_action_count = len(optimal_actions)
optimal_action_counts = Counter(optimal_actions)
print(f'optimal_action_counts: {optimal_action_counts}')
print(f'optimal_action_ratios: {np.round(np.asarray(list(optimal_action_counts.values()))/optimal_action_count*100, 2)}')

action_counts = Counter(policy_actions)
print(f'action_counts: {action_counts}')
print(f'action_ratios: {np.round(np.asarray(list(action_counts.values()))/action_count*100, 2)}')
print(f'policy_actions length: {action_count}')


action_accuracy = 0
for idx in range(optimal_action_count):
    if policy_actions[idx] == optimal_actions[idx]:
        action_accuracy += 1
action_accuracy = round(action_accuracy/optimal_action_count * 100, 2)
print(f'action_accuracy: {action_accuracy}')

import os
os.makedirs("results/eval", exist_ok=True)

with open(f'results/eval/{model_name}_action_trajectory.npy', 'wb') as f:
    np.save(f, policy_actions)

with open(f'results/eval/{model_name}_patient_wise_action_trajectory.npy', 'wb') as f:
    np.save(f, patient_wise_policy_actions)

print("EVAL")

# ---------- ACTION DISTRIBUTION (PERCENT) ----------

actions = np.array(policy_actions)


display_order = [3, 2, 1, 0]
display_labels = ["×1", "×2", "×4", "×8"]

# Count actions
counts = np.array([np.sum(actions == a) for a in display_order])
percentages = counts / counts.sum() * 100.0

# Blue gradient: light (×1) → dark (×8)
colors = Blues(np.linspace(0.35, 0.85, 4))

plt.figure(figsize=(6, 4))
bars = plt.bar(display_labels, percentages, color=colors, edgecolor="none")

plt.xlabel("Downsampling Action (Decimation Factor)")
plt.ylabel("Percentage of Actions (%)")
plt.title("Policy Downsampling Action Distribution")

plt.ylim(0, 100)


plt.tight_layout()
plt.savefig(f"results/eval/{model_name}_action_distribution_percent.png", dpi=500)
plt.close()

# ---------- PATIENT-WISE HEATMAP  ----------

T = 50  # first 50 timesteps (segments)


keys = sorted([k for k, seq in patient_wise_policy_actions.items() if len(seq) >= T])

heatmap = np.zeros((len(keys), T), dtype=float)

for row_idx, subject_id in enumerate(keys):
    seq = patient_wise_policy_actions[subject_id][:T]
    heatmap[row_idx, :] = seq

cmap = ListedColormap(["#08306B", "#2171B5", "#6BAED6", "#C6DBEF"])

norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

plt.figure(figsize=(12, 6))
im = plt.imshow(
    heatmap,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    norm=norm
)

cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(["×8", "×4", "×2", "×1"])
cbar.set_label("Downsampling Action")

Fs = 360  # MIT-BIH sampling rate
segment_duration = 256 / Fs
time_axis = np.arange(T) * segment_duration

plt.xticks(
    ticks=np.linspace(0, T-1, 5),
    labels=np.round(np.linspace(0, (T-1)*segment_duration, 5), 1)
)
plt.xlabel("Time (Seconds)")

plt.ylabel("Patient Index")
plt.title(f"Adaptive Sampling Policy over Time")

plt.tight_layout()
plt.savefig(f"results/eval/{model_name}_policy_heatmap_T{T}_filtered.png", dpi=500)
plt.close()



# ---------------- ECG plot with RL-selected sampling rate (ZOOMED) ----------------

patient_id = 18  


X_patient = environment.data_eval[patient_id][0]            
policy_actions = patient_wise_policy_actions[patient_id]      

SEG_LEN = 256  


ecg = X_patient.flatten()

action_per_sample = np.repeat(policy_actions, SEG_LEN)

min_len = min(len(ecg), len(action_per_sample))
ecg = ecg[:min_len]
action_per_sample = action_per_sample[:min_len]

start_sample = 0
end_sample = 3500

start_sample = max(0, start_sample)
end_sample = min(len(ecg), end_sample)

ecg_zoom = ecg[start_sample:end_sample]
action_zoom = action_per_sample[start_sample:end_sample]

x = np.arange(start_sample, end_sample)

points = np.array([x, ecg_zoom]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

blue_colors = [
    "#08306b", 
    "#1a6db5",
    "#6baed6",  
    "#9cbede",  
]

cmap = ListedColormap(blue_colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(action_zoom[:-1])      
lc.set_linewidth(1.8)


fig, ax = plt.subplots(figsize=(14, 4))
ax.add_collection(lc)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(ecg_zoom.min() * 1.1, ecg_zoom.max() * 1.1)

ax.set_xlabel("Time (Samples)")
ax.set_ylabel("ECG Normalized Amplitude")
ax.set_title(f"ECG with DDQN-controlled Adaptive Sampling Policy")

cbar = plt.colorbar(lc, ax=ax, ticks=[0, 1, 2, 3])
cbar.set_label("Downsampling Action")
cbar.ax.set_yticklabels(["×8", "×4", "×2", "×1"])

plt.tight_layout()
plt.savefig(f"results/eval/{model_name}_ecg_policy_patient{patient_id}_{start_sample}_{end_sample}.png", dpi=500)
plt.close()

# ---------- ACTION DISTRIBUTION BY BEAT TYPE ----------


os.makedirs("results/eval", exist_ok=True)


action_labels = ["×8", "×4", "×2", "×1"]

beat_labels = ["N", "S", "V"]
num_beats = len(beat_labels)
num_actions = len(action_labels)

counts = np.zeros((num_beats, num_actions), dtype=np.int64)

for subject_id, actions_seq in patient_wise_policy_actions.items():
    if subject_id >= len(environment.data_eval):
        continue
    if len(actions_seq) == 0:
        continue

    y_seq = environment.data_eval[subject_id][1]  
    if len(y_seq) == 0:
        continue

    L = min(len(actions_seq), len(y_seq))
    actions_seq = actions_seq[:L]
    y_seq = y_seq[:L]

    for a, y in zip(actions_seq, y_seq):
        if 0 <= y < num_beats and 0 <= a < num_actions:
            counts[y, a] += 1

# --------- GROUPED COUNTS PLOT ----------
x = np.arange(num_beats)
bar_w = 0.18

plt.figure(figsize=(9, 5))
for a in range(num_actions):
    plt.bar(x + (a - (num_actions-1)/2)*bar_w, counts[:, a], width=bar_w, label=action_labels[a])

plt.xticks(x, beat_labels)
plt.xlabel("Beat type")
plt.ylabel("Count")
plt.title("Policy action distribution by beat type (counts)")
plt.legend(title="Sampling action")
plt.tight_layout()
plt.savefig(f"results/eval/{model_name}_action_dist_by_beat_counts.png", dpi=200)
plt.close()


# --------- NORMALIZED (PERCENT) PLOT ----------


beat_labels = [
    "Normal",
    "Supraventricular Ectopic",
    "Ventricular Ectopic"
]

action_labels = ["×1", "×2", "×4", "×8"]

reorder_idx = [3, 2, 1, 0]   
counts_reordered = counts[:, reorder_idx]


row_sums = counts_reordered.sum(axis=1, keepdims=True)
percent = np.divide(counts_reordered, np.maximum(row_sums, 1)) * 100.0

num_beats = len(beat_labels)
num_actions = len(action_labels)
x = np.arange(num_beats)
bar_w = 0.18
colors = cm.Blues(np.linspace(0.35, 0.85, num_actions))

plt.figure(figsize=(9, 5))

for a in range(num_actions):
    plt.bar(
        x + (a - (num_actions - 1) / 2) * bar_w,
        percent[:, a],
        width=bar_w,
        label=action_labels[a],
        color=colors[a]
    )

plt.xticks(x, beat_labels)
plt.xlabel("Beat Type")
plt.ylabel("Percentage of Agent Actions (%)")
plt.title("Distribution of Policy Downsampling Actions by Beat Type")
plt.ylim(0, 100)
plt.legend(title="Downsampling factor")
plt.tight_layout()

plt.savefig(
    f"results/eval/{model_name}_action_dist_by_beat_percent.png",
    dpi=500
)
plt.close()

print(
    "Saved:",
    f"results/eval/{model_name}_action_dist_by_beat_percent.png"
)
# ---------------- Q-VALUE HEATMAP (ZOOMED) ----------------
patient_id = 18
SEG_LEN = 256

start_sample = 0
end_sample = 3000

Q_all = patient_wise_q_values[patient_id]

start_seg = start_sample // SEG_LEN
end_seg = int(np.ceil(end_sample / SEG_LEN))

start_seg = max(0, start_seg)
end_seg = min(Q_all.shape[0], end_seg)

Q = Q_all[start_seg:end_seg, :] 

reorder = [3, 2, 1, 0]
Q_disp = Q[:, reorder]          


H = Q_disp.T                  

x0 = start_seg * SEG_LEN
x1 = end_seg * SEG_LEN

x0 = max(x0, start_sample)
x1 = min(x1, end_sample)

plt.figure(figsize=(14, 3.8))
im = plt.imshow(
    H,
    aspect="auto",
    interpolation="nearest",
    extent=[start_seg * SEG_LEN, end_seg * SEG_LEN, 3.5, -0.5]  
)

cbar = plt.colorbar(im)
cbar.set_label("Q-value")

plt.yticks([0, 1, 2, 3], ["×1", "×2", "×4", "×8"])
plt.xlabel("Time (Samples)")
plt.ylabel("Action (Decimation Factor)")
plt.title(f"DDQN Q-values over Time")

plt.xlim(start_sample, end_sample)

plt.tight_layout()
plt.savefig(
    f"results/eval/{model_name}_qvalue_heatmap_patient{patient_id}_{start_sample}_{end_sample}.png",
    dpi=500
)
plt.close()

print(
    "Saved:",
    f"results/eval/{model_name}_qvalue_heatmap_patient{patient_id}_{start_sample}_{end_sample}.png"
)
