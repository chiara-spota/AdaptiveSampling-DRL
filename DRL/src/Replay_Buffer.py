from collections import namedtuple, deque
import random
import torch
import numpy as np

class Replay_Buffer():
    def __init__(self, config):
        self.config = config
        self.memory = deque(maxlen=self.config.replay_memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "prev_action", "action", "reward", "next_state", "done"])

    def add_experience(self, state, prev_action, action, reward, next_state, done):
        experience = self.experience(state, prev_action, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            current_states, prev_actions, actions, rewards, next_states, done = self.separate_out_data_types(experiences)
            return current_states, prev_actions, actions, rewards, next_states, done
        else:
            return experiences
            
    def separate_out_data_types(self, experiences):
        current_states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.config.device)
        prev_actions = torch.from_numpy(np.vstack([e.prev_action for e in experiences if e is not None])).float().to(self.config.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.config.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.config.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.config.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.config.device)
        return current_states, prev_actions, actions, rewards, next_states, dones
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else: batch_size = self.config.replay_batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
