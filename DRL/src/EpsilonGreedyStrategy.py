import random
import torch
import numpy as np

class Epsilon_Greedy_Exploration():
    def __init__(self, config):
        self.config = config

    def decay(self, current_episode, num_episodes_to_run):
        return self.config.min_epsilon + (self.config.max_epsilon - self.config.min_epsilon)\
               * max((num_episodes_to_run - current_episode * self.config.epsilon_decay_rate) / num_episodes_to_run, 0)