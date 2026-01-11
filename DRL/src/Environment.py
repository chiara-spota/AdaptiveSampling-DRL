import random
import torch
import pandas as pd
from glob import glob
import numpy as np

import pickle
import pickle5


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from scipy.stats import rv_discrete

from collections import Counter

class Environment:
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

        self.decimation_reward = {0: 8, 1: 4, 2: 2, 3: 1}
        self.reverse_decimation_reward = {0: -1, 1: -2, 2: -4, 3: -8}
        self.norm_reward = {8: 1.0, 4: 0.5, 2: 0.0, 1: -0.5, -10: -1}

        self.data_train = None
        self.data_eval = None

        self.sample_state_train_index = 0
        self.sample_metric_train_index = 0

        self.sample_state_eval_index = 0
        self.sample_metric_eval_index = 0

        self.num_episodes_eval = 0

        self.load_data()

        self.reward = 0
        self.normalized_reward = 0
        self.standardized_reward = 0
        self.case = 'DRL'

        self.subject_index = 0
        self.done = 0
        self.num_data_segments_train = 0
        self.num_metric_segments_train = 0

    def reset(self):
        self.sample_state_train_index = 0
        self.sample_metric_train_index = 0

        self.sample_state_eval_index = 0
        self.sample_metric_eval_index = 0

        self.reward = 0
        self.normalized_reward = 0
        self.standardized_reward = 0

        self.subject_index = 0
        self.done = 0
        self.num_data_segments_train = 0
        self.num_metric_segments_train = 0

    def list_to_dict(self, list):
        it = iter(a)
        res_dct = dict(zip(it, it))
        return res_dct

    def load_data(self):
        if '.pkl' in self.config.data_train_path or '.pickle' in self.config.data_train_path:
            with open(self.config.data_train_path, 'rb') as file:
                self.data_train = pickle5.load(file)
        elif '.npy' in self.config.data_train_path or '.npz' in self.config.data_train_path:
            self.data_train = np.load(self.config.data_train_path, allow_pickle=True)
        else:
            raise Exception(f'loading for file type {self.config.data_train_path.split(".")[-1]} not implemented')
        if '.pkl' in self.config.data_eval_path or '.pickle' in self.config.data_eval_path:
            with open(self.config.data_eval_path, 'rb') as file2:
                self.data_eval = pickle5.load(file2)
        elif '.npy' in self.config.data_eval_path or '.npz' in self.config.data_eval_path:
            self.data_eval = np.load(self.config.data_eval_path, allow_pickle=True)
        else:
            raise Exception(f'loading for file type {self.config.data_eval_path.split(".")[-1]} not implemented')

        if not self.config.override_num_subjects:
            self.config.num_subjects_train = len(self.data_train)
        self.num_subjects_eval = len(self.data_eval)

    def sample_state_train(self, increment=True):
        if increment:
            self.sample_state_train_index += 1
            if self.sample_state_train_index >= self.num_data_segments_train:
                sample_state_train = self.data_train[self.subject_index][0][self.sample_state_train_index-1]
                self.sample_state_train_index = 0
                self.done = 1
                return sample_state_train
        data_segments = self.data_train[self.subject_index][0]
        self.num_data_segments_train = len(data_segments)
        sample_state_train = data_segments[self.sample_state_train_index]

        return sample_state_train

    def sample_state_eval(self, increment=True):
        if increment:
            self.sample_state_eval_index += 1
            if self.sample_state_eval_index >= self.num_data_segments_eval:
                sample_state_eval = self.data_eval[self.subject_index][0][self.sample_state_eval_index-1]
                self.sample_state_eval_index = 0
                self.done = 1
                return sample_state_eval
        data_segments = self.data_eval[self.subject_index][0]
        self.num_data_segments_eval = len(data_segments)
        sample_state_eval = data_segments[self.sample_state_eval_index]
        return sample_state_eval

    def sample_metric_train(self, increment=True):
        if increment:
            self.sample_metric_train_index += 1
            if self.sample_metric_train_index >= self.num_metric_segments_train:
                sample_metric_train = self.data_train[self.subject_index][2][self.sample_metric_train_index-1]
                self.sample_metric_train_index = 0
                return sample_metric_train
        metric_segments = self.data_train[self.subject_index][2]
        self.num_metric_segments_train = len(metric_segments)
        sample_metric_train = metric_segments[self.sample_state_train_index]
        return sample_metric_train

    def sample_metric_eval(self, increment=True):
        if increment:
            self.sample_metric_eval_index += 1
        sample_metric_eval = self.data_eval[self.sample_metric_eval_index]
        return sample_metric_eval

    def get_state(self, increment=True):
        if self.mode == 'train':
            return self.sample_state_train(increment)
        elif self.mode == 'eval':
            return self.sample_state_eval(increment)

    def get_metric(self, increment=True):
        if self.mode == 'train':
            return self.sample_metric_train(increment)
        elif self.mode == 'eval':
            return self.sample_metric_eval(increment)

    def normalize_reward(self, energy):
        normalized_energy_reward = -(energy - self.min_energy) / (self.max_energy - self.min_energy)
        return normalized_energy_reward

    def standardize_reward(self, energy):
        standardized_energy_reward = -(energy - self.mean_energy) / self.standard_deviation_energy
        return standardized_energy_reward

    def get_reward(self, action):
        if self.mode == 'train':
            self.reward = self.norm_reward[self.data_train[self.subject_index][2][self.sample_state_train_index][action]]
        elif self.mode == 'eval':
            pass
        return self.reward

    def calculate_reward(self, metric, action):
        self.reward = 0
        if metric[action] > 0:
            self.reward = self.config.reward_lambda * self.decimation_reward[action]
        else:
            self.reward = self.reverse_decimation_reward[action]
        return self.reward

    def step(self, action):
        if self.mode == 'train':
            reward = self.get_reward(action)
            next_state = self.get_state()
            return reward, next_state
        elif self.mode == 'eval':
            next_state = self.get_state()
            return None, next_state