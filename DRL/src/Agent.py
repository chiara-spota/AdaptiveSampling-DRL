import random
import torch
from Replay_Buffer import Replay_Buffer
from Prioritized_Replay_Buffer import Prioritized_Replay_Buffer
from DNN import DNN

import numpy as np

class Agent():
    def __init__(self, config, exploration_strategy):
        super(Agent, self).__init__()
        self.config = config
        self.exploration_strategy = exploration_strategy
        self.policy_net = DNN(self.config)
        self.target_net = DNN(self.config)
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(policy_param)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.config.learning_rate_max)
        if self.config.use_PER:
            self.replay_buffer = Prioritized_Replay_Buffer(self.config)
        else:
            self.replay_buffer = Replay_Buffer(self.config)
        self.epsilon = self.config.max_epsilon
        self.status = 0

    def epsilon_decay(self, current_episode, num_episodes):
        # epsilon annealing
        return self.exploration_strategy.decay(current_episode, num_episodes)

    def select_action(self, current_state, prev_action):
        random_val = random.random()
        # exploration
        if self.epsilon > random_val:
            action = random.randrange(self.config.num_actions)
            return action
        # exploitation
        else:
            self.policy_net.eval()
            with torch.no_grad():
                features = torch.tensor([current_state], dtype=torch.float, device=self.config.device)
                prev_action = torch.tensor([[prev_action]], dtype=torch.float, device=self.config.device)
                action_values = self.policy_net(features, prev_action)
                action = torch.argmax(action_values)
            self.policy_net.train()
            return action.item()
