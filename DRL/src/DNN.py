import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.feature_size, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16+1, config.num_actions)
        self.relu = nn.LeakyReLU()

    def forward(self, features, prev_action):
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))

        x = torch.cat((x, prev_action), dim=1)
        x = self.fc3(x)
        return x
