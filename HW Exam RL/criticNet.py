import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, fc3_dims=256, fc4_dims=256,
            name='critic', lr= 1e-3):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.name = name
        self.lr= lr
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3= nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.q = nn.Linear(self.fc4_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) #lr = beta

    def zerog(self):
        self.optimizer.zero_grad()

    def stepg(self):
        self.optimizer.step()

    def freezeParameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreezeParameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc3(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc4(action_value)

        q = self.q(action_value)
        return q

