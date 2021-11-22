import torch as T
import torch.nn.functional as F
import torch.nn as nn


""" Q network, that take a state and an action and return it's estimated Q value"""
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, fc3_dims=256, fc4_dims=256,
            name='critic', lr= 1e-3):

        # definition for the structure of the network
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

    # forward step to compute the Q value
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

