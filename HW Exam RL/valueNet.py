import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

""" Value network, that take a state and return it's estimated value"""
class ValueNetwork(nn.Module):
    # number of action no needed here, cause this network evaluates only the state value
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', lr= 1e-3):

        # definition for the structure of the network
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # start function for the gradient

    def zerog(self):
        self.optimizer.zero_grad()

    def freezeParameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreezeParameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def stepg(self):
        self.optimizer.step()

    # end function for the gradient

    # forward step to compute the value
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v

