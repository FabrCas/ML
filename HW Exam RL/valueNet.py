import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac', lr= 1e-3):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
       # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

       # self.to(self.device)

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

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
