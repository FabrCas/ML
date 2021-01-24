import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac', lr= 1e-3):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.lr= lr
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) #lr = beta
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

       # self.to(self.device)
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

        q = self.q(action_value)

        return q

