import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions.normal import Normal

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac', lr= 1e-3):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.lr = lr

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def zerog(self):
        self.optimizer.zero_grad()

    def stepg(self):
        self.optimizer.step()

    def freezeParameters(self):
        for param in self.parameters():
            param.requires_grad =False

    def unfreezeParameters(self):
        for param in self.parameters():
            param.requires_grad =True

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action)

        log_probs = probabilities.log_prob(actions).sum(axis=-1)  # used the reshape the outoput from -1 to 1

        val = (2*(np.log(2) - actions - F.softplus(-2*action)))
        #print("val  "+str(val))
        log_probs -= val.sum(axis=-1) #T.log(1-action.pow(2)+self.reparam_noise)  #axis=1

        return action, log_probs

