import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions.normal import Normal

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions = None, name='actor', lr= 1e-3):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.lr = lr

        # definition of the layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # 2 different final layers for mu and sigma
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state):
        out = self.fc1(state)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        mu = self.mu(out)
        sigma = self.sigma(out)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)  # clamp sigma into the the interval
        # sigma = T.exp(sigma)
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

    def sample_normal(self, state):
        #print(state.shape[0])
        # evaluate sigma and mu with the network
        mu, sigma = self.forward(state)
        #print("mu " + str(mu))
        #print("sigma " + str(sigma))

        # define the normal
        probabilities = Normal(mu, sigma)
        #if (state.shape[0] == 13):
        #    actions = mu
        #else:
            # take the probabilities in a non deterministic matter (batch size)
        actions = probabilities.rsample()

        # hyperbolic tangent to limit between 1 and -1, and normalization, multiplying with the action tensor
        action = T.tanh(actions) * T.tensor(self.max_action)
        # print("actions " + str(actions))

      # if not (state.shape[0] == 6):   # 13 or 6
        # compute the entropy
        log_probs = probabilities.log_prob(actions).sum(axis=-1)  # used the reshape the outoput from -1 to 1
        #print("log_probs  " + str(log_probs))
        val = (2*(np.log(2) - actions - F.softplus(-2*actions)))

        try:
            log_probs -= val.sum(axis=1) #T.log(1-action.pow(2)+self.reparam_noise)  #axis=1
        except:
            log_probs -= val.sum(axis=-1)

        return action, log_probs




    # def sample_normal(self, state, reparameterize=True):
    #     mu, sigma = self.forward(state)
    #     probabilities = Normal(mu, sigma)


        # actions = probabilities.rsample()

        # action = T.tanh(actions) * T.tensor(self.max_action)
        # log_probs = probabilities.log_prob(actions)
        # log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        # log_probs = log_probs.sum(1, keepdim=True)

        # return action, log_probs

