import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions.normal import Normal

""" Policy network, that take a state and return the action sampled according to its probability """
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions = None, name='actor', lr= 1e-3):

        # definition for the structure of the network
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.lr = lr

        # definition of the hidden layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # 2 different output layers for mean and standard deviation
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        # optimizer for the actor params
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # forward step to compute the mean and the standard deviation
    def forward(self, state):
        out = self.fc1(state)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        mu = self.mu(out)
        sigma = self.sigma(out)

        sigma = T.clamp(sigma, min= -20, max=2)  # clamp sigma into the the interval
        sigma = T.exp(sigma)
        return mu, sigma

    # start function for the gradient

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

    # end function for the gradient



    # given a single state, sample a probability distribution and return an action
    def sample_normal(self, state):

        # evaluate sigma and mu with the network
        mu, sigma = self.forward(state)

        # define the normal
        probabilities = Normal(mu, sigma)

        # take the probabilities in a non deterministic matter
        actions = probabilities.rsample()

        # hyperbolic tangent to limit between 1 and -1, and normalization, multiplying with the MAX action tensor
        action = T.tanh(actions) * T.tensor(self.max_action)

        # reduce the effective of the action, (fewer action range)
        # action = action *0.5

        return action

    # given a batch of states, sample their probability distribution and return actions and entropies
    def sample_normal_batch(self, state):
        # evaluate sigma and mu with the network
        mu, sigma = self.forward(state)

        # define the normal
        probabilities = Normal(mu, sigma)

        # take the probabilities in a non deterministic matter (batch size)
        actions = probabilities.rsample()

        # hyperbolic tangent to limit between 1 and -1, and normalization, multiplying with the action tensor
        action = T.tanh(actions) * T.tensor(self.max_action)

        # compute the entropy
        log_probs = probabilities.log_prob(actions).sum(axis=-1)  # used the reshape the outoput from -1 to 1
        val = (2*(np.log(2) - actions - F.softplus(-2*actions)))
        try:
            log_probs -= val.sum(axis=1)
        except:
            log_probs -= val.sum(axis=-1)

        return action, log_probs
        


