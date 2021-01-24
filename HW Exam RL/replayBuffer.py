import numpy as np
import torch

""" continuous environment so in this case number of actions is the number of components for that actions"""
class ReplayBuffer():
    def __init__(self, max_size_buffer, input_shape, n_actions):
        self.size = max_size_buffer
        self.state_memory = np.zeros((self.size, *input_shape)) # initialization of the memory
        self.new_state_memory = np.zeros((self.size, *input_shape)) # to keep track of the states that result after our actions
        self.action_memory = np.zeros((self.size, n_actions)) #keep track of actions
        self.reward_memory = np.zeros(self.size)  # keep track of memory
        self.terminal_memory = np.zeros(self.size, dtype=np.bool)
        self.pointer = 0  # first available memory

    # store the transition information after step
    def store_transition(self, state, action, reward, state_, done):
        index = self.pointer % self.size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.pointer += 1
    # random sampling function from the replay buffer e
    def sample_buffer(self, batch_size = 32):
        max_mem = min(self.pointer, self.size)

        batch = np.random.choice(max_mem, batch_size)
        batch_tmp = dict(states = self.state_memory[batch],
        states_ = self.new_state_memory[batch],
        actions = self.action_memory[batch],
        rewards = self.reward_memory[batch],
        dones = self.terminal_memory[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch_tmp.items()} # states, actions, rewards, states_, dones
