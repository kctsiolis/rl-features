import numpy as np
from scipy.stats import multivariate_normal
import random
import torch

class GridEnv():
    def __init__(self, dim=2, length=10, num_modes=2, eps=0):
        self.dim = dim 
        self.length = length #Side length of grid (all sides of equal length)
        self.num_modes = num_modes #Number of modes (Gaussians) in reward function

        self.means = np.random.uniform(low=0, high=self.length, size=(self.num_modes, self.dim))
        self.coeffs = np.random.uniform(low=0, high=10, size=(self.num_modes,)) #Coefficients for each Gaussian

        self.num_actions = 2 * self.dim + 1
        self.eps = eps #Probability of action being overriden by a random action

        self.reset()

    def reset(self):
        self.state = np.random.randint(low=0, high=self.length, size=(self.dim,))

    def compute_reward(self):
        reward = 0
        for i in range(self.num_modes):
            reward += multivariate_normal.pdf(self.state, mean=self.means[i])

        return reward

    def step(self, action):
        if action > 2 * self.dim or action < 0:
            return ValueError('Action must be in range [0, {}].'.format(self.num_actions))
        r = np.random.binomial(1, self.eps)
        if r == 1: #Override with a random action
            action = random.randint(0, self.num_actions-1)
        if action == self.num_actions - 1: #Do nothing
            return self.state

        coord = self.state[action // 2]
        dir = 2 * (action % 2) - 1 #1 is left, 1 is right
        if (coord == 0 and dir == -1) or (coord == self.length - 1 and dir == 1): #Can't go off the grid
            return self.state
        self.state[action // 2] += dir

        return self.state

    #Collect tuples x_t = (s_t, a_t, s_{t+1}) with an agent's policy
    def explore(self, agent, batch_size):
        cur_states = torch.zeros(batch_size, self.dim)
        actions = torch.zeros(batch_size, dtype=torch.long)
        next_states = torch.zeros(batch_size, self.dim)
        for i in range(batch_size):
            self.reset()
            cur_states[i,:] = torch.Tensor(self.state)
            action = agent.act(self.state)
            actions[i] = action
            next_states[i,:] = torch.Tensor(self.step(action))

        return cur_states, actions, next_states
