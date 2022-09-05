import numpy as np
from scipy.stats import multivariate_normal
import random

class GridEnv():
    def __init__(self, dim=2, length=10, num_modes=2, eps=0):
        self.dim = dim 
        self.length = length #Side length of grid (all sides of equal length)
        self.num_modes = num_modes #Number of modes (Gaussians) in reward function

        self.means = np.random.uniform(low=0, high=self.length, size=(self.num_modes, self.dim))
        self.coeffs = np.random.uniform(low=0, high=10, size=(self.num_modes,)) #Coefficients for each Gaussian

        self.state = np.zeros(self.dim)

        self.eps = eps #Probability of action being overriden by a random action

    def compute_reward(self):
        reward = 0
        for i in range(self.num_modes):
            reward += multivariate_normal.pdf(self.state, mean=self.means[i])

        return reward

    def step(self, action):
        if abs(action) > self.dim:
            return ValueError('Action must be in range [-dim,dim].')
        r = np.random.binomial(1, self.eps)
        if r == 1: #Override with a random action
            action = random.randint(-1 * self.dim, self.dim)
        coord = self.state[abs(action)-1]
        dir = np.sign(action)
        if (coord == 0 and dir == -1) or (coord == self.length - 1 and dir == 1): #Can't go off the grid
            return
        self.state[abs(action)-1] += dir
