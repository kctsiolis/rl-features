import numpy as np

class Agent():
    def __init__(self):
        pass

    def act(self):
        pass

class DiscreteAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return np.random.choice(self.action_space)
