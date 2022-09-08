import torch
from torch import nn
from gridworld import GridEnv
from agent import DiscreteAgent

class ActionLearner():
    def __init__(self, env, agent, lr=1e-3, batch_size=32, num_iters=10000):
        self.env = env
        self.agent = agent
        self.lr = lr
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.loss = nn.MSELoss()

        self.w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((env.num_actions, env.dim))))
        self.optimizer = torch.optim.Adam([self.w], lr=self.lr)


    def train(self):
        for iter in range(self.num_iters):
            self.optimizer.zero_grad()
            cur_states, actions, next_states = self.env.explore(self.agent, self.batch_size)
            loss = self.loss(self.w[actions], next_states - cur_states)
            loss.backward()
            self.optimizer.step()
            print('Iteration {}: Loss = {}'.format(iter, loss.item()))

        return self.w

if __name__ == '__main__':
    grid = GridEnv()
    agent = DiscreteAgent(range(grid.num_actions))
    learner = ActionLearner(grid, agent)
    w = learner.train()
    print(w)
