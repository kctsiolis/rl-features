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

class StateActionLearner():
    def __init__(self, env, agent, lr=1e-3, batch_size=32, num_iters=10000):
        self.env = env
        self.agent = agent
        self.lr = lr 
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.loss = nn.MSELoss()

        #State representations
        self.v = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((env.num_states, env.dim))))
        #Action representations
        self.w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((env.num_actions, env.dim))))
        self.optimizer = torch.optim.Adam([self.w, self.v], lr=self.lr)


    def train(self):
        for iter in range(self.num_iters):
            self.optimizer.zero_grad()
            cur_states, actions, next_states = self.env.explore(self.agent, self.batch_size)
            hash_cur_states = torch.zeros(self.batch_size, dtype=torch.long)
            hash_next_states = torch.zeros(self.batch_size, dtype=torch.long)
            for i in range(self.batch_size):
                hash_cur_states[i] = self.env.hash_state(cur_states[i,:])
                hash_next_states[i] = self.env.hash_state(next_states[i,:])
            loss = self.loss(self.w[actions], self.v[hash_next_states] - self.v[hash_cur_states])
            loss.backward()
            self.optimizer.step()
            print('Iteration {}: Loss = {}'.format(iter, loss.item()))

        return self.v, self.w

if __name__ == '__main__':
    grid = GridEnv()
    agent = DiscreteAgent(range(grid.num_actions))
    learner = StateActionLearner(grid, agent)
    v, w = learner.train()

    actual_states = torch.zeros((grid.num_states, grid.dim))
    for i in range(grid.num_states):
        actual_states[i,:] = grid.unhash_state(i)

    actual_actions = torch.zeros((grid.num_actions, grid.dim))
    for i in range(grid.num_actions-1):
        idx = i // 2
        dir = -1 if i % 2 == 0 else 1
        actual_actions[i, idx] = dir

    print("State Representations")
    print(v)
    print("Actual States")
    print(actual_states)
    print("Action Representations")
    print(w)
    print("Actual Actions")
    print(actual_actions)
