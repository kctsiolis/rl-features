import torch
import torch.nn.functional as F

def check_distribution(probs):
    assert torch.all(probs >= 0)
    l = len(probs.size())
    row_sums = torch.sum(probs, dim=l-1).to(torch.float64)
    assert torch.all(row_sums == 1)
    

class MDP():
    """Markov Decision Process class. We assume reward is deterministic and depends only on state.

    Attributes:
        p: Transition probabilities
        rho: Initial state distribution
        r: Rewards
    """
    def __init__(self, num_states=2, num_actions=2, p=None, rho=None, r=None, gamma=0.9):
        self.num_states = num_states
        assert num_states > 0
        self.num_actions = num_actions
        assert num_actions > 0
        self.initialize_p(p)
        self.initialize_rho(rho)
        self.initialize_r(r)
        self.reward = 0
        self.time = 0
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.cur_state = torch.multinomial(self.rho, 1).item()

    def step(self, action):
        self.cur_state = torch.multinomial(self.p[self.cur_state, action], 1).item()
        self.reward += self.gamma**(self.time) * self.r[self.cur_state].item()
        self.time += 1

    def initialize_p(self, p):
        if p is None:
            self.p = torch.randn((self.num_states, self.num_actions, self.num_states))
            self.p = F.softmax(self.p, dim=2)
        else:
            assert p.size() == (self.num_states, self.num_actions, self.num_states)
            check_distribution(p)
            self.p = p

    def initialize_rho(self, rho):
        if rho is None:
            self.rho = torch.ones(self.num_states) * 1/self.num_states
        else:
            assert rho.size() == (self.num_states)
            check_distribution(rho)
            self.rho = rho

    def initialize_r(self, r):
        if r is None:
            self.r = torch.randn(self.num_states)
        else:
            assert r.size() == (self.num_states)
            self.r = r