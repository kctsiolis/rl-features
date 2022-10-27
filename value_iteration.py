import torch

class ValueIterationAgent():
    
    def __init__(self, mdp):
        self.mdp = mdp
        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions
        self.p = mdp.p
        self.r = mdp.r
        self.gamma = mdp.gamma
        self.value = torch.randn(self.num_states)
        self.policy = torch.zeros(self.num_states)

    def train(self, tol=1e-3, max_epochs=1000):
        #Implements value iteration as presented in Sutton and Barto
        for epoch in range(max_epochs):
            error = 0
            for s in range(self.num_states):
                v = float('-inf')
                for a in range(self.num_actions):
                    candidate = torch.sum(self.p[s,a] * (self.r + self.gamma * self.value)).item()
                    if candidate > v:
                        v = candidate
                error = max([error, abs(v-self.value[s])])
                self.value[s] = v
            if error <= tol:
                break

        #Get a deterministic policy from value function
        for s in range(self.num_states):
            v = float('-inf')
            argmax = 0
            for a in range(self.num_actions):
                candidate = torch.sum(self.p[s,a] * (self.r + self.gamma * self.value)).item()
                if candidate > v:
                    argmax = a
            self.policy[s] = argmax

        

                
