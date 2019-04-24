import torch
import torch.nn as nn
from random import uniform
class Controller(torch.nn.Module):
    
    def __init__(self, D_in, H, D_out):
        super(Controller, self).__init__()
        self.din = D_in
        self.dout = D_out
        self.controller = nn.Sequential(nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H,H),
            nn.ReLU(),
            nn.Linear(H, D_out))
        self.sigma = torch.ones(1, D_out) * 0.01

        #for p, sigma in zip(self.controller.parameters(), [uniform(0, 100) for i in range(100000)]):
        #    p = torch.randn_like(p) * sigma
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        means = self.controller(x)
        dist = torch.distributions.Normal(means, self.sigma)
        action = dist.sample()
        return action, dist.log_prob(action)
        
class ControllerCombinator(torch.nn.Module):

    def __init__(self, D_in, N, H, D_out):
        super(ControllerCombinator, self).__init__()
        self.modules = []
        for i in range(N):
            c = Controller(D_in, H, D_out)
            self.modules.append(c)
            c.requires_grad = False
        self.combinator = nn.Linear(D_out * N, D_out)
    
    def forward(self, x):
        votes = map(lambda fnc: fnc(x), self.modules)
        votes = torch.tensor(votes)
        out = self.combinator(votes)
        out.clamp_(-1, 1)
        return out

    def expose_modules(self):
        for module in self.modules:
            module.requires_grad = True
        self.combinator.requires_grad = False
    
    def expose_combinaror(self):
        for module in self.modules:
            module.requires_grad = False
        self.combinator.requires_grad = True