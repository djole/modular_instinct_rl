import torch
import torch.nn as nn
from random import uniform
import math


class Controller(torch.nn.Module):
    
    def __init__(self, D_in, H, D_out, min_std=1e-6, init_std=1.0):
        super(Controller, self).__init__()
        self.din = D_in
        self.dout = D_out
        self.controller = nn.Sequential(nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H,H),
            nn.ReLU(),
            nn.Linear(H, D_out))
        self.sigma = nn.Parameter(torch.Tensor(D_out))
        self.sigma.data.fill_(math.log(init_std))

        self.min_log_std = math.log(min_std)

        #for p, sigma in zip(self.controller.parameters(), [uniform(0, 100) for i in range(100000)]):
        #    p = torch.randn_like(p) * sigma
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        means = self.controller(x)
        scales = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        dist = torch.distributions.Normal(means, scales)
        action = dist.sample()
        return action, dist.log_prob(action)

        
class ControllerCombinator(torch.nn.Module):

    def __init__(self, D_in, N, H, D_out, min_std=1e-6, init_std=1.0):
        super(ControllerCombinator, self).__init__()
        self.elements = torch.nn.ModuleList()
        for i in range(N):
            c = Controller(D_in, H, D_out)
            self.elements.append(c)
        self.combinator = nn.Linear(D_out * N, D_out)
        
        self.sigma = nn.Parameter(torch.Tensor(D_out))
        self.sigma.data.fill_(math.log(init_std))
        self.min_log_std = math.log(min_std)

    def forward(self, x):
        #if not torch.any(torch.isnan(self.sigma)):
        #    print("sigma ok {}".format(self.sigma))
        #else:
        #    assert not torch.any(torch.isnan(self.sigma))
        #votes = list(map(lambda fnc: fnc(x), self.elements))
        votes = []
        votes_log_probs = []
        for fnc in self.elements:
            vote, vlp = fnc(x)
            votes.append(vote)
            votes_log_probs.append(vlp)
        
        votes = torch.cat(votes)
        means = self.combinator(votes)
        scales = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        dist = torch.distributions.Normal(means, scales)
        out = dist.sample()
        return out, dist.log_prob(out) + sum(votes_log_probs)

    def expose_modules(self):
        for module in self.elements:
            module.requires_grad = True
        self.combinator.requires_grad = True
    
    def cover_modules(self):
        for module in self.elements:
            module.requires_grad = False
        self.combinator.requires_grad = True
    
    def get_combinator_params(self):
        comb_params = []
        dct = self.named_parameters()
        for pkey, ptensor in dct:
            if 'combinator' in pkey or pkey == 'sigma':
                comb_params.append(ptensor)
        return comb_params