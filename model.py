import torch
import torch.nn as nn
from random import uniform
import math

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class Controller(torch.nn.Module):
    '''Sigle element of the modular network'''
    
    def __init__(self, D_in, H, D_out, min_std=1e-6, init_std=1.0):
        super(Controller, self).__init__()
        self.din = D_in
        self.dout = D_out
        self.controller = nn.Sequential(nn.Linear(D_in, H), 
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D_out))
        self.sigma = nn.Parameter(torch.Tensor(D_out))
        self.sigma.data.fill_(math.log(init_std))

        self.min_log_std = math.log(min_std)
        self.saved_log_probs = []
        self.rewards = []
   
    def forward(self, x):
        means = self.controller(x)
        scales = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        dist = torch.distributions.Normal(means, scales)
        action = dist.mean
        return action, 0#, dist.log_prob(action)

       
class ControllerCombinator(torch.nn.Module):
    ''' The combinator that is modified during lifetime'''
    def __init__(self, D_in, N, H, D_out, min_std=1e-6, init_std=1.0, det=False):
        super(ControllerCombinator, self).__init__()
    
        # Initialize the 
        self.elements = torch.nn.ModuleList()
        for i in range(N):
            c = Controller(D_in, H, D_out)
            self.elements.append(c)
      
        # Networks that will combine the outputs of the different elemenrts
        self.combinators = nn.ModuleList()
        for _ in range(D_out):
            self.combinators.append(nn.Sequential(nn.Linear(N,N*2),
                nn.ReLU(),
                nn.Linear(N*2, 1)))
     
        self.sigma = nn.Parameter(torch.Tensor(D_out))

        self.apply(weight_init)

        self.sigma.data.fill_(math.log(init_std))
        self.min_log_std = math.log(min_std)
        self.deterministic = det


    def forward(self, x):
        votes = []
        votes_log_probs = []
        for fnc in self.elements:
            vote, vlp = fnc(x)
            votes.append(vote)
            votes_log_probs.append(vlp)
        
        votes_ = torch.stack(votes)
        votes_t = torch.transpose(votes_, 0, 1)

        means = []
        for vm, com in zip(votes_t, self.combinators):
            means.append(com(vm))

        means = torch.stack(means)
        means = torch.flatten(means)
        scales = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        dist = torch.distributions.Normal(means, scales)
        out = dist.mean if self.deterministic else dist.sample()

        votes = torch.cat(votes)
        debug_info = (means.detach().numpy(), votes.detach().numpy())
        return out, dist.log_prob(out) + sum(votes_log_probs), debug_info

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