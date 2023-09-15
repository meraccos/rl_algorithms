import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.shape[0]
    
        self.l1 = nn.Linear(self.input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, self.output_dim * 2)

    def forward(self, x):
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        mu, log_std = self.l3(a).chunk(2, dim=-1)
        return mu, log_std
    
    def sample(self, x, test=False):
        mu, log_std = self(x)
        if test:
            action = mu
            logp = None
        
        else:
            std = torch.exp(log_std)
            n = Normal(mu, std)
            action = n.rsample().tanh()
            logp = (n.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)).sum(1)
        
        return action, logp
    
    def logp(self, x, action):
        mu, log_std = self(x)
        std = torch.exp(log_std)
        n = Normal(mu, std)
        logp = (n.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)).sum(1)
        
        return logp

    
class Critic(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.input_dim = env.observation_space.shape[0]
        
        self.l1 = nn.Linear(self.input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x.squeeze()


class TrajectoryStorage:
    def __init__(self, device='cpu'):
        self.device = device

        self.states = []
        self.actions = []
        self.logp = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def store(self, s, a, lp, r, s_, d):
        self.states.append(s)
        self.actions.append(a)
        self.logp.append(lp)
        self.rewards.append(r)
        self.next_states.append(s_)
        self.dones.append(int(d))

    def load(self):
        states = torch.tensor(self.states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(self.actions, device=self.device, dtype=torch.float32)
        logp = torch.tensor(self.logp, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(self.next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.dones, device=self.device, dtype=torch.float32)
        
        return states, actions, logp, rewards, next_states, dones
    
    def reset(self):
        self.states = []
        self.actions = []
        self.logp = []
        self.rewards = []
        self.next_states = []
        self.dones = []