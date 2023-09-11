import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        
        
class Actor(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.input_dim = env.observation_space.shape[0]             
        self.output_dim = env.action_space.shape[0]
        
        self.l1 = nn.Linear(self.input_dim, 400)
        self.l2 = nn.Linear(400, 300)
        
        self.mu = nn.Linear(300, self.output_dim)
        self.std = nn.Linear(300, self.output_dim)
        
        self.apply(weights_init_)

    def forward(self, x):
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        
        mu = self.mu(a)
        log_std = torch.clamp(self.std(a), -20, 2)
        return mu, log_std

    def sample(self, obs, test=False):
        mu, log_std = self(obs)
        if test:
            return torch.tanh(mu), 0
        n = Normal(mu, log_std.exp())
        action = n.rsample().tanh()
        log_prob = (n.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)).sum(1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.obs_dim = env.observation_space.shape[0]             
        self.action_dim = env.action_space.shape[0]
        input_dim = self.obs_dim + self.action_dim

        self.l1 = nn.Linear(input_dim , 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
        self.apply(weights_init_)

    def forward(self, s, act):   
        a = torch.cat((s, act), dim=1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return a.squeeze()

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, length=10000, batch_size=32, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.length = length
        self.batch_size = batch_size
        self.device = device

        self.states = torch.zeros((length, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((length, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros(length, dtype=torch.float32).to(device)
        self.next_states = torch.zeros((length, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros(length, dtype=torch.float32).to(device)

        self.idx = 0
        self.count = 0
    
    def store(self, s, a, r, s_, d):
        self.states[self.idx] = torch.tensor(s, dtype=torch.float32).to(self.device)
        self.actions[self.idx] = torch.tensor(a, dtype=torch.float32).to(self.device)
        self.rewards[self.idx] = torch.tensor(r, dtype=torch.float32).to(self.device)
        self.next_states[self.idx] = torch.tensor(s_, dtype=torch.float32).to(self.device)
        self.dones[self.idx] = torch.tensor(d, dtype=torch.float32).to(self.device)

        self.idx = (self.idx + 1) % self.length
        self.count = min(self.count + 1, self.length)

    def sample(self):
        idxs = torch.randint(0, self.count, (self.batch_size,), device=self.device)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], 
                self.next_states[idxs], self.dones[idxs])