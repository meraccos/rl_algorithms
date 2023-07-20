import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import numpy as np


# class Actor(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         state_dim = env.observation_space.shape[0]
#         action_dim = env.action_space.shape[0]
#         self.act_min = env.action_space.low[0]
#         self.act_max = env.action_space.high[0]
        
#         self.l1 = nn.Linear(state_dim, 400)
#         self.bn1 = nn.BatchNorm1d(400)
#         self.l2 = nn.Linear(400, 300)
#         self.bn2 = nn.BatchNorm1d(300)
#         self.l3 = nn.Linear(300, action_dim)
        
#         nn.init.orthogonal_(self.l1.weight)
#         nn.init.orthogonal_(self.l2.weight)
#         nn.init.orthogonal_(self.l3.weight)
#         self.eval()
    
#     def forward(self, x):
#         a = F.relu(self.bn1(self.l1(x)))
#         a = F.relu(self.bn2(self.l2(a)))
#         a = F.tanh(self.l3(a))
        
#         # Scale the action
#         a = self.act_max * (1+a)/2 + self.act_min * (1-a)/2
#         return a

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.act_min = env.action_space.low[0]
        self.act_max = env.action_space.high[0]
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        nn.init.orthogonal_(self.l1.weight)
        nn.init.orthogonal_(self.l2.weight)
        nn.init.orthogonal_(self.l3.weight)
        self.eval()
    
    def forward(self, x):
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        a = F.tanh(self.l3(a))
        
        # Scale the action
        a = self.act_max * (1+a)/2 + self.act_min * (1-a)/2
        return a


# class Critic(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         state_dim = env.observation_space.shape[0]
#         action_dim = env.action_space.shape[0]
#         self.act_min = env.action_space.low[0]
#         self.act_max = env.action_space.high[0]
        
#         self.l1 = nn.Linear(state_dim, 400)
#         self.bn1 = nn.BatchNorm1d(400)
#         self.l2 = nn.Linear(400+action_dim, 300)
#         self.bn2 = nn.BatchNorm1d(300)
#         self.l3 = nn.Linear(300, 1)
        
#         nn.init.orthogonal_(self.l1.weight)
#         nn.init.orthogonal_(self.l2.weight)
#         nn.init.orthogonal_(self.l3.weight)
#         self.eval()

#     def forward(self, obs, action):
#         a = F.relu(self.bn1(self.l1(obs))) 
#         a = torch.cat((a, action), dim=1)
#         a = F.relu(self.bn2(self.l2(a)))
#         a = self.l3(a)
#         return a #.squeeze()

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.act_min = env.action_space.low[0]
        self.act_max = env.action_space.high[0]
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400+action_dim, 300)
        self.l3 = nn.Linear(300, 1)
        
        nn.init.orthogonal_(self.l1.weight)
        nn.init.orthogonal_(self.l2.weight)
        nn.init.orthogonal_(self.l3.weight)
        self.eval()

    def forward(self, obs, action):
        a = F.relu(self.l1(obs))
        a = torch.cat((a, action), dim=1)
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return a #.squeeze()
    
    
class ReplayBuffer:
    def __init__(self, length=10000, batch_size = 32):
        self.mem = deque(maxlen=length)
        self.length = length
        self.batch_size = batch_size
    
    def store(self, s, a, r, s_, d):
        self.mem.append([s, a, r, s_, d])
    
    def sample(self):
        idxs = torch.randint(high=len(self.mem), size=[self.batch_size])
        batch = [self.mem[idx] for idx in idxs]
        return map(torch.Tensor, list(zip(*batch)))

#     def sample(self):
#         idxs = np.random.randint(len(self.mem), size=self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*[self.mem[idx] for idx in idxs])
        
#         return map(torch.tensor, (np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)))

    
class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, 
                 max_sigma=0.3, min_sigma=0.2, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)