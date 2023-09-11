import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


# class Actor(nn.Module):
#     def __init__(self, env=None):
#         super().__init__()
#         # self.input_dim = env.observation_space.n                    # for Discrete obs
#         self.input_dim = env.observation_space.shape[0]              # for Box obs
#         self.output_dim = env.action_space.n
        
#         self.l1 = nn.Linear(self.input_dim, 64)
#         self.l2 = nn.Linear(64, 64)
#         self.l3 = nn.Linear(64, self.output_dim)

#     def forward(self, x):
#         # x = F.one_hot(x.to(int), self.input_dim).to(torch.float32)    # for Discrete obs
#         a = F.relu(self.l1(x))
#         a = F.relu(self.l2(a))
#         a = self.l3(a)
#         a = Categorical(logits=a)
#         return a


# class Critic(nn.Module):
#     def __init__(self, env=None):
#         super().__init__()
#         # self.input_dim = env.observation_space.n                      # for Discrete obs
#         self.input_dim = env.observation_space.shape[0]              # for Box obs
        
#         self.l1 = nn.Linear(self.input_dim, 64)
#         self.l2 = nn.Linear(64, 64)
#         self.l3 = nn.Linear(64, 1)

#     def forward(self, x):
#         # x = F.one_hot(x.to(int), self.input_dim).to(torch.float32)    # For Discrete obs
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.l3(x)
#         return x.squeeze()


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