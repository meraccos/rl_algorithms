import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


class Policy(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        # self.input_dim = env.observation_space.n                    # for Discrete obs
        self.input_dim = env.observation_space.shape[0]              # for Box obs
        self.output_dim = env.action_space.n
        
        self.l1 = nn.Linear(self.input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, self.output_dim)

    def forward(self, x):
        # x = F.one_hot(x.to(int), self.input_dim).to(torch.float32)    # for Discrete obs
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        a = Categorical(logits=a)
        return a


class Value(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        # self.input_dim = env.observation_space.n                      # for Discrete obs
        self.input_dim = env.observation_space.shape[0]              # for Box obs
        
        self.l1 = nn.Linear(self.input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, x):
        # x = F.one_hot(x.to(int), self.input_dim).to(torch.float32)    # For Discrete obs
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x.squeeze()
    