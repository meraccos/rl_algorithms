import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

env = gym.make("FrozenLake-v1")