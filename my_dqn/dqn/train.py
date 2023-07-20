import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, TransformObservation
from gymnasium.wrappers import FrameStack, FlattenObservation, ResizeObservation

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from dqn import QNetwork, ReplayBuffer, DQN

        
if __name__ == "__main__":
    # env = gym.make("CartPole-v1")
    # env = FlattenObservation(env)
    
    env = gym.make("ALE/Pong-v5")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 64)
    env = FrameStack(env, 3)
    env = FlattenObservation(env)
    env = TransformObservation(env, lambda obs: obs / 255.0)
    
    lr=0.001
    batch_size = 1024
    eps_s = 1.00
    eps_e = 0.01
    eps_decay_steps = 5e5
    gamma = 0.998
    max_episodes = 100000
    eval_freq_ep = 20
    updt_freq_st = 200
    training_starts =  10 * batch_size
    n_eval_episodes = 10
    
    model = QNetwork(env)
    model_ = QNetwork(env)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rb = ReplayBuffer(length=100000, batch_size=batch_size)
    writer = SummaryWriter()
    
    dqn = DQN(env, model, model_, loss_fn,  optimizer, 
              rb, writer, batch_size, gamma, max_episodes,
              eval_freq_ep, n_eval_episodes, updt_freq_st, eps_s, eps_e, eps_decay_steps, 
              training_starts)
    
    dqn.train()
    