"""
Simple vanilla policy gradient implementation.
Advantage fn is the sum of future rewards.
"""
import torch 
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Policy

env = gym.make("FrozenLake-v1", is_slippery=False)
# env = gym.make("Taxi-v3")
# env = gym.make("CliffWalking-v0")

policy = Policy(env)
optimizer = torch.optim.Adam(policy.parameters())

# Hyperparameters
gamma = 0.99
batch_size = 16

rews = []
loss = 0.0

for episode in tqdm(range(100)):
    m_obs = []
    m_act = []
    m_rew = []
    
    obs, _ = env.reset()
    done = False
    
    # Run the episode
    while not done:
        p = policy(torch.tensor(obs))
        action = p.sample()
        m_obs.append(obs)
    
        obs, reward, terminated, truncated, _ = env.step(action.detach().item())
        done = terminated or truncated
    
        m_act.append(action.item())
        m_rew.append(reward)
    
    # Policy update    
    m_obs = torch.tensor(m_obs).unsqueeze(1)
    m_act = torch.tensor(m_act)
    
    # Calculate the discounted rewards
    G = 0
    disc_rew = []
    for reward in reversed(m_rew):
        G = reward + gamma * G
        disc_rew.insert(0, G)
    
    rews.append(G)
    
    # Collect Policy Gradient loss
    # Advantage is simply the discounted future rewards 
    for obs_, act_, rew_ in zip(m_obs, m_act, disc_rew):
        log_prob = policy(obs_).log_prob(act_)
        loss -= log_prob * rew_
    
    # Optimize the policy
    if episode % batch_size == 0 and episode != 0:    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = 0.0
        
        print(sum(rews) / batch_size)
        rews = []