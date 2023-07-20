import gymnasium as gym
from ddpg import DDPG

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    lr = 0.001
    batch_size = 64
    gamma = 0.999
    max_episodes = 100000
    eval_freq_ep = 30
    n_eval_episodes = 10
    
    dqn = DDPG(env, lr, batch_size, gamma, max_episodes, 
              eval_freq_ep, n_eval_episodes)
    
    dqn.train()
    