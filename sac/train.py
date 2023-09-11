import torch 
import torch.nn.functional as F

import gymnasium as gym
from tqdm import tqdm
from model import ReplayBuffer, Actor, Critic
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
    
# env = gym.make("LunarLander-v2", continuous=True, render_mode="rgb_array")
env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = Actor(env).to(device)
Q1 = Critic(env).to(device)
Q2 = Critic(env).to(device)

Q1_ = Critic(env).to(device)
Q2_ = Critic(env).to(device)

Q1_.load_state_dict(Q1.state_dict())
Q2_.load_state_dict(Q2.state_dict())

optim_a = torch.optim.Adam(actor.parameters(), lr=0.0001)
optim_q1 = torch.optim.Adam(Q1.parameters(), lr=0.0003)
optim_q2 = torch.optim.Adam(Q2.parameters(), lr=0.0003)

# Hyperparameters
gamma = 0.99
temp = 0.0003     # Temperature of the entropy term
train_ep_freq = 1
num_opt_steps = 1
eval_freq = 20
num_eval_ep = 10
start_steps = 50000
clip_value = 0.5

global_step = 0

rb = ReplayBuffer(state_dim=env.observation_space.shape[0], 
                  action_dim=env.action_space.shape[0], 
                  length=1000000,
                  batch_size=256,
                  device=device)

def soft_update(target, source, tau=0.01):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + 
                                (1.0 - tau) * target_param.data)

def run_episode(test=False, render=False, random=False):
    global global_step
    obs, _ = env.reset()
    obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) * 2 -1
    done = False
    rews_ep = []

    step = 0
    
    # Play the game and collect data
    while not done:
        if random:
            action = env.action_space.sample()
        else:
            action = actor.sample(torch.tensor(obs).unsqueeze(0).to(device), test=test)[0]
            action = action[0].to('cpu').detach().numpy()
    
        obs_, reward, terminated, truncated, _ = env.step(action)
        obs_ = (obs_ - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) * 2 -1
        done = terminated or truncated
        
        if not truncated:
            rb.store(obs, action, reward, obs_, terminated)
        obs = obs_
        
        rews_ep.append(reward)
        step += 1
        global_step += 1
    return rews_ep, step

def optimize(episode):
    s, a, r, s_, d = rb.sample()
    
    # Optimize the critics
    a_theta, log_prob = actor.sample(s_)
    q = torch.min(Q1_(s_, a_theta), Q2_(s_, a_theta)).to(device) 
    
    target = r + gamma * (1 - d) * (q - temp * log_prob)
    
    loss_q1 = F.mse_loss(Q1(s, a), target.detach())
    loss_q2 = F.mse_loss(Q2(s, a), target.detach())

    Q1.zero_grad()      
    loss_q1.backward()     
    clip_grad_norm_(Q1.parameters(), clip_value)
    optim_q1.step()        
    
    Q2.zero_grad()
    loss_q2.backward()
    clip_grad_norm_(Q2.parameters(), clip_value)
    optim_q2.step()
    
    # Optimize the actor
    a_theta, log_prob = actor.sample(s)
    q = torch.min(Q1(s, a_theta), Q2(s, a_theta)).to(device)
    
    loss_a = -q.mean() - temp * log_prob.mean()
    
    actor.zero_grad()
    loss_a.backward()
    # break
    clip_grad_norm_(actor.parameters(), clip_value)
    optim_a.step()
    
    writer.add_scalar("Loss/q1", loss_q1.cpu().detach(), episode)
    writer.add_scalar("Loss/q2", loss_q2.cpu().detach(), episode)
    writer.add_scalar("Loss/a", loss_a.cpu().detach(), episode)
    writer.add_scalar("Loss/entropy", -temp * log_prob.mean(), episode)
    
    soft_update(Q1_, Q1, tau=0.01)
    soft_update(Q2_, Q2, tau=0.01)

# Collect some data before training
while global_step < start_steps:
    rews_ep, step = run_episode()


# Train
for episode in tqdm(range(4000000)):   
    rews_ep, step = run_episode()
    
    if episode % eval_freq == 0:
        eval_rews = []
        eval_steps = []
        for _ in range(num_eval_ep):
            rews_ep, step = run_episode(test=True)
            eval_rews.append(sum(rews_ep))
            eval_steps.append(step)
        writer.add_scalar("Eval/reward", sum(eval_rews) / len(eval_rews), episode)
        writer.add_scalar("Eval/step", sum(eval_steps) / len(eval_steps), episode)

    if episode % train_ep_freq == 0:
        for opt_step in range(num_opt_steps):
            optimize(episode)
            