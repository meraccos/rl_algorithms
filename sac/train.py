import torch
import torch.nn.functional as F
import gym
from gym.wrappers import TransformObservation
from tqdm import tqdm
from model import ReplayBuffer, Actor, Critic
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
    
# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99
temp = 0.01     # Temperature of the entropy term
eval_freq = 20
num_eval_ep = 20
start_steps = 5000
clip_value = 2.5
tau = 0.0001

# Tensorboard Writer
writer = SummaryWriter()

# Environment setup
env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
obs_high, obs_low = env.observation_space.high, env.observation_space.low
obs_a, obs_b = 2 / (obs_high - obs_low), (obs_high + obs_low) / (obs_high - obs_low)
env = TransformObservation(env, lambda obs: obs * obs_a - obs_b)

# Model initialization
actor = Actor(env).to(device)
Q1 = Critic(env).to(device)
Q2 = Critic(env).to(device)

Q1_ = Critic(env).to(device)
Q2_ = Critic(env).to(device)

Q1_.load_state_dict(Q1.state_dict())
Q2_.load_state_dict(Q2.state_dict())

optim_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
optim_q1 = torch.optim.Adam(Q1.parameters(), lr=3e-4)
optim_q2 = torch.optim.Adam(Q2.parameters(), lr=3e-4)

rb = ReplayBuffer(state_dim=env.observation_space.shape[0], 
                  action_dim=env.action_space.shape[0], 
                  length=1000000,
                  batch_size=512,
                  device=device)

def soft_update(target, source, tau=0.01):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + 
                                (1.0 - tau) * target_param.data)

def run_episode(test=False, render=False, random=False):
    global global_step
    obs, _ = env.reset()
    done = False
    rews_ep = []

    step = 0
    
    # Play the game and collect data
    while not done:
        if random:
            action = env.action_space.sample()
        else:
            action = actor.sample(torch.tensor(obs).unsqueeze(0).to(device), test=test)
            action = action[0][0].to('cpu').detach().numpy()
    
        obs_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rb.store(obs, action, reward, obs_, terminated)
        obs = obs_
        step += 1

        rews_ep.append(reward)
        
        if not test:
            global_step += 1
            optimize(global_step)
    return rews_ep, step

def optimize(episode):
    s, a, r, s_, d = rb.sample()
    
    # Optimize the critics
    a_theta, log_prob = actor.sample(s_)
    q = torch.min(Q1_(s_, a_theta), Q2_(s_, a_theta)).to(device) 
    
    target = r + gamma * (1 - d) * (q - temp * log_prob)
    target = target.detach()
    
    loss_q1 = F.mse_loss(Q1(s, a), target)
    loss_q2 = F.mse_loss(Q2(s, a), target)

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
    clip_grad_norm_(actor.parameters(), clip_value)
    optim_a.step()
    
    writer.add_scalar("Loss/Critic", loss_q1.cpu().detach(), episode)
    writer.add_scalar("Loss/Actor", loss_a.cpu().detach(), episode)
    writer.add_scalar("Loss/entropy", -temp * log_prob.mean(), episode)
    
    soft_update(Q1_, Q1, tau=tau)
    soft_update(Q2_, Q2, tau=tau)


# Fill the replay buffer before training
global_step = 0
while global_step < start_steps:
    _, step = run_episode()
    global_step += step


# Train
for episode in tqdm(range(4000000)):   
    _, step = run_episode()
    global_step += step
    
    if episode % eval_freq == 0:
        eval_rewards, eval_steps = [], []
        for _ in range(num_eval_ep):
            episode_reward, episode_step = run_episode(test=True)
            eval_rewards.append(sum(episode_reward))
            eval_steps.append(episode_step)
            
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        avg_step = sum(eval_steps) / len(eval_steps)
            
        writer.add_scalar("Eval/reward", avg_reward, global_step)
        writer.add_scalar("Eval/step", avg_step, global_step)
            