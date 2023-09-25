import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers import NormalizeReward
from gymnasium.experimental.wrappers import RecordEpisodeStatisticsV0

from tqdm import tqdm
import numpy as np

from model import Actor, Critic, Buffer

class PPO:
    def __init__(self):
        self.env = self.get_env()
        self.writer = SummaryWriter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = Buffer(self.device)

        self.gamma = 0.99
        self.lam = 0.97
        self.temp = 0.02
        self.eps_clip = 0.1
        self.buffer_size = 4096
        self.mini_batch = 128
        batches_in_buffer = self.buffer_size // self.mini_batch
        self.num_epoch = batches_in_buffer * 8
        self.step = 0
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.get_models()
        
    def get_env(self):
        env = gym.make("LunarLanderContinuous-v2")       
        obs_high, obs_low = env.observation_space.high, env.observation_space.low
        obs_a = 2 / (obs_high - obs_low)
        obs_b = (obs_high + obs_low) / (obs_high - obs_low)
        env = TransformObservation(env, lambda obs: obs * obs_a - obs_b)
        env = NormalizeObservation(env)
        env = RecordEpisodeStatisticsV0(env)
        env = NormalizeReward(env)

        return env
    
    def get_models(self):
        self.actor = Actor(self.env).to(self.device)
        self.critic = Critic(self.env).to(self.device)
        self.optim_a = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=0.0001)
        self.optim_v = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=0.0001)
    
    def normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)

    @torch.no_grad()
    def compute_gae(self, rewards, values, terminal_value=0.0):
        values = torch.cat([values, torch.tensor([terminal_value], device=self.device)])
        deltas = rewards + self.gamma * values[1:] - values[:-1]

        # Compute GAE advantages
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lam * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        # returns = self.buffer.discounted_rewards(rewards, self.gamma)
        return advantages, returns
        
    def optimize(self):
        actor_losses = []
        critic_losses = []
        kl_divs = []
        for epoch in range(self.num_epoch):
            s, a, lp, r, s_, d, adv, returns = self.buffer.load_all(self.mini_batch, epoch)
            advantages = self.normalize(adv)
            
            # Actor loss
            logp = self.actor.logp(s, a)
            ratios = torch.exp(logp - lp)   
            kl_div = (logp - lp).mean().cpu().detach()
            if abs(kl_div) > 0.1:
                print('early stopping at ', epoch)
                break
            surr1 = ratios * advantages            
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages  
            loss_policy = -torch.min(surr1, surr2).mean() - self.temp * logp.mean()

            loss_value = F.mse_loss(self.critic(s), returns)                           
            
            # Actor Network
            self.optim_a.zero_grad()
            loss_policy.backward()
            self.optim_a.step()

            # Critic network
            self.optim_v.zero_grad()
            loss_value.backward()
            self.optim_v.step()   
            
            actor_losses.append(loss_policy.cpu().detach())
            critic_losses.append(loss_value.cpu().detach()) 
            kl_divs.append(kl_div)

        self.writer.add_scalar('Train/critic_loss', sum(critic_losses)/self.num_epoch, self.step)
        self.writer.add_scalar('Train/actor_loss', sum(actor_losses)/self.num_epoch, self.step)

        self.writer.add_scalar('Train/kl_div', sum(kl_divs)/self.num_epoch, self.step)
    
    def _get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, logp = self.actor.sample(obs)
        action = action.detach().cpu().numpy()[0]
        logp = logp.cpu().detach()
        return action, logp
    
    def _save_advantages(self, states, rews, terminal_value=0.0):
        values = self.critic(states).detach()
        advs, returns = self.compute_gae(rews, values, terminal_value)
        
        self.buffer.store_adv_and_rets(advs.cpu().tolist(), returns.cpu().tolist())
    
    @torch.no_grad()
    def collect_data(self):
        obs, _ = self.env.reset()

        while True:
            action, logp = self._get_action(obs)
            obs_, reward, terminated, truncated, info = self.env.step(action)
            obs_ = np.clip(obs_, -5.0, 5.0)

            done = terminated or truncated
            
            self.buffer.store(obs, action, logp, reward, obs_, done)
            self.step += 1
            obs = obs_

            if done:
                s, r= self.buffer.load_state_and_rewards() 
                self._save_advantages(s, r)           
                self.buffer.save_ep()
                self.buffer.reset_ep()
                obs, _ = self.env.reset() 
                self.writer.add_scalar('Train/rews', info['episode']['r'], self.step)
                self.writer.add_scalar('Train/length', info['episode']['l'], self.step) 
                
            if self.step % self.buffer_size == 0:
                s, r= self.buffer.load_state_and_rewards()                
                if s.shape[0] > 1:
                    future_reward = self.critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)).item()
                    self._save_advantages(s, r, terminal_value=future_reward)
                    self.buffer.save_ep()
                    
                self.buffer.reset_ep()
                break

    def train(self, num_episodes=100000):
        for episode in tqdm(range(num_episodes)):    
            # Collect data
            self.collect_data()
            self.optimize()
            self.buffer.reset_all()
            
if __name__ == "__main__":
    ppo = PPO()
    ppo.train(100000)
    