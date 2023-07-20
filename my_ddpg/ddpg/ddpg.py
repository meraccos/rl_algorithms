import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Actor, Critic, ReplayBuffer, OUNoise


class DDPG:
    def __init__(self, env, lr, batch_size, gamma, max_episodes, 
                 eval_freq_ep, n_eval_episodes, updt_freq_st, training_starts=None):
        self.env = env
        self.actor = Actor(env)     # Optimized network
        self.actor_ = Actor(env)    # Target network
        self.critic = Critic(env)    
        self.critic_ = Critic(env) 
        self.soft_update(self.actor_, self.actor, 1.0)
        self.soft_update(self.critic_, self.critic, 1.0)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.rb = ReplayBuffer(length=200000, batch_size=batch_size)
        self.writer = SummaryWriter()
        self.gamma = gamma
        self.tau = 0.005
        self.noise = OUNoise(env.action_space)
        self.max_episodes = max_episodes
        self.eval_freq_ep = eval_freq_ep
        self.updt_freq_st = updt_freq_st
        self.n_eval_episodes = n_eval_episodes
        self.training_starts = training_starts or 10 * batch_size
        self.step = 0
        
    def optimize_critic(self):
        self.optimizer_critic.zero_grad()
        s, a, r, s_, d = self.rb.sample()
        q = self.critic(s, a.unsqueeze(1)).squeeze()
        a_ = self.actor_(s_).detach()
        q_ = self.critic_(s_, a_).squeeze().detach()
        target = r + self.gamma * (1 - d) * q_
        
        loss = F.mse_loss(q, target)
        loss.backward()
        self.clamp_gradient(self.critic)
        self.optimizer_critic.step()
        self.writer.add_scalar('Train/critic_loss', loss, self.step)
        
    def optimize_actor(self):
        self.optimizer_actor.zero_grad()
        s, _, _, _, _ = self.rb.sample()
        a = self.actor(s)
        q = self.critic(s, a)
        loss = -q.mean()
        loss.backward()
        self.clamp_gradient(self.actor)
        self.optimizer_actor.step()
        self.writer.add_scalar('Train/actor_loss', loss, self.step)

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + 
                                    (1.0 - tau) * target_param.data)
    
    def clamp_gradient(self, model):
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)    

    def run_episode(self, eval_mode=False):
        self.noise.reset()
        obs, _ = self.env.reset()
        done = False
        
        ep_len = 0
        ep_rew = 0.0
        while not done:
            action = self.actor(torch.tensor(obs).view(1, -1)).detach()
            # if not eval_mode:
            #     action = self.noise.get_action(action, self.step)
            if not eval_mode:
                noise = torch.Tensor(action.shape).data.normal_(0, 0.1)
                action = (action + noise).clamp(torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high))
            obs_, reward, terminated, truncated, _ = self.env.step(action.squeeze(0))
            done = terminated or truncated
            if not eval_mode and not truncated:
                action = action.squeeze(0)
                self.rb.store(obs, action, reward, obs_, terminated)
                self.step += 1
                
                train_starts = self.step >= self.training_starts
                if self.step % self.updt_freq_st == 0 and train_starts:
                    self.actor.train()
                    self.critic.train()
                    self.optimize_critic()
                    self.optimize_actor()
                    self.actor.eval()
                    self.critic.eval()
                    
                    self.soft_update(self.actor, self.actor_, self.tau)
                    self.soft_update(self.critic, self.critic_, self.tau)
                    
            obs = obs_
            ep_len += 1
            ep_rew += reward
        return ep_len, ep_rew

    def train(self):
        for episode in tqdm(range(self.max_episodes)):
            ep_len, ep_rew = self.run_episode(eval_mode=False) 
            self.writer.add_scalar('Train/length', ep_len, self.step)
            self.writer.add_scalar('Train/reward', ep_rew, self.step)
            if episode % self.eval_freq_ep == 0:
                    self.evaluate()
        
    def evaluate(self):
        rews = 0.0
        lens = 0.0
        for _ in range(self.n_eval_episodes):
            ep_len, ep_rew = self.run_episode(eval_mode=True)
            rews += ep_rew
            lens += ep_len
        self.writer.add_scalar('Eval/length', lens / self.n_eval_episodes, self.step)
        self.writer.add_scalar('Eval/reward', rews / self.n_eval_episodes, self.step)
    