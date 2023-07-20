import torch
import torch.nn as nn

from collections import deque
from tqdm import tqdm


class DQN:
    def __init__(self, env, model, model_, loss_fn, optimizer, replay_buffer, writer, 
                 batch_size, gamma, max_episodes, eval_freq_ep, n_eval_episodes, 
                 updt_freq_st, eps_s, eps_e, eps_decay_steps, train_starts):
        self.env = env
        self.model = model
        self.model_ = model_
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.rb = replay_buffer
        self.writer = writer
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.eval_freq_ep = eval_freq_ep
        self.updt_freq_st = updt_freq_st
        self.n_eval_episodes = n_eval_episodes
        self.step = 0
        self.eps_s = eps_s
        self.eps_e = eps_e
        self.eps_slope = (eps_e - eps_s) / eps_decay_steps
        self.training_starts = train_starts
        
        self.opt_step = 0
        
    def optimize(self):
        self.opt_step += 1
        s, a, r, s_, d = self.rb.sample()
        qs = self.model(s)
        qs_ = self.model_(s_)
        qs_max = torch.max(qs_, dim=1)[0]
        target = r + self.gamma * qs_max * (1 - d)
        actual = qs[range(self.batch_size), a.tolist()]

        self.optimizer.zero_grad()
        loss = self.loss_fn(actual, target)
        loss.backward()
        # Clamp the gradient
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.writer.add_scalar('Train/loss', loss, self.step)
        
        if self.opt_step % 5 == 0:
            self.soft_update(self.model, self.model_, tau=0.85)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def linear_schedule(self):
        eps_ = self.eps_s + self.eps_slope * self.step
        eps = max(self.eps_e, eps_)
        self.writer.add_scalar('Train/epsilon', eps, self.step)
        return eps
    
    def select_action(self, obs, eval_mode):
        eps = 0.0 if eval_mode else self.linear_schedule()
        rand_act = self.env.action_space.sample()
        best_act = torch.argmax(self.model(torch.Tensor(obs))).item()
        action = rand_act if (torch.rand(1) < eps) else best_act   
        return action
    
    def run_episode(self, eval_mode=False):
        obs, _ = self.env.reset()
        done = False
        
        ep_len = 0
        ep_rew = 0.0
        while not done:
            action = self.select_action(obs, eval_mode)
            obs_, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if not eval_mode and not truncated:
                self.rb.store(obs, action, reward, obs_, terminated)
                self.step += 1
                
                train_starts = self.step >= self.training_starts
                if self.step % self.updt_freq_st == 0 and train_starts:
                    self.optimize()
                    
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
                    self.eval()
        
    def eval(self):
        rews = 0.0
        lens = 0.0
        for _ in range(self.n_eval_episodes):
            ep_len, ep_rew = self.run_episode(eval_mode=True)
            rews += ep_rew
            lens += ep_len
        self.writer.add_scalar('Eval/length', lens / self.n_eval_episodes, self.step)
        self.writer.add_scalar('Eval/reward', rews / self.n_eval_episodes, self.step)
        

class QNetwork(nn.Module):
    def __init__(self, env=None, input_dim=None, output_dim=None):
        super().__init__()
        input_dim = input_dim or env.observation_space.shape[0]
        output_dim = output_dim or env.action_space.n
        h = 256
        self.model = nn.Sequential(
            nn.Linear(input_dim, h), 
            nn.ReLU(),
            nn.Linear(h, h), 
            nn.ReLU(),
            nn.Linear(h, output_dim)
        )
        # Explicit initialization
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.model(x)
    

class ReplayBuffer:
    def __init__(self, length=10000, batch_size = 32):
        self.mem = deque(maxlen=length)
        self.length = length
        self.batch_size = batch_size
    
    def store(self, s, a, r, s_, d):
        self.mem.append([s, a, r, s_, d])
    
    def sample(self):
        idxs = torch.randint(high=len(self), size=[self.batch_size])
        batch = [self.mem[idx] for idx in idxs]
        return map(torch.Tensor, list(zip(*batch)))
        
    def __len__(self):
        return len(self.mem)
    