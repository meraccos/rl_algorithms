import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd
from torch.distributions.normal import Normal

import numpy as np
import math


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
class Actor(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.shape[0]
    
        self.l1 = nn.Linear(self.input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, self.output_dim * 2)
        
        initialize_weights(self)

    def forward(self, x):
        a = F.leaky_relu(self.l1(x), 0.1)
        a = F.leaky_relu((self.l2(a)), 0.1)
        mu, std = self.l3(a).chunk(2, dim=-1)
        std = torch.sigmoid(std)
        std = 7.0 * std + 1e-8

        return mu, std
    
    def dist(self, x):
        mu, std = self(x)
        n = Normal(mu, std)
        # n = SquashedNormal(mu, std)
        return n
        
    def sample(self, x, deterministic = False):
        if deterministic:
            return self(x)[0].tanh(), 0
        dist = self.dist(x)
        action = dist.rsample().tanh()
        logp = (dist.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)).sum(1)
        
        return action, logp
    
    def logp(self, x, action):
        dist = self.dist(x)
        logp = (dist.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)).sum(1)
        return logp
                    
    
class Critic(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.input_dim = env.observation_space.shape[0]
        
        self.l1 = nn.Linear(self.input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)
        
        initialize_weights(self)

    def forward(self, x):
        x = F.leaky_relu((self.l1(x)), 0.1)
        x = F.leaky_relu((self.l2(x)), 0.1)
        x = self.l3(x)
        return x.squeeze()

class Buffer:
    '''
    '''
    def __init__(self, device='cpu'):
        self.device = device
        self.dtype = torch.float32
        self.reset_all()
    
    def store(self, s, a, lp, r, s_, d):
        '''
        Stores the returns from each step into the episode memory
        '''
        self.states.append(s)
        self.actions.append(a)
        self.logp.append(lp)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(int(d))
    
    def store_adv_and_rets(self, advantages, returns):
        self.advantages += advantages
        self.returns += returns
            
    def load_ep(self):
        '''
        Returns s, a, lp, r, s_, d of the last finished episode
        '''
        states = torch.tensor(np.array(self.states), device=self.device, dtype=self.dtype)
        actions = torch.tensor(np.array(self.actions), device=self.device, dtype=self.dtype)
        logp = torch.tensor(np.array(self.logp), device=self.device, dtype=self.dtype)
        rewards = torch.tensor(np.array(self.rewards), device=self.device, dtype=self.dtype)
        states_ = torch.tensor(np.array(self.states_), device=self.device, dtype=self.dtype)
        dones = torch.tensor(np.array(self.dones), device=self.device, dtype=self.dtype)
        advs = torch.tensor(np.array(self.advantages), device=self.device, dtype=self.dtype)
        rets = torch.tensor(np.array(self.returns), device=self.device, dtype=self.dtype)
        
        return states, actions, logp, rewards, states_, dones, advs, rets

    def load_state_and_rewards(self):
        '''
        Returns s, r of the last finished episode
        '''
        states = torch.tensor(np.array(self.states), device=self.device, dtype=self.dtype)
        rewards = torch.tensor(np.array(self.rewards), device=self.device, dtype=self.dtype)
        
        return states, rewards
    
    def load_all(self, batch_size=None, ith=0):
        '''
        Returns logs of all the episodes in the batch
        '''
        size_frac = (self.all_actions.shape[0] // batch_size)
        ith = ith % size_frac
        indices = self.shuffled_indices[ith*batch_size:(ith+1)*batch_size]

        return (self.all_states[indices], 
                self.all_actions[indices], 
                self.all_logp.squeeze()[indices], 
                self.all_rewards.squeeze()[indices], 
                self.all_states_[indices], 
                self.all_dones.squeeze()[indices],
                self.all_advs.squeeze()[indices],
                self.all_returns.squeeze()[indices])
  
    def discounted_rewards(self, r, gamma):
        '''
        Returns the discounted rewards based on the input array of rewards, r
        '''
        t_steps = torch.arange(len(r)).to(self.device)
        r_gamma = r * gamma ** t_steps
        r_disc = reversed(reversed(r_gamma).cumsum(0)) / gamma ** t_steps
        return r_disc
    
    def save_ep(self):
        '''
        Loads all the episodic buffer to the batch buffer. 
        Call at the end of each episode.
        '''
        s, a, lp, r, s_, d, adv, ret = self.load_ep() 
        self.all_states = torch.vstack((self.all_states, s))
        self.all_actions = torch.vstack((self.all_actions, a))
        self.all_logp = torch.vstack((self.all_logp, lp))
        self.all_rewards = torch.vstack((self.all_rewards, r.view(-1, 1)))
        self.all_states_ = torch.vstack((self.all_states_, s_))
        self.all_dones = torch.vstack((self.all_dones, d.view(-1, 1)))
        self.all_advs = torch.vstack((self.all_advs, adv.view(-1, 1)))
        self.all_returns = torch.vstack((self.all_returns, ret.view(-1, 1)))
        
        self.shuffled_indices = torch.randperm(self.all_states.shape[0])
    
    def reset_ep(self):     
        '''
        Erases the episodic buffer. 
        Call at the beginning of each episode.
        '''   
        self.states = []
        self.actions = []
        self.logp = []
        self.rewards = []
        self.states_ = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def reset_all(self):
        '''
        Erases both the episodic buffer and batch buffer. 
        Call after one optimization session is done.
        '''
        self.reset_ep()
        
        # Batch of episodes memory, concated
        self.all_states = torch.empty((0, 8), device=self.device, dtype=self.dtype)
        self.all_actions = torch.empty((0, 2), device=self.device, dtype=self.dtype)
        self.all_logp = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.all_rewards = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.all_states_ = torch.empty((0, 8), device=self.device, dtype=self.dtype)
        self.all_dones = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.all_advs = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.all_returns = torch.empty((0, 1), device=self.device, dtype=self.dtype)



# class Buffer:
#     def __init__(self, device='cuda'):
#         self.device = device
#         self.dtype = torch.float32
#         self.reset()

#     def reset(self):     
#         self.states = []
#         self.actions = []
#         self.logp = []
#         self.rewards = []
#         self.states_ = []
#         self.dones = []
#         self.advantages = []
        
#     def store(self, s, a, lp, r, s_, d):
#         self.states.append(s)
#         self.actions.append(a)
#         self.logp.append(lp)
#         self.rewards.append(r)
#         self.states_.append(s_)
#         self.dones.append(int(d))
    
#     def store_adv(self, adv):
#         self.advantages.append(adv)

#     def load_all(self):
#         states = torch.tensor(np.array(self.states), device=self.device, dtype=self.dtype)
#         actions = torch.tensor(np.array(self.actions), device=self.device, dtype=self.dtype)
#         logp = torch.tensor(np.array(self.logp), device=self.device, dtype=self.dtype)
#         rewards = torch.tensor(np.array(self.rewards), device=self.device, dtype=self.dtype)
#         states_ = torch.tensor(np.array(self.states_), device=self.device, dtype=self.dtype)
#         dones = torch.tensor(np.array(self.dones), device=self.device, dtype=self.dtype)
#         advantages = torch.tensor(np.hstack(self.advantages), device=self.device, dtype=self.dtype)
        
#         return states, actions, logp.squeeze(), rewards, states_, dones, advantages

#     def get_batch(self, batch_size):
#         s, a, lp, r, s_, d, advantages = self.load_all() 
#         init_idx = random.randint(0, s.shape[0] - batch_size)
#         end_idx = init_idx + batch_size
        
#         return (s[init_idx:end_idx],
#                 a[init_idx:end_idx],
#                 lp[init_idx:end_idx],
#                 r[init_idx:end_idx],
#                 s_[init_idx:end_idx],
#                 d[init_idx:end_idx],
#                 advantages[init_idx:end_idx])

