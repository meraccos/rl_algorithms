import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
from tqdm import tqdm
from model import ReplayBuffer, Actor, Critic
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


class SAC:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter()
        
        # Hyperparameters
        self.gamma = 0.99
        self.lr_actor = 2e-4
        self.lr_critic = 3e-4
        self.eval_freq = 20
        self.num_eval_ep = 20
        self.start_steps = 10000
        self.clip_value = 2.5
        self.tau = 0.01
        self.learn_temp = False
        self.init_temp = torch.tensor(0.02)
            
        self.env = self.setup_environment()
        self.rb_size = 1000000
        self.batch_size = 256
        self.replay_buffer = self.initialize_replay_buffer()
        self.log_freq = 10
        self.save_freq = 200000
        self.global_step = 0
        self.episode = 0

        self.setup_models()

    def setup_environment(self):
        env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
        obs_high, obs_low = env.observation_space.high, env.observation_space.low
        obs_a = 2 / (obs_high - obs_low)
        obs_b = (obs_high + obs_low) / (obs_high - obs_low)
        return TransformObservation(env, lambda obs: obs * obs_a - obs_b)

    def setup_models(self):
        self.actor = Actor(self.env).to(self.device)
        self.Q1 = Critic(self.env).to(self.device)
        self.Q2 = Critic(self.env).to(self.device)
        self.Q1_ = Critic(self.env).to(self.device)
        self.Q2_ = Critic(self.env).to(self.device)
        self.Q1_.load_state_dict(self.Q1.state_dict())
        self.Q2_.load_state_dict(self.Q2.state_dict())
        
        self.optim_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.lr_critic)
        self.optim_q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.lr_critic)
        
        if self.learn_temp:
            self.temp_logit = self.init_temp
            self.temp_logit.requires_grad = True 
            self.temp = self.init_temp
            
            self.target_temp = self.env.action_space.shape[0]
            self.optim_temp = torch.optim.Adam([self.temp_logit])
        else:
            self.temp = self.init_temp            
    
    def initialize_replay_buffer(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        return ReplayBuffer(state_dim=state_dim, action_dim=action_dim, 
                            length=self.rb_size, batch_size=self.batch_size, 
                            device=self.device)

    def fill_replay_buffer(self):
        current_buffer_size = 0
        while current_buffer_size < self.start_steps:
            _, step = self.run_episode(random=True)
            current_buffer_size += step
    
    def soft_update(self):
        for target_p, source_p in zip(self.Q1_.parameters(), self.Q1.parameters()):
            target_p.data.copy_(self.tau * source_p.data + 
                                (1.0 - self.tau) * target_p.data)
            
        for target_p, source_p in zip(self.Q2_.parameters(), self.Q2.parameters()):
            target_p.data.copy_(self.tau * source_p.data + 
                                (1.0 - self.tau) * target_p.data)
    
    def save(self):
        file_name = f'models/{self.global_step // self.save_freq}'
        torch.save(self.actor.state_dict(), file_name +'_a.pt')
        torch.save(self.Q1.state_dict(), file_name +'_q1.pt')
        torch.save(self.Q2.state_dict(), file_name +'_q2.pt')
        torch.save(self.Q1_.state_dict(), file_name +'_q1_.pt')
        torch.save(self.Q2_.state_dict(), file_name +'_q2_.pt')
    
    def load(self, idx):
        file_name = f'models/{idx}'
        self.actor.load_state_dict(torch.load(file_name +'_a.pt'))
        self.Q1.load_state_dict(torch.load(file_name +'_q1.pt'))
        self.Q2.load_state_dict(torch.load(file_name +'_q2.pt'))
        self.Q1_.load_state_dict(torch.load(file_name +'_q1_.pt'))
        self.Q2_.load_state_dict(torch.load(file_name +'_q2_.pt'))

    def run_episode(self, test=False, random=False):
        obs, _ = self.env.reset()
        done = False
        rews_ep = []
        step = 0
        
        # Play the game and collect data
        while not done:
            if random:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.tensor(obs).unsqueeze(0).to(self.device)
                action = self.actor.sample(obs_tensor, test=test)
                action = action[0][0].to('cpu').detach().numpy()
        
            obs_, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.replay_buffer.store(obs, action, reward, obs_, terminated)
            obs = obs_
            step += 1

            rews_ep.append(reward)
            
            if not test:
                self.global_step += 1
                self.optimize()

            if self.global_step % self.save_freq == 0:
                self.save()
        return rews_ep, step

    def optimize(self):
        s, a, r, s_, d = self.replay_buffer.sample()
            
        # Optimize the critics
        a_theta, log_prob = self.actor.sample(s_)
        q = torch.min(self.Q1_(s_, a_theta), self.Q2_(s_, a_theta)).to(self.device) 
        
        target = r + self.gamma * (1 - d) * (q - self.temp * log_prob)
        target = target.detach()
        
        loss_q1 = F.mse_loss(self.Q1(s, a), target)
        loss_q2 = F.mse_loss(self.Q2(s, a), target)

        self.Q1.zero_grad()      
        loss_q1.backward()     
        clip_grad_norm_(self.Q1.parameters(), self.clip_value)
        self.optim_q1.step()        
        
        self.Q2.zero_grad()
        loss_q2.backward()
        clip_grad_norm_(self.Q2.parameters(), self.clip_value)
        self.optim_q2.step()
        
        # Optimize the actor
        a_theta, log_prob = self.actor.sample(s)
        q = torch.min(self.Q1(s, a_theta), self.Q2(s, a_theta)).to(self.device)
        
        loss_a = -q.mean() - self.temp * log_prob.mean()
        
        self.actor.zero_grad()
        loss_a.backward()
        clip_grad_norm_(self.actor.parameters(), self.clip_value)
        self.optim_a.step()
        
        # Optimize the temp
        if self.learn_temp:
            log_prob = self.actor.sample(s)[0].detach()
            self.temp = torch.exp(self.temp_logit)
            loss_t = self.temp_logit * (log_prob.mean() - self.target_temp) + self.temp
                        
            self.optim_temp.zero_grad()
            loss_t.backward()
            self.optim_temp.step()
             
            self.temp = self.temp.detach()
            self.writer.add_scalar("Loss/Temperature_loss", loss_t, self.global_step)

            
        self.soft_update()
        
        if self.global_step % self.log_freq == 0:
            self.writer.add_scalar("Loss/Critic", loss_q1.cpu().detach(), self.global_step)
            self.writer.add_scalar("Loss/Actor", loss_a.cpu().detach(), self.global_step)
            self.writer.add_scalar("Loss/Entropy", -log_prob.mean(), self.global_step)
            self.writer.add_scalar("Loss/Temperature", self.temp, self.global_step)
        
    def train(self, num_episodes):
        self.fill_replay_buffer()
        for episode in tqdm(range(num_episodes)):   
            _, step = self.run_episode()
            self.episode += 1
            
            if episode % self.eval_freq == 0:
                self.evaluate()
    
    def evaluate(self):
        eval_rewards, eval_steps = [], []
        for _ in range(self.num_eval_ep):
            episode_reward, episode_step = self.run_episode(test=True)
            eval_rewards.append(sum(episode_reward))
            eval_steps.append(episode_step)
            
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        avg_step = sum(eval_steps) / len(eval_steps)
            
        self.writer.add_scalar("Eval/reward", avg_reward, self.global_step)
        self.writer.add_scalar("Eval/length", avg_step, self.global_step)
            
            
if __name__ == "__main__":
    agent = SAC()
    agent.train(4000000)