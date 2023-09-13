import torch
import torch.nn.functional as F
import gym
from gym.wrappers import TransformObservation
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
        self.temp = 0.01
        self.eval_freq = 20
        self.num_eval_ep = 20
        self.start_steps = 5000
        self.clip_value = 2.5
        self.tau = 0.0001
    
        self.env = self.setup_environment()
        self.actor, self.critics, self.optimizers = self.setup_models()
        self.replay_buffer = self.initialize_replay_buffer()
        self.global_step = 0

    def setup_environment(self):
        env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
        obs_high, obs_low = env.observation_space.high, env.observation_space.low
        obs_a = 2 / (obs_high - obs_low)
        obs_b = (obs_high + obs_low) / (obs_high - obs_low)
        return TransformObservation(env, lambda obs: obs * obs_a - obs_b)

    def setup_models(self):
        actor = Actor(self.env).to(self.device)
        Q1 = Critic(self.env).to(self.device)
        Q2 = Critic(self.env).to(self.device)
        Q1_ = Critic(self.env).to(self.device)
        Q2_ = Critic(self.env).to(self.device)
        Q1_.load_state_dict(Q1.state_dict())
        Q2_.load_state_dict(Q2.state_dict())
        
        optim_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
        optim_q1 = torch.optim.Adam(Q1.parameters(), lr=3e-4)
        optim_q2 = torch.optim.Adam(Q2.parameters(), lr=3e-4)
        
        return actor, (Q1, Q2, Q1_, Q2_), (optim_a, optim_q1, optim_q2)
    
    def initialize_replay_buffer(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        return ReplayBuffer(state_dim=state_dim, action_dim=action_dim, 
                            length=1000000, batch_size=512, device=self.device)

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
                self.optimize(self.global_step)
        return rews_ep, step


    def optimize(self, episode):
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
        
        self.writer.add_scalar("Loss/Critic", loss_q1.cpu().detach(), episode)
        self.writer.add_scalar("Loss/Actor", loss_a.cpu().detach(), episode)
        self.writer.add_scalar("Loss/entropy", -self.temp * log_prob.mean(), episode)
        
        self.soft_update(self.Q1_, self.Q1, tau=self.tau)
        self.soft_update(self.Q2_, self.Q2, tau=self.tau)

    def train(self, num_episodes):
        self.fill_replay_buffer()
        for episode in tqdm(range(num_episodes)):   
            _, step = self.run_episode()
            
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
        self.writer.add_scalar("Eval/step", avg_step, self.global_step)
            
            
if __name__ == "__main__":
    agent = SAC()
    agent.train(4000000)