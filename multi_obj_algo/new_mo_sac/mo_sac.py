import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256], activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_sizes
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))


class MOSAC:
    def __init__(self, env, objectives, hidden_sizes=[256, 256], actor_lr=3e-4, critic_lr=3e-4,
                 alpha=0.2, gamma=0.99, tau=0.005, batch_size=256, buffer_capacity=100000,
                 max_steps_per_episode=500, writer_filename='mosac_runs'):
        self.env = env
        self.objectives = objectives
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_steps = max_steps_per_episode

        self.actor = MLP(self.obs_dim, self.act_dim * 2, hidden_sizes)
        self.critics = [MLP(self.obs_dim + self.act_dim, 1, hidden_sizes) for _ in range(objectives)]
        self.target_critics = [MLP(self.obs_dim + self.act_dim, 1, hidden_sizes) for _ in range(objectives)]

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critics_opt = [optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        for tc, c in zip(self.target_critics, self.critics):
            tc.load_state_dict(c.state_dict())

        self.buffer = ReplayBuffer(buffer_capacity)
        self.writer = SummaryWriter(writer_filename)

    def select_action(self, obs, deterministic=False):
        mean_logstd = self.actor(torch.FloatTensor(obs))
        mean, log_std = mean_logstd.chunk(2, dim=-1)
        std = log_std.exp()

        if deterministic:
            return mean.detach().numpy()

        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        return torch.tanh(action).detach().numpy()

    def update(self, step):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(obs)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            next_action = torch.FloatTensor(self.select_action(next_obs))
            next_input = torch.cat([next_obs, next_action], dim=-1)


            target_q = []
            for idx, tc in enumerate(self.target_critics):
                q_next = tc(next_input)
                target_q.append(rewards[:, idx].unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * q_next)

        critic_losses = []
        for idx, (critic, critic_opt, target) in enumerate(zip(self.critics, self.critics_opt, target_q)):
            critic_opt.zero_grad()
            q_value = critic(torch.cat([obs, actions], dim=-1))
            critic_loss = F.mse_loss(q_value, target)
            critic_loss.backward()
            critic_opt.step()
            critic_losses.append(critic_loss.item())

        self.actor_opt.zero_grad()
        sampled_action = self.select_action(obs)
        actor_input = torch.cat([obs, torch.FloatTensor(sampled_action)], dim=-1)
        q_values = torch.stack([critic(actor_input) for critic in self.critics], dim=1).mean(dim=1)
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        std = log_std.exp()  # Ensuring positivity
        dist = torch.distributions.Normal(mean, std)
        actor_loss = -(q_values.mean() - self.alpha * dist.entropy().mean())
        actor_loss.backward()
        self.actor_opt.step()

        for tc, c in zip(self.target_critics, self.critics):
            for tp, p in zip(tc.parameters(), c.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

        self.writer.add_scalar('Loss/Actor', actor_loss.item(), step)
        for i, loss in enumerate(critic_losses):
            self.writer.add_scalar(f'Loss/Critic_{i}', loss, step)

    def train(self, num_episodes):
        step = 0
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            episode_rewards = np.zeros(self.objectives)
            for _ in range(self.max_steps):
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.buffer.add((obs, action, reward, next_obs, terminated or truncated))
                episode_rewards += reward
                obs = next_obs

                if len(self.buffer.buffer) > self.batch_size:
                    self.update(step)
                    step += 1

                if terminated or truncated:
                    break

            for idx, r in enumerate(episode_rewards):
                self.writer.add_scalar(f'Reward/Objective_{idx}', r, ep)

    def evaluate(self, episodes=10):
        rewards = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            ep_reward = np.zeros(self.objectives)
            for _ in range(self.max_steps):
                action = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)

        avg_rewards = np.mean(rewards, axis=0)
        return avg_rewards
