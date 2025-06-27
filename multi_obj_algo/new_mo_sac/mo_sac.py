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
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
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
    def __init__(
        self,
        env,
        objectives,
        hidden_sizes=[256, 256],
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=100000,
        max_steps_per_episode=500,
        writer_filename='mosac_runs',
        verbose=False
    ):
        self.env = env
        self.objectives = objectives
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_steps = max_steps_per_episode
        self.verbose = verbose
        self.global_step = 0

        # Networks
        self.actor = MLP(self.obs_dim, self.act_dim * 2, hidden_sizes)
        self.critics = [MLP(self.obs_dim + self.act_dim, 1, hidden_sizes) for _ in range(objectives)]
        self.target_critics = [MLP(self.obs_dim + self.act_dim, 1, hidden_sizes) for _ in range(objectives)]

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critics_opt = [optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        # Copy target network weights
        for tc, c in zip(self.target_critics, self.critics):
            tc.load_state_dict(c.state_dict())

        # Replay buffer and TensorBoard writer
        self.buffer = ReplayBuffer(buffer_capacity)
        self.writer = SummaryWriter(writer_filename)

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs)
        mean, log_std = self.actor(obs_tensor).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        if deterministic:
            return mean.detach().numpy()

        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action).detach().numpy()

    def update(self, step):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(obs)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)

        # Compute target Q-values
        with torch.no_grad():
            next_action = torch.FloatTensor(self.select_action(next_obs))
            next_input = torch.cat([next_obs, next_action], dim=-1)
            target_q = []
            for idx, tc in enumerate(self.target_critics):
                q_next = tc(next_input)
                target_q.append(
                    rewards[:, idx].unsqueeze(-1)
                    + self.gamma * (1 - dones.unsqueeze(-1)) * q_next
                )

        # Critic update
        critic_losses = []
        for critic, critic_opt, target in zip(self.critics, self.critics_opt, target_q):
            critic_opt.zero_grad()
            q_value = critic(torch.cat([obs, actions], dim=-1))
            loss = F.mse_loss(q_value, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5)
            critic_opt.step()
            critic_losses.append(loss.item())

        # Actor update
        self.actor_opt.zero_grad()
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        sampled_action = torch.tanh(dist.rsample())
        actor_input = torch.cat([obs, sampled_action], dim=-1)
        q_vals = torch.stack([c(actor_input) for c in self.critics], dim=1).mean(dim=1)
        actor_loss = -(
            q_vals.mean() - self.alpha * dist.entropy().mean()
        )
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)
        self.actor_opt.step()

        # Soft update of target networks
        for tc, c in zip(self.target_critics, self.critics):
            for tp, p in zip(tc.parameters(), c.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

        # Log losses
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), step)
        for i, loss in enumerate(critic_losses):
            self.writer.add_scalar(f'Loss/Critic_{i}', loss, step)

    def _log_info(self, info, ep_idx):
        # Unrolled logging for ISO fields
        iso = info['iso']
        # Each field unconditionally logged
        self.writer.add_scalar('Info/iso/predicted_demands', np.mean(iso['predicted_demands']), ep_idx)
        self.writer.add_scalar('Info/iso/realized_demands', np.mean(iso['realized_demands']), ep_idx)
        self.writer.add_scalar('Info/iso/pcs_demands', np.mean(iso['pcs_demands']), ep_idx)
        self.writer.add_scalar('Info/iso/net_demands', np.mean(iso['net_demands']), ep_idx)
        self.writer.add_scalar('Info/iso/shortfalls', np.mean(iso['shortfalls']), ep_idx)
        self.writer.add_scalar('Info/iso/dispatch_costs', np.mean(iso['dispatch_costs']), ep_idx)
        self.writer.add_scalar('Info/iso/reserve_costs', np.mean(iso['reserve_costs']), ep_idx)
        self.writer.add_scalar('Info/iso/total_costs', np.mean(iso['total_costs']), ep_idx)
        self.writer.add_scalar('Info/iso/buy_prices', np.mean(iso['buy_prices']), ep_idx)
        self.writer.add_scalar('Info/iso/sell_prices', np.mean(iso['sell_prices']), ep_idx)
        self.writer.add_scalar('Info/iso/energy_bought', np.mean(iso['energy_bought']), ep_idx)
        self.writer.add_scalar('Info/iso/energy_sold', np.mean(iso['energy_sold']), ep_idx)
        self.writer.add_scalar('Info/iso/revenues', np.mean(iso['revenues']), ep_idx)
        self.writer.add_scalar('Info/iso/rewards', np.mean(iso['rewards']), ep_idx)
        total_iso_reward = iso['total_reward']
        self.writer.add_scalar('Info/iso/total_reward', np.mean(total_iso_reward) if isinstance(total_iso_reward, (list, np.ndarray)) else total_iso_reward, ep_idx)

        # Unrolled logging for PCS fields
        pcs = info['pcs']
        self.writer.add_scalar('Info/pcs/battery_levels', np.mean(pcs['battery_levels']), ep_idx)
        self.writer.add_scalar('Info/pcs/energy_exchanges', np.mean(pcs['energy_exchanges']), ep_idx)
        self.writer.add_scalar('Info/pcs/costs', np.mean(pcs['costs']), ep_idx)
        self.writer.add_scalar('Info/pcs/rewards', np.mean(pcs['rewards']), ep_idx)
        self.writer.add_scalar('Info/pcs/battery_utilization', np.mean(pcs['battery_utilization']), ep_idx)
        # total_reward possibly scalar
        self.writer.add_scalar('Info/pcs/total_reward', pcs['total_reward'], ep_idx)

    def train(self, num_episodes):
        step = 0
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            episode_rewards = np.zeros(self.objectives)
            last_info = None
            for t in range(self.max_steps):
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                last_info = info
                self.buffer.add((obs, action, reward, next_obs, terminated or truncated))
                episode_rewards += reward

                                # Raw per-step action logging
                if self.verbose:
                    # Log ISO action directly from the action variable
                    for comp, val in enumerate(action):
                        self.writer.add_scalar(
                            f'Raw/iso/action_comp{comp}',
                            val,
                            self.global_step
                        )
                    # Log PCS action from info dict if available
                    pcs_actions = info.get('pcs', {}).get('actions', [])
                    if len(pcs_actions) > t:
                        pcs_act = pcs_actions[t]
                        for comp, val in enumerate(pcs_act):
                            self.writer.add_scalar(
                                f'Raw/pcs/action_comp{comp}',
                                val,
                                self.global_step
                            )
                    self.global_step += 1

                obs = next_obs

                if len(self.buffer.buffer) > self.batch_size:
                    self.update(step)
                    step += 1

                if terminated or truncated:
                    break

            # Episode-level rewards
            for idx, r in enumerate(episode_rewards):
                self.writer.add_scalar(f'Reward/Objective_{idx}', r, ep)
            if self.verbose and last_info is not None:
                self._log_info(last_info, ep)

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

        return np.mean(rewards, axis=0)
