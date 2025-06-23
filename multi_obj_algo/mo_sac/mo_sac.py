import mo_gymnasium  # Ensure multi-objective envs are registered
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

# Replay buffer for multi-objective transitions
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, weight_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, weight_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.weight = np.zeros((max_size, weight_dim), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, weight):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.weight[self.ptr] = weight

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.state[idx], dtype=torch.float32),
            torch.tensor(self.action[idx], dtype=torch.float32),
            torch.tensor(self.reward[idx], dtype=torch.float32),
            torch.tensor(self.next_state[idx], dtype=torch.float32),
            torch.tensor(self.done[idx], dtype=torch.float32),
            torch.tensor(self.weight[idx], dtype=torch.float32),
        )

# Actor network: outputs gaussian policy conditioned on state and weight
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim, max_action):
        super().__init__()
        self.max_action = max_action
        inp_dim = state_dim + weight_dim
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state, weight):
        x = torch.cat([state, weight], dim=-1)
        h = self.net(x)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        return mean, log_std

    def sample(self, state, weight):
        mean, log_std = self.forward(state, weight)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_action
        # log prob with Tanh correction
        log_prob = dist.log_prob(z) - torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

# Critic network: two Q-functions
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim):
        super().__init__()
        inp_dim = state_dim + action_dim + weight_dim
        def build_q():
            return nn.Sequential(
                nn.Linear(inp_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, 1),
            )
        self.q1 = build_q()
        self.q2 = build_q()

    def forward(self, state, action, weight):
        x = torch.cat([state, action, weight], dim=-1)
        return self.q1(x), self.q2(x)

# MOSAC agent with hyper-parameters, replay warm-up, truncation handling, and gradient clipping
class MOSAC:
    def __init__(
        self, state_dim, action_dim, weight_dim, max_action, device,
        actor_lr=1e-4, critic_lr=1e-3, alpha_lr=3e-4,
        gamma=0.98, tau=0.01, target_entropy=None,
        batch_size=256, gradient_steps=4, warmup_multiplier=10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_dim = weight_dim
        self.device = device

        # Hyper-parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy is not None else -0.5 * float(action_dim)
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.min_buffer_size = int(batch_size * warmup_multiplier)

        # Networks
        self.actor = Actor(state_dim, action_dim, weight_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, weight_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, weight_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, weight, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        weight_t = torch.tensor(weight, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state_t, weight_t)
            action = torch.tanh(mean) * self.actor.max_action
            return action.cpu().detach().numpy()[0]
        action, _ = self.actor.sample(state_t, weight_t)
        return action.cpu().detach().numpy()[0]

    def train_step(self, buffer, batch_size):
        s, a, rvec, s2, done, w = buffer.sample(batch_size)
        s, a, s2, done, w = [t.to(self.device) for t in (s, a, s2, done, w)]
        rvec = rvec.to(self.device)

        # scalarize
        r = (rvec * w).sum(-1, keepdim=True)
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2, w)
            q1_t, q2_t = self.critic_target(s2, a2, w)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            q_target = r + (1 - done) * self.gamma * q_t

        # critic update
        q1, q2 = self.critic(s, a, w)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn_utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # actor update
        a_new, logp_new = self.actor.sample(s, w)
        q1_n, q2_n = self.critic(s, a_new, w)
        qn = torch.min(q1_n, q2_n)
        actor_loss = (self.alpha * logp_new - qn).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn_utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # soft target update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def train(self, env_name, env_kwargs=None, episodes=500, max_steps=1000):
        env_kwargs = env_kwargs or {}
        env = gym.make(env_name, disable_env_checker=True, **env_kwargs)
        buffer = ReplayBuffer(self.state_dim, self.action_dim, self.weight_dim)
        writer = SummaryWriter()

        for ep in range(1, episodes + 1):
            c_l = a_l = al_l = 0.0
            w = np.random.dirichlet(np.ones(self.weight_dim))
            s, _ = env.reset()
            ep_r = np.zeros(self.weight_dim)
            ep_rs = 0.0

            for _ in range(max_steps):
                a = self.select_action(s, w)
                s2, rvec, term, trunc, _ = env.step(a)
                real_done = bool(term or trunc)
                done_flag = float(term)
                rvec = np.array(rvec, dtype=np.float32)

                buffer.add(s, a, rvec, s2, done_flag, w)
                s = s2; ep_r += rvec; ep_rs += np.dot(w, rvec)

                if buffer.size > self.min_buffer_size:
                    for _ in range(self.gradient_steps):
                        c_l, a_l, al_l = self.train_step(buffer, self.batch_size)

                if real_done:
                    break

            for i, val in enumerate(ep_r): writer.add_scalar(f"Return/obj{i+1}", val, ep)
            writer.add_scalar("Return/scalar", ep_rs, ep)
            writer.add_scalar("Loss/critic", c_l, ep)
            writer.add_scalar("Loss/actor", a_l, ep)
            writer.add_scalar("Loss/alpha", al_l, ep)
            print(f"Episode {ep}: scalar={ep_rs:.2f}, rewards={ep_r}")

        writer.close()
