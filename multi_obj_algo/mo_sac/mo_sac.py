import mo_gymnasium  # Ensure multi-objective envs are registered
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, weight_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.weight = np.zeros((max_size, weight_dim), dtype=np.float32)

    def add(self, s, a, r, s2, d, w):
        self.state[self.ptr] = s
        self.next_state[self.ptr] = s2
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.done[self.ptr] = d
        self.weight[self.ptr] = w
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idx], dtype=torch.float32),
            torch.as_tensor(self.action[idx], dtype=torch.float32),
            torch.as_tensor(self.reward[idx], dtype=torch.float32),
            torch.as_tensor(self.next_state[idx], dtype=torch.float32),
            torch.as_tensor(self.done[idx], dtype=torch.float32),
            torch.as_tensor(self.weight[idx], dtype=torch.float32),
        )

# Simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256,256), activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Actor with tanh-squashed Gaussian and correct log-prob
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim, max_action, log_std_bounds=(-20,2)):
        super().__init__()
        self.net = MLP(state_dim + weight_dim, 2 * action_dim)
        self.max_action = max_action
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, state, weight):
        x = torch.cat([state, weight], dim=-1)
        mean_logstd = self.net(x)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.exp()

    def sample(self, state, weight):
        mean, std = self(state, weight)
        dist = Normal(mean, std)
        z = dist.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z * self.max_action
        log_prob_z = dist.log_prob(z).sum(-1, keepdim=True)
        log_det = torch.log(self.max_action * (1 - tanh_z.pow(2)) + 1e-6).sum(-1, keepdim=True)
        logp = log_prob_z - log_det
        return action, logp

# Critic networks
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim + weight_dim, 1)
        self.q2 = MLP(state_dim + action_dim + weight_dim, 1)

    def forward(self, state, action, weight):
        x = torch.cat([state, action, weight], dim=-1)
        return self.q1(x), self.q2(x)

# Pareto-Conditioned Soft Actor-Critic
class MOSAC:
    def __init__(self, state_dim, action_dim, weight_dim, max_action, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_dim = weight_dim
        self.device = device

        self.actor = Actor(state_dim, action_dim, weight_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, weight_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, weight_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, weight, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state, weight)
            return (torch.tanh(mean) * self.actor.max_action).detach().cpu().numpy()[0]
        action, _ = self.actor.sample(state, weight)
        return action.detach().cpu().numpy()[0]

    def train_step(self, buffer, batch_size, gamma=0.99, tau=0.005):
        s, a, rvec, s2, done, w = buffer.sample(batch_size)
        s, a, s2, done, w = [t.to(self.device) for t in (s, a, s2, done, w)]
        rvec = rvec.to(self.device)

        r = (rvec * w).sum(-1, keepdim=True)
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2, w)
            q1_t, q2_t = self.critic_target(s2, a2, w)
            q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            q_target = r + (1 - done) * gamma * q_t

        q1, q2 = self.critic(s, a, w)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_new, logp_new = self.actor.sample(s, w)
        q1_n, q2_n = self.critic(s, a_new, w)
        qn = torch.min(q1_n, q2_n)
        actor_loss = (self.alpha * logp_new - qn).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def train(self, env_name, env_kwargs=None, episodes=500, max_steps=1000, batch_size=256):
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
                done = term or trunc
                rvec = np.array(rvec, dtype=np.float32)

                buffer.add(s, a, rvec, s2, float(done), w)
                s = s2
                ep_r += rvec
                ep_rs += np.dot(w, rvec)

                if buffer.size > batch_size:
                    c_l, a_l, al_l = self.train_step(buffer, batch_size)
                if done:
                    break

            for i, val in enumerate(ep_r):
                writer.add_scalar(f"Return/obj{i+1}", val, ep)
            writer.add_scalar("Return/scalar", ep_rs, ep)
            writer.add_scalar("Loss/critic", c_l, ep)
            writer.add_scalar("Loss/actor", a_l, ep)
            writer.add_scalar("Loss/alpha", al_l, ep)
            print(f"Episode {ep}: scalar={ep_rs:.2f}, rewards={ep_r}")

        writer.close()