import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

# Replay buffer for multi-objective transitions, including preference vector
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
            torch.as_tensor(self.state[idx]),
            torch.as_tensor(self.action[idx]),
            torch.as_tensor(self.reward[idx]),
            torch.as_tensor(self.next_state[idx]),
            torch.as_tensor(self.done[idx]),
            torch.as_tensor(self.weight[idx]),
        )

# MLP for actor and critic
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256,256), activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = (input_dim, *hidden_dims)
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Actor network conditioned on preference vector w
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim, max_action):
        super().__init__()
        self.net = MLP(state_dim + weight_dim, 2 * action_dim)
        self.max_action = max_action
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state, weight):
        x = torch.cat([state, weight], dim=-1)
        mean_logstd = self.net(x)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        return mean, std

    def sample(self, state, weight):
        mean, std = self(state, weight)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

# Critic Q-network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim + weight_dim, 1)
        self.q2 = MLP(state_dim + action_dim + weight_dim, 1)

    def forward(self, state, action, weight):
        x = torch.cat([state, action, weight], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

# Pareto-Conditioned SAC Agent
class MOSAC:
    def __init__(self, state_dim, action_dim, weight_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, weight_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, weight_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, weight_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Entropy temperature
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, weight, evaluate=False):
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        weight = torch.as_tensor(weight, device=self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state, weight)
            action = torch.tanh(mean) * self.actor.max_action
            return action.cpu().detach().numpy()[0]
        action, _ = self.actor.sample(state, weight)
        return action.cpu().detach().numpy()[0]

    def train_step(self, replay_buffer, batch_size, gamma=0.99, tau=0.005):
        state, action, reward_vec, next_state, done, weight = replay_buffer.sample(batch_size)
        state, action = state.to(self.device), action.to(self.device)
        next_state, done = next_state.to(self.device), done.to(self.device)
        reward_vec, weight = reward_vec.to(self.device), weight.to(self.device)

        # Scalarize reward
        reward = (reward_vec * weight).sum(-1, keepdim=True)

        # Critic loss
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state, weight)
            q1_t, q2_t = self.critic_target(next_state, next_action, weight)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = reward + (1 - done) * gamma * q_t

        q1, q2 = self.critic(state, action, weight)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor loss
        action_new, logp_new = self.actor.sample(state, weight)
        q1_new, q2_new = self.critic(state, action_new, weight)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * logp_new - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

# Training loop with Gymnasium API
def train(env_name='YourMOEnv-v0', episodes=500, max_steps=1000, batch_size=256):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    # You can override weight_dim if your env doesn't have reward_space attribute
    weight_dim = env.reward_space.shape[0] if hasattr(env, 'reward_space') else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = MOSAC(state_dim, action_dim, weight_dim, max_action, device)
    buffer = ReplayBuffer(state_dim, action_dim, weight_dim)
    writer = SummaryWriter()

    total_steps = 0
    for ep in range(1, episodes + 1):
        w = np.random.dirichlet(np.ones(weight_dim))
        state, _ = env.reset()
        ep_rewards = np.zeros(weight_dim)
        ep_scalar = 0.0

        for step in range(1, max_steps + 1):
            action = agent.select_action(state, w)
            next_state, reward_vec, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_vec = np.array(reward_vec, dtype=np.float32)

            buffer.add(state, action, reward_vec, next_state, float(done), w)

            state = next_state
            ep_rewards += reward_vec
            ep_scalar += np.dot(w, reward_vec)
            total_steps += 1

            if buffer.size > batch_size:
                c_loss, a_loss, alpha_loss = agent.train_step(buffer, batch_size)

            if done:
                break

        # Logging
        for i, r in enumerate(ep_rewards):
            writer.add_scalar(f'Returns/objective_{i+1}', r, ep)
        writer.add_scalar('Returns/scalarized', ep_scalar, ep)
        writer.add_scalar('Loss/critic', c_loss, ep)
        writer.add_scalar('Loss/actor', a_loss, ep)
        writer.add_scalar('Loss/alpha', alpha_loss, ep)

        print(f'Episode {ep}: scalar_return={ep_scalar:.2f}, objectives={ep_rewards}')

    writer.close()

if __name__ == '__main__':
    train()
