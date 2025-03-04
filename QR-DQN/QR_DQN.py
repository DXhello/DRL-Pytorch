import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class QRDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, n_quantiles):
        super(QRDQNNetwork, self).__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim * n_quantiles)
        )

    def forward(self, state):
        batch_size = state.size(0)
        quantiles = self.net(state).view(batch_size, self.action_dim, self.n_quantiles)
        return quantiles


class QRDQNAgent:
    def __init__(self, state_dim, action_dim, net_width, n_quantiles, gamma, lr, batch_size, dvc, exp_noise, **kwargs):
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.gamma = gamma
        self.batch_size = batch_size
        self.dvc = dvc
        self.exp_noise = exp_noise

        self.q_net = QRDQNNetwork(state_dim, action_dim, net_width, n_quantiles).to(dvc)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(state_dim, dvc, max_size=int(1e6))
        self.tau = 0.005

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
            if deterministic:
                quantiles = self.q_net(state)
                q_values = quantiles.mean(dim=2)
                action = q_values.argmax(dim=1).item()
            else:
                if np.random.rand() < self.exp_noise:
                    action = np.random.randint(0, self.action_dim)
                else:
                    quantiles = self.q_net(state)
                    q_values = quantiles.mean(dim=2)
                    action = q_values.argmax(dim=1).item()
        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_quantiles = self.q_target(next_states)
            next_q_values = next_quantiles.mean(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            next_quantiles = next_quantiles[range(self.batch_size), next_actions]
            target_quantiles = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_quantiles

        quantiles = self.q_net(states)
        current_quantiles = quantiles[range(self.batch_size), actions.squeeze()]

        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        huber_loss = F.smooth_l1_loss(td_errors, torch.zeros_like(td_errors), reduction='none')
        quantile_huber_loss = (torch.abs(self.cumulative_density - (td_errors < 0).float()) * huber_loss).mean(dim=2).sum(dim=1).mean()

        self.q_net_optimizer.zero_grad()
        quantile_huber_loss.backward()
        self.q_net_optimizer.step()

        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, algo, env_name, steps):
        torch.save(self.q_net.state_dict(), f"./model/{algo}_{env_name}_{steps}.pth")

    def load(self, algo, env_name, steps):
        self.q_net.load_state_dict(torch.load(f"./model/{algo}_{env_name}_{steps}.pth", map_location=self.dvc))
        self.q_target.load_state_dict(torch.load(f"./model/{algo}_{env_name}_{steps}.pth", map_location=self.dvc))


class ReplayBuffer:
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((max_size, state_dim), dtype=torch.float, device=dvc)
        self.actions = torch.zeros((max_size, 1), dtype=torch.long, device=dvc)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float, device=dvc)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float, device=dvc)
        self.dones = torch.zeros((max_size, 1), dtype=torch.bool, device=dvc)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = torch.from_numpy(state).to(self.dvc)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(self.dvc)
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.dvc)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices]
