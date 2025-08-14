import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, chaotic_feature_dim, hidden_size, num_layers, num_stocks):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(state_dim + chaotic_feature_dim, hidden_size, num_layers, batch_first=True)
        self.fc_portfolio = nn.Linear(hidden_size, hidden_size)
        self.fc_selection = nn.Linear(hidden_size, num_stocks)  # Stock selection
        self.fc_allocation = nn.Linear(hidden_size, num_stocks)  # Cash allocation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, chaotic_features, hidden=None):
        # state shape: (batch_size, lookback_window, num_stocks * num_features)
        # chaotic_features shape: (batch_size, lookback_window, chaotic_feature_dim_per_stock * num_stocks)

        # Concatenate state and chaotic features along the feature dimension
        combined_features = torch.cat([state, chaotic_features], dim=-1)

        lstm_out, hidden = self.lstm(combined_features, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep

        fc_out = torch.relu(self.fc_portfolio(lstm_out))
        stock_selection = self.sigmoid(self.fc_selection(fc_out))
        allocation = self.softmax(self.fc_allocation(fc_out))
        allocation = allocation * stock_selection
        allocation = allocation / allocation.sum(dim=1, keepdim=True)
        return stock_selection, allocation, hidden

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, chaotic_feature_dim, action_dim, hidden_size, num_layers):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(state_dim + chaotic_feature_dim, hidden_size, num_layers, batch_first=True)
        self.fc_concat = nn.Linear(hidden_size + action_dim, 256)
        self.fc_q1 = nn.Linear(256, 256)
        self.fc_q2 = nn.Linear(256, 256)
        self.fc_q1_out = nn.Linear(256, 1)
        self.fc_q2_out = nn.Linear(256, 1)

    def forward(self, state, chaotic_features, action, hidden=None):
        # state shape: (batch_size, lookback_window, num_stocks * num_features)
        # chaotic_features shape: (batch_size, lookback_window, chaotic_feature_dim_per_stock * num_stocks)
        # action shape: (batch_size, action_dim)

        # Concatenate state and chaotic features along the feature dimension
        combined_features = torch.cat([state, chaotic_features], dim=-1)

        lstm_out, hidden = self.lstm(combined_features, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep

        combined = torch.cat([lstm_out, action], dim=1)
        concat_out = torch.relu(self.fc_concat(combined))
        q1 = torch.relu(self.fc_q1(concat_out))
        q1 = self.fc_q1_out(q1)
        q2 = torch.relu(self.fc_q2(concat_out))
        q2 = self.fc_q2_out(q2)
        return q1, q2

    def Q1(self, state, chaotic_features, action, hidden=None):
        combined_features = torch.cat([state, chaotic_features], dim=-1)
        lstm_out, hidden = self.lstm(combined_features, hidden)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat([lstm_out, action], dim=1)
        concat_out = torch.relu(self.fc_concat(combined))
        q1 = torch.relu(self.fc_q1(concat_out))
        return self.fc_q1_out(q1)

# Replay Buffer (remains largely the same, but will store sequences)
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, chaotic_features, actions, rewards, next_states, next_chaotic_features, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.stack(chaotic_features).astype(np.float32), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.stack(next_chaotic_features).astype(np.float32), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

# TD3 Algorithm
class TD3:
    def __init__(self, state_dim, chaotic_feature_dim, action_dim, hidden_size, num_layers, num_stocks, max_action, env_action_space_high, env_action_space_low):
        self.exploration_phase = 600  # Number of episodes for chaotic exploration
        self.actor = Actor(state_dim, chaotic_feature_dim, hidden_size, num_layers, num_stocks)
        self.actor_target = Actor(state_dim, chaotic_feature_dim, hidden_size, num_layers, num_stocks)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, chaotic_feature_dim, action_dim, hidden_size, num_layers)
        self.critic_target = Critic(state_dim, chaotic_feature_dim, action_dim, hidden_size, num_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(size=100000)

        self.max_action = max_action
        self.env_action_space_high = env_action_space_high
        self.env_action_space_low = env_action_space_low

        self.policy_noise = 0.2  # Noise standard deviation
        self.noise_clip = 0.5  # Noise clipping
        self.policy_delay = 3  # Delayed policy updates
        self.total_it = 0

        # Chaotic noise parameters
        self.chaotic_map_state = np.random.rand()  # Initial state for the chaotic map

    def chaotic_noise(self, scale=0.01):
        """Logistic map to generate chaotic noise."""
        r = 3.99  # Chaos parameter
        self.chaotic_map_state = r * self.chaotic_map_state * (1 - self.chaotic_map_state)
        return scale * (self.chaotic_map_state - 0.5)

    def select_action(self, state, chaotic_features, current_episode=0, hidden=None):
        """Select action with or without chaotic noise during exploration."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        chaotic_features = torch.tensor(chaotic_features, dtype=torch.float32).unsqueeze(0) # Add batch dimension

        # Generate actions using the actor network
        stock_selection, allocation, _ = self.actor(state, chaotic_features, hidden)
        actions = torch.cat([stock_selection, allocation], dim=1).detach().numpy().flatten()

        # Add chaotic noise during exploration phase
        if current_episode < self.exploration_phase:
            chaotic = np.array([self.chaotic_noise(scale=0.01) for _ in range(actions.shape[0])])
            gaussian_noise = np.random.normal(0, 0.1, size=actions.shape)  # More variance
            actions = np.clip(actions + chaotic + gaussian_noise, self.env_action_space_low, self.env_action_space_high)

        # Clip actions to valid range
        return np.clip(actions, self.env_action_space_low, self.env_action_space_high)

    def train(self, batch_size=64, discount=0.99, tau=1e-3):
        if len(self.replay_buffer.buffer) < batch_size:
            return 0.0, 0.0

        # Sample from the replay buffer
        states, chaotic_features, actions, rewards, next_states, next_chaotic_features, dones = self.replay_buffer.sample(batch_size)

        # Compute target Q-values
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            # Generate next actions using the target actor
            next_stock_selection, next_allocation, _ = self.actor_target(next_states, next_chaotic_features)
            next_actions = torch.cat([next_stock_selection, next_allocation], dim=1) + noise
            next_actions = next_actions.clamp(self.env_action_space_low, self.env_action_space_high)

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_chaotic_features, next_actions)
            target_q = rewards + discount * (1 - dones) * torch.min(target_q1, target_q2)

            if torch.isnan(target_q).any():
                print("Warning: NaN detected in target_q!")

        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, chaotic_features, actions)

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(current_q2, target_q)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0)

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            stock_selection, allocation, _ = self.actor(states, chaotic_features)
            predicted_actions = torch.cat([stock_selection, allocation], dim=1)
            actor_loss = -self.critic.Q1(states, chaotic_features, predicted_actions).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.total_it += 1
        return critic_loss.item(), actor_loss.item()