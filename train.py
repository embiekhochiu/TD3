import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor


data = np.load("data/full_data.npy")

# Training configuration

num_stocks = data.shape[1]
initial_cash = 100_000
num_episodes = 500
max_steps = 100
batch_size = 64
discount = 0.99
tau = 0.005
exploration_phase = 365

# Environment setup
env = StockEnv(num_stocks=num_stocks, data=data, initial_cash=initial_cash)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Chaotic Feature Extractor setup
chaotic_extractor = ChaoticFeatureExtractor()
all_chaotic_features = chaotic_extractor.extract_features(data)  # Extract chaotic features
chaotic_feature_dim = chaotic_extractor.output_dim

# TD3 Agent setup
max_action = 1.0
agent = TD3(
    state_dim=state_dim,
    chaotic_feature_dim=chaotic_feature_dim,
    portfolio_dim=env.num_stocks,
    action_dim=action_dim,
    hidden_size=256,
    num_layers=2,
    num_stocks=num_stocks,
    max_action=1.0,
    env_action_space_high=1.0,
    env_action_space_low=0.0
)

agent.exploration_phase = exploration_phase

# Logging
reward_history = []
avg_reward_history = []
critic_loss_history = []

# Training loop
for episode in range(num_episodes):
    state, portfolio_state = env.reset()
    total_reward = 0
    episode_critic_loss = []


    for step in range(max_steps):
        # Select action
        chaotic_features = all_chaotic_features[:, step, :] # (N, F*C)
        
        action = agent.select_action(
            state=state,
            chaotic_features=chaotic_features,
            portfolio_state=portfolio_state,
            current_episode=episode
        )

        # Step in environment
        next_state, reward, done, info = env.step(action)
        next_portfolio_state = info["portfolio_state"]
        total_reward += reward

        # Add transition to replay buffer
        agent.replay_buffer.add(
            (state, chaotic_features, portfolio_state, action, reward, next_state, next_portfolio_state, done)
        )

        # Train the agent
        critic_loss = agent.train(batch_size=batch_size, discount=discount, tau=tau)
        episode_critic_loss.append(critic_loss)

        state = next_state
        portfolio_state = next_portfolio_state

        if done:
            break

    # Logging
    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-10:])  # Moving average over last 10 episodes
    avg_reward_history.append(avg_reward)
    avg_critic_loss = np.mean(episode_critic_loss)
    critic_loss_history.append(avg_critic_loss)

    print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, Avg Critic Loss = {avg_critic_loss:.4f}")

# Plot rewards
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Total Reward")
plt.plot(avg_reward_history, label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("TD3 Training Rewards")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(critic_loss_history, label="Critic Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("TD3 Critic Loss Over Time")
plt.legend()
plt.show()
