import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

data = np.load("data/train_processed_data.npy")

# Training configuration

num_stocks = data.shape[1]
initial_cash = 100_000
num_episodes = 1100
max_steps = data.shape[0]
batch_size = 64
discount = 0.99
tau = 1e-3
exploration_phase = 600
lookback_window = 5  # Define the lookback window


# Chaotic Feature Extractor setup
chaotic_extractor = ChaoticFeatureExtractor()
all_chaotic_features = chaotic_extractor.extract_features(data)  # Extract chaotic features
chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks

# Kalman filter setup
data = apply_kalman_filter(data,observation_covariance=1.0,transition_covariance=0.1)

# Environment setup
env = StockEnv(num_stocks=num_stocks, data=data, initial_cash=initial_cash)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# TD3 Agent setup
max_action = 1.0
agent = TD3(
    state_dim=num_stocks * (data.shape[-1]),  # Size of features per timestep
    chaotic_feature_dim=chaotic_feature_dim, # Size of chaotic features per timestep
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
    state = env.reset()
    total_reward = 0
    episode_critic_loss = []


    for step in range(max_steps - lookback_window + 1): # Adjust step range
        # Select action
        start_index = max(0, step)
        end_index = step + lookback_window
        current_state_sequence = state[:num_stocks * lookback_window * (data.shape[-1])].reshape(lookback_window, num_stocks * (data.shape[-1])) # Extract price/indicator features
        portfolio_state = state[num_stocks * lookback_window * (data.shape[-1]):] # Extract portfolio state

        chaotic_features_sequence = all_chaotic_features[:, step:step + lookback_window, :].reshape(lookback_window, -1) # Get sequence of chaotic features

        if chaotic_features_sequence.dtype == np.dtype('O'):
            # print("Converting object array to float32...")
            chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)

        action = agent.select_action(
            state=current_state_sequence, # Pass the sequence
            chaotic_features=chaotic_features_sequence, # Pass the sequence
            current_episode=episode
        )

        # Step in environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # Extract next state sequence (handle end of episode)
        next_state_sequence = next_state[:num_stocks * lookback_window * (data.shape[-1])].reshape(lookback_window, num_stocks * (data.shape[-1]))
        next_chaotic_features_sequence = all_chaotic_features[:, step + 1:step + lookback_window + 1, :].reshape(lookback_window, -1) if step + 1 < max_steps - lookback_window + 1 else chaotic_features_sequence # Handle last step

        # Add transition to replay buffer
        agent.replay_buffer.add(
            (current_state_sequence, chaotic_features_sequence, action, reward, next_state_sequence, next_chaotic_features_sequence, done)
        )

        # Train the agent
        critic_loss,_ = agent.train(batch_size=batch_size, discount=discount, tau=tau)
        episode_critic_loss.append(critic_loss)

        state = next_state

        if done:
            print(f"🚨 Episode {episode} ended early at step {step + lookback_window} due to termination condition.")
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

torch.save(agent, 'td3_agent_sequence.pth')

plt.figure(figsize=(10, 5))
plt.plot(critic_loss_history, label="Critic Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("TD3 Critic Loss Over Time")
plt.legend()
plt.show()
