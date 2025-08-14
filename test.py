import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

data = np.load("data/test_processed_data.npy")
data = data[:95]  # Limit to the first 1000 timesteps for testing

# Load the trained agent
agent = torch.load('td3_agent_sequence.pth')  
agent.exploration_phase = 0  # No exploration during testing

# Define the lookback window (same as in training)
lookback_window = 5

num_stocks = data.shape[1]

# Chaotic Feature Extractor setup
chaotic_extractor = ChaoticFeatureExtractor()
all_chaotic_features = chaotic_extractor.extract_features(data)  # Extract chaotic features
chaotic_feature_dim = chaotic_extractor.output_dim * data.shape[1]

# Kalman filter setup
data = apply_kalman_filter(data, observation_covariance=1.0, transition_covariance=0.1)

# Environment setup
test_env = StockEnv(num_stocks=num_stocks, data=data, initial_cash=100_000)

# Testing configuration
num_test_episodes = 2
max_steps = data.shape[0]

# Log the test performance
test_rewards = []
portfolio_values = []

# Testing loop
for episode in range(num_test_episodes):
    state = test_env.reset()
    done = False
    total_reward = 0
    episode_portfolio_values = [test_env.portfolio_value]  # Track portfolio value over time
    
    for step in range(max_steps - lookback_window + 1):
        # Extract state sequence and portfolio state
        current_state_sequence = state[:num_stocks * lookback_window * (data.shape[-1])].reshape(lookback_window, num_stocks * (data.shape[-1]))
        portfolio_state = state[num_stocks * lookback_window * (data.shape[-1]):]
        
        # Get sequence of chaotic features
        chaotic_features_sequence = all_chaotic_features[:, step:step + lookback_window, :].reshape(lookback_window, -1)
        
        # Debug information
        print(f"Chaotic features type: {type(chaotic_features_sequence)}")
        print(f"Chaotic features dtype: {chaotic_features_sequence.dtype}")
        print(f"Chaotic features shape: {chaotic_features_sequence.shape}")
        
        # Convert object array to float32 if needed
        if chaotic_features_sequence.dtype == np.dtype('O'):
            print("Converting object array to float32...")
            chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)
        
        # Select action using the actor network
        action = agent.select_action(
            state=current_state_sequence,
            chaotic_features=chaotic_features_sequence,
            current_episode=episode + agent.exploration_phase
        )
        
            
            # Step in environment
        next_state, reward, done, info = test_env.step(action)
        total_reward += reward
        
        # Update state
        state = next_state
        
        # Track portfolio value
        episode_portfolio_values.append(test_env.portfolio_value)
        
        if done:
            print(f"Test Episode {episode + 1} ended early at step {step + lookback_window}.")
            break
    
    # Record the total reward for this test episode
    test_rewards.append(total_reward)
    portfolio_values.append(episode_portfolio_values)
    print(f"Test Episode {episode + 1}/{num_test_episodes} - Total Reward: {total_reward:.2f}")

# Compute the average reward over all test episodes
average_test_reward = np.mean(test_rewards)
print(f"Average Test Reward: {average_test_reward:.2f}")

# Plot test performance
plt.figure(figsize=(10, 5))
plt.plot(test_rewards, label="Total Reward")
plt.xlabel("Test Episode")
plt.ylabel("Total Reward")
plt.title("TD3 Test Rewards")
plt.legend()
plt.show()

# Plot portfolio value over time for the best episode
best_episode = np.argmax(test_rewards)
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values[best_episode], label=f"Episode {best_episode + 1}")
plt.xlabel("Trading Step")
plt.ylabel("Portfolio Value ($)")
plt.title("TD3 Test - Portfolio Value Over Time (Best Episode)")
plt.legend()
plt.grid(True)
plt.show()

# Plot portfolio value over time for all episodes
plt.figure(figsize=(12, 6))
for i, values in enumerate(portfolio_values):
    plt.plot(values, label=f"Episode {i + 1}")
plt.xlabel("Trading Step")
plt.ylabel("Portfolio Value ($)")
plt.title("TD3 Test - Portfolio Value Over Time (All Episodes)")
plt.legend()
plt.grid(True)
plt.show()