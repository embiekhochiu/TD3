import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

data = np.load("data/test_data.npy")

agent = torch.load('td3_agent.pth')
agent.exploration_phase = 0

test_env = StockEnv(num_stocks=data.shape[1], data=data, initial_cash=100_000)

# Chaotic Feature Extractor setup
chaotic_extractor = ChaoticFeatureExtractor()
all_chaotic_features = chaotic_extractor.extract_features(data)  # Extract chaotic features
chaotic_feature_dim = chaotic_extractor.output_dim * data.shape[1]

# Kalman filter setup
data = apply_kalman_filter(data,observation_covariance=1.0,transition_covariance=0.1)

# Testing configuration
num_test_episodes = 10
max_steps = data.shape[0]

# Log the test performance
test_rewards = []

# Testing loop
for episode in range(num_test_episodes):
    state = test_env.reset()  
    done = False

    for step in range(max_steps):
        chaotic_features = all_chaotic_features[:, step, :].flatten()  
        
        # Select action using the actor network (no noise added during testing)
        action = agent.select_action(state=state, chaotic_features=chaotic_features, current_episode=episode)
        
        # Step in environment
        next_state, reward, done, info = test_env.step(action)
        total_reward += reward
        
        # Update state
        state = next_state

        if done:
            print(f"Test Episode {episode + 1} ended early at step {step}.")
            break
    
    # Record the total reward for this test episode
    test_rewards.append(total_reward)
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