import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

def objective(trial):
    # Load data
    data = np.load("data/full_data.npy")
    
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_int("batch_size", 32, 256, log=True)
    discount = trial.suggest_float("discount", 0.9, 0.99)
    tau = trial.suggest_float("tau", 1e-4, 1e-2, log=True)
    exploration_phase = trial.suggest_int("exploration_phase", 200, 1000)
    
    # Kalman filter hyperparameters
    observation_covariance = trial.suggest_float("observation_covariance", 0.1, 10.0, log=True)
    transition_covariance = trial.suggest_float("transition_covariance", 0.01, 1.0, log=True)
    
    # Fixed parameters
    num_stocks = data.shape[1]
    initial_cash = 100_000
    max_steps = data.shape[0]
    num_episodes = 1300
    
    # Chaotic Feature Extractor setup
    chaotic_extractor = ChaoticFeatureExtractor()
    all_chaotic_features = chaotic_extractor.extract_features(data)
    chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
    
    # Kalman filter application
    filtered_data = apply_kalman_filter(
        data,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    
    # Environment setup
    env = StockEnv(num_stocks=num_stocks, data=filtered_data, initial_cash=initial_cash)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # TD3 Agent setup
    max_action = 1.0
    agent = TD3(
        state_dim=state_dim,
        chaotic_feature_dim=chaotic_feature_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        max_action=1.0,
        env_action_space_high=1.0,
        env_action_space_low=0.0
    )
    
    agent.exploration_phase = exploration_phase
    
    # Logging
    reward_history = []
    critic_loss_history = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_critic_loss = 0  # Track critic loss
        
        for step in range(max_steps):
            # Select action
            chaotic_features = all_chaotic_features[:, step, :].flatten()
            
            action = agent.select_action(
                state=state,
                chaotic_features=chaotic_features,
                current_episode=episode
            )
            
            # Step in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Track critic loss
            critic_loss, _ = agent.train(batch_size=batch_size, discount=discount, tau=tau)
            total_critic_loss += critic_loss
            
            next_chaotic_features = all_chaotic_features[:, step + 1, :].flatten() if step + 1 < max_steps else chaotic_features
            
            # Add transition to replay buffer
            agent.replay_buffer.add(
                (state, chaotic_features, action, reward, next_state, next_chaotic_features, done)
            )
            
            state = next_state
            
            if done:
                break
        
        # Append the reward and critic loss for this episode
        reward_history.append(total_reward)
        critic_loss_history.append(total_critic_loss)
        
        # Report intermediate results to Optuna
        if episode % 10 == 0 and episode > 0:
            avg_reward = np.mean(reward_history[-10:])
            avg_critic_loss = np.mean(critic_loss_history[-10:])
            # Define a weighted objective (balance between reward and critic loss)
            alpha = 0.8  # Weight for the reward
            beta = 0.2   # Weight for the critic loss
            combined_metric = alpha * avg_reward - beta * avg_critic_loss
            
            trial.report(combined_metric, episode)
            
            # Pruning (early stopping of unpromising trials)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            avg_critic_loss = np.mean(critic_loss_history[-10:]) if len(critic_loss_history) >= 10 else np.mean(critic_loss_history)
            print(f"Trial {trial.number}, Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Critic Loss = {avg_critic_loss:.2f}")
    
    # Return combined objective: average reward - critic loss
    final_reward = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
    final_critic_loss = np.mean(critic_loss_history[-50:]) if len(critic_loss_history) >= 50 else np.mean(critic_loss_history)
    final_combined_metric = alpha * final_reward - beta * final_critic_loss
    
    return final_combined_metric


def run_optimization(n_trials=50, study_name=None, storage=None, load_if_exists=True):
    # Create study name if not provided
    if study_name is None:
        study_name = f"td3_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print and save results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters to file
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best parameters
    best_params_file = os.path.join(results_dir, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(trial.params, f, indent=2)
    
    # Generate optimization visualization plots
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(results_dir, f"{study_name}_history.png"))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(results_dir, f"{study_name}_importance.png"))
    
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(os.path.join(results_dir, f"{study_name}_contour.png"))
    
    return study.best_params

def train_with_best_params(best_params):
    """Train a model with the best parameters found by Optuna"""
    # Load data
    data = np.load("data/full_data.npy")
    
    # Kalman filter hyperparameters
    observation_covariance = best_params["observation_covariance"]
    transition_covariance = best_params["transition_covariance"]
    
    num_stocks = data.shape[1]
    initial_cash = 100_000
    max_steps = data.shape[0]
    num_episodes = best_params["num_episodes"]
    
    # Chaotic Feature Extractor setup
    chaotic_extractor = ChaoticFeatureExtractor()
    all_chaotic_features = chaotic_extractor.extract_features(data)
    chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
    
    # Kalman filter application
    filtered_data = apply_kalman_filter(
        data,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    
    # Environment setup
    env = StockEnv(num_stocks=num_stocks, data=filtered_data, initial_cash=initial_cash)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # TD3 Agent setup with the best parameters
    agent = TD3(
        state_dim=state_dim,
        chaotic_feature_dim=chaotic_feature_dim,
        action_dim=action_dim,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        num_stocks=num_stocks,
        max_action=1.0,
        env_action_space_high=1.0,
        env_action_space_low=0.0
    )
    
    agent.exploration_phase = best_params["exploration_phase"]
    
    # Training loop
    # Logging
    reward_history = []
    avg_reward_history = []
    critic_loss_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_critic_loss = []


        for step in range(max_steps):
            # Select action
            chaotic_features = all_chaotic_features[:, step, :].flatten() # (N*F*C)
            
            action = agent.select_action(
                state=state,
                chaotic_features=chaotic_features,
                current_episode=episode
            )

            # print(f"Actions at step {step}: {action}")

            # Step in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            next_chaotic_features = all_chaotic_features[:, step + 1, :].flatten() if step + 1 < max_steps else chaotic_features  # Handle last step case

            # Add transition to replay buffer
            agent.replay_buffer.add(
                (state, chaotic_features, action, reward, next_state,next_chaotic_features, done)
            )

            # Train the agent
            critic_loss,_ = agent.train(batch_size=best_params['batch_size'], discount=best_params['discount'], tau=best_params['tau'])
            episode_critic_loss.append(critic_loss)

            state = next_state

            if done:
                print(f"🚨 Episode {episode} ended early at step {step} due to termination condition.")

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

    torch.save(agent, 'td3_agent_tuned.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(critic_loss_history, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("TD3 Critic Loss Over Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    storage = None
    
    # Run the optimization
    best_params = run_optimization(
        n_trials=50, 
        storage=storage
    )
    
    train_with_best_params(best_params)