import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

def objective(trial):
    # Load data
    data = np.load("TD3/data/train_processed_data.npy")
    
    # Fixed parameters
    num_stocks = data.shape[1]
    initial_cash = 100_000
    max_steps = data.shape[0]
    
    # Parameters to optimize
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 384, 512])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    discount = trial.suggest_float("discount", 0.95, 0.995, step=0.005)
    tau = trial.suggest_float("tau", 5e-4, 5e-3, log=True)
    exploration_phase = trial.suggest_int("exploration_phase", 100, 300, step=50)
    lookback_window = 15
    
    # Kalman filter parameters
    observation_covariance = trial.suggest_float("observation_covariance", 0.1, 3.0, log=True)
    transition_covariance = trial.suggest_float("transition_covariance", 0.01, 0.5, log=True)
    
    # For hyperparameter tuning, we'll use fewer episodes
    num_episodes = 400
    
    # Apply Kalman filter
    filtered_data = apply_kalman_filter(
        data,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    
    # Chaotic Feature Extractor setup
    chaotic_extractor = ChaoticFeatureExtractor()
    all_chaotic_features = chaotic_extractor.extract_features(filtered_data)
    chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
    
    # Environment setup
    env = StockEnv(num_stocks=num_stocks, data=filtered_data, initial_cash=initial_cash)
    action_dim = env.action_space.shape[0]
    
    # TD3 Agent setup
    agent = TD3(
        state_dim=num_stocks * (filtered_data.shape[-1]),
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
        episode_critic_loss = []
        
        for step in range(max_steps - lookback_window + 1):
            # Extract state sequence and portfolio state
            current_state_sequence = state[:num_stocks * lookback_window * (filtered_data.shape[-1])].reshape(
                lookback_window, num_stocks * (filtered_data.shape[-1]))
            portfolio_state = state[num_stocks * lookback_window * (filtered_data.shape[-1]):]
            
            # Get chaotic features sequence
            chaotic_features_sequence = all_chaotic_features[:, step:step + lookback_window, :].reshape(lookback_window, -1)
            
            # Convert to float32 if needed
            if chaotic_features_sequence.dtype == np.dtype('O'):
                chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)
            
            # Select action
            action = agent.select_action(
                state=current_state_sequence,
                chaotic_features=chaotic_features_sequence,
                current_episode=episode
            )
            
            # Step in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Extract next state sequence
            next_state_sequence = next_state[:num_stocks * lookback_window * (filtered_data.shape[-1])].reshape(
                lookback_window, num_stocks * (filtered_data.shape[-1]))
            
            # Handle last step case for chaotic features
            next_chaotic_features_sequence = all_chaotic_features[:, step + 1:step + lookback_window + 1, :].reshape(
                lookback_window, -1) if step + 1 < max_steps - lookback_window + 1 else chaotic_features_sequence
            
            # Add transition to replay buffer
            agent.replay_buffer.add(
                (current_state_sequence, chaotic_features_sequence, action, reward, 
                 next_state_sequence, next_chaotic_features_sequence, done)
            )
            
            # Train the agent
            critic_loss, _ = agent.train(batch_size=batch_size, discount=discount, tau=tau)
            episode_critic_loss.append(critic_loss)
            
            state = next_state
            
            if done:
                break
        
        # Track rewards and losses
        reward_history.append(total_reward)
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0
        critic_loss_history.append(avg_critic_loss)
        
        # Calculate moving average reward
        window_size = min(10, len(reward_history))
        avg_reward = np.mean(reward_history[-window_size:])
        
        # Report to Optuna every 10 episodes
        if episode % 10 == 0:
            # Weight reward higher than critic loss for optimization
            alpha = 0.8  # Weight for reward
            beta = 0.2   # Weight for critic loss
            
            # Use weighted metric, normalizing critic loss for scale
            norm_factor = 100.0  # Scale factor for critic loss
            metric = alpha * avg_reward - beta * (avg_critic_loss / norm_factor)
            
            trial.report(metric, episode)
            
            # Pruning (early stopping of unpromising trials)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            print(f"Trial {trial.number}, Episode {episode}: "
                  f"Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, "
                  f"Critic Loss = {avg_critic_loss:.4f}")
    
    # Return final score (last 50 episodes or all episodes if fewer)
    final_window = min(50, len(reward_history))
    final_reward = np.mean(reward_history[-final_window:])
    final_critic_loss = np.mean(critic_loss_history[-final_window:])
    
    # Final combined metric
    final_metric = alpha * final_reward - beta * (final_critic_loss / norm_factor)
    
    return final_metric


def run_optimization(n_trials=30):
    """Run the hyperparameter optimization study"""
    # Create study name and results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"td3_stock_trading_{timestamp}"
    results_dir = "optuna_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30)
    )
    
    # Run optimization
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print and save results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    best_params = trial.params.copy()
    best_params["num_episodes"] = 500  # Set full training episodes for final model
    
    best_params_file = os.path.join(results_dir, f"best_params_{timestamp}.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest parameters saved to {best_params_file}")
    
    # Generate visualization plots
    try:
        # Optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"optimization_history_{timestamp}.png"))
        
        # Parameter importance
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"param_importance_{timestamp}.png"))
        
        # Parallel coordinate plot
        plt.figure(figsize=(12, 6))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"parallel_coordinate_{timestamp}.png"))
        
        plt.close('all')
    except Exception as e:
        print(f"Error generating visualization: {e}")
    
    return best_params


def train_with_best_params(params):
    """Train the final model using the best parameters found"""
    print("\nTraining final model with best parameters...")
    
    # Load data
    data = np.load("data/train_processed_data.npy")
    
    # Fixed parameters
    num_stocks = data.shape[1]
    initial_cash = 100_000
    max_steps = data.shape[0]
    num_episodes = params.get("num_episodes", 300)
    
    # Apply Kalman filter with optimized parameters
    filtered_data = apply_kalman_filter(
        data,
        observation_covariance=params["observation_covariance"],
        transition_covariance=params["transition_covariance"]
    )
    
    # Get lookback window
    lookback_window = 15
    
    # Chaotic Feature Extractor setup
    chaotic_extractor = ChaoticFeatureExtractor()
    all_chaotic_features = chaotic_extractor.extract_features(filtered_data)
    chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
    
    # Environment setup
    env = StockEnv(num_stocks=num_stocks, data=filtered_data, initial_cash=initial_cash)
    action_dim = env.action_space.shape[0]
    
    # TD3 Agent setup
    agent = TD3(
        state_dim=num_stocks * (filtered_data.shape[-1]),
        chaotic_feature_dim=chaotic_feature_dim,
        action_dim=action_dim,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        num_stocks=num_stocks,
        max_action=1.0,
        env_action_space_high=1.0,
        env_action_space_low=0.0
    )
    
    agent.exploration_phase = params["exploration_phase"]
    
    # Logging
    reward_history = []
    avg_reward_history = []
    critic_loss_history = []
    
    # Training loop (similar to original train.py)
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_critic_loss = []
        
        for step in range(max_steps - lookback_window + 1):
            # Extract state sequence and portfolio state
            current_state_sequence = state[:num_stocks * lookback_window * (filtered_data.shape[-1])].reshape(
                lookback_window, num_stocks * (filtered_data.shape[-1]))
            portfolio_state = state[num_stocks * lookback_window * (filtered_data.shape[-1]):]
            
            # Get chaotic features sequence
            chaotic_features_sequence = all_chaotic_features[:, step:step + lookback_window, :].reshape(lookback_window, -1)
            
            # Convert to float32 if needed
            if chaotic_features_sequence.dtype == np.dtype('O'):
                chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)
            
            # Select action
            action = agent.select_action(
                state=current_state_sequence,
                chaotic_features=chaotic_features_sequence,
                current_episode=episode
            )
            
            # Step in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Extract next state sequence
            next_state_sequence = next_state[:num_stocks * lookback_window * (filtered_data.shape[-1])].reshape(
                lookback_window, num_stocks * (filtered_data.shape[-1]))
            
            # Handle last step case for chaotic features
            next_chaotic_features_sequence = all_chaotic_features[:, step + 1:step + lookback_window + 1, :].reshape(
                lookback_window, -1) if step + 1 < max_steps - lookback_window + 1 else chaotic_features_sequence
            
            # Add transition to replay buffer
            agent.replay_buffer.add(
                (current_state_sequence, chaotic_features_sequence, action, reward, 
                 next_state_sequence, next_chaotic_features_sequence, done)
            )
            
            # Train the agent
            critic_loss, _ = agent.train(
                batch_size=params["batch_size"], 
                discount=params["discount"], 
                tau=params["tau"]
            )
            episode_critic_loss.append(critic_loss)
            
            state = next_state
            
            if done:
                print(f"🚨 Episode {episode} ended early at step {step + lookback_window} due to termination condition.")
                break
        
        # Logging
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
        avg_reward_history.append(avg_reward)
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0
        critic_loss_history.append(avg_critic_loss)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Total Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, "
              f"Avg Critic Loss = {avg_critic_loss:.4f}")
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"td3_agent_tuned_{timestamp}.pth"
    torch.save(agent, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot rewards
    # plt.figure(figsize=(12, 6))
    # plt.plot(reward_history, label="Total Reward", alpha=0.7)
    # plt.plot(avg_reward_history, label="Moving Avg (10 episodes)", linewidth=2)
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.title("TD3 Training Rewards with Optimized Parameters")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig(f"final_rewards_{timestamp}.png")
    # plt.show()
    
    # Plot critic loss
    # plt.figure(figsize=(12, 6))
    # plt.plot(critic_loss_history, label="Critic Loss", alpha=0.7)
    
    # Add moving average for smoothing
    # window_size = 20
    # if len(critic_loss_history) > window_size:
    #     moving_avg = np.convolve(critic_loss_history, np.ones(window_size)/window_size, mode='valid')
    #     plt.plot(range(window_size-1, len(critic_loss_history)), moving_avg, 
    #              label=f"Moving Avg ({window_size} episodes)", linewidth=2)
    
    # plt.xlabel("Episode")
    # plt.ylabel("Loss")
    # plt.title("TD3 Critic Loss Over Time with Optimized Parameters")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.yscale('log')
    # plt.savefig(f"final_critic_loss_{timestamp}.png")
    # plt.show()
    
    return agent


if __name__ == "__main__":
    # Number of trials to run
    n_trials = 50
    
    # Run hyperparameter optimization
    best_params = run_optimization(n_trials=n_trials)
    
    # Train final model with best parameters
    # final_agent = train_with_best_params(best_params)
    
    print("\nHyperparameter optimization and final training completed!")