from pykalman import KalmanFilter
import numpy as np

def apply_kalman_filter(data, observation_covariance=1.0, transition_covariance=0.1):
    """Applies a Kalman filter to smooth the given data."""
    
    T, N, F = data.shape
    smoothed_data = np.zeros_like(data)
    
    for stock_idx in range(N):
        for feature_idx in range(F):
            series = data[:, stock_idx, feature_idx]
            
            # Initialize Kalman Filter
            kf = KalmanFilter(
                initial_state_mean=series[0],
                observation_covariance=observation_covariance,
                transition_covariance=transition_covariance,
            )
            
            # Apply Kalman Filter
            smoothed_state_means, _ = kf.filter(series)
            smoothed_data[:, stock_idx, feature_idx] = smoothed_state_means.flatten()
            
    return smoothed_data