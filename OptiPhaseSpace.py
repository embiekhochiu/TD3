import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from scipy.signal import find_peaks
from enum import Enum
from scipy.spatial.distance import cdist


class ChaoticFeatureExtractor:
    def __init__(self, embedding_dim=2, time_delay=1): #Optimize this params later using autocorr or mutual info
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.output_dim = None

    def phase_space_reconstruction(self, data):
        num_points = len(data) - (self.embedding_dim - 1) * self.time_delay
        if num_points <= 0:
            raise ValueError(
                f"Input time series length is too short for the given embedding_dim and time_delay. "
                f"Data length: {len(data)}, Required length: {(self.embedding_dim - 1) * self.time_delay + 1}"
            )
        phase_space = np.zeros((num_points, self.embedding_dim))
        for i in range(num_points):
            for j in range(self.embedding_dim):
                phase_space[i, j] = data[i + j * self.time_delay]
        return phase_space

    def calculate_lyapunov_exponent(self, data):
        phase_space = self.phase_space_reconstruction(data)

        # Compute pairwise distances and identify nearest neighbors
        distances = cdist(phase_space, phase_space)
        np.fill_diagonal(distances, np.inf)  # Exclude self-pairs
        neighbors = np.argmin(distances, axis=1)  # Indices of nearest neighbors

        lyapunov = []
        for i in range(len(phase_space)):
            d_i = np.linalg.norm(phase_space[i] - phase_space[neighbors[i]])
            if d_i <= 0:
                continue  # Avoid log(0) or invalid cases
            for j in range(1, 3):
                idx1 = i + j
                idx2 = neighbors[i] + j
                if idx1 >= len(phase_space) or idx2 >= len(phase_space):
                    break
                d_ij = np.linalg.norm(phase_space[idx1] - phase_space[idx2])
                if d_ij <= 0:
                    continue
                lyapunov.append(np.log(d_ij / d_i))

        return np.mean(lyapunov) if lyapunov else None
    
    def calculate_rolling_lyapunov(self, data, window_size, step):
        lyapunov_values = []
        for start_idx in range(0, len(data) - window_size + 1, step):
            window = data[start_idx : start_idx + window_size]
            lyapunov_exponent = self.calculate_lyapunov_exponent(window)
            lyapunov_values.append(lyapunov_exponent)
        return np.array(lyapunov_values)


    def peak_count(self, data, window_size, step):
        peak_counts = []
        for start_idx in range(0, len(data) - window_size + 1, step):
            window = data[start_idx : start_idx + window_size]
            peaks, _ = find_peaks(window)
            peak_counts.append(len(peaks))
        return np.array(peak_counts)
    

    def calculate_fractal_dimension(self, data):
        #Calculate the fractal dimension using a simple box-counting method.
        N = len(data)
        box_sizes = np.arange(1, N // 2, step=2)
        counts = []

        for size in box_sizes:
            coarse_grained = [data[i:i+size].mean() for i in range(0, N, size) if i+size <= N]
            counts.append(len(set(coarse_grained)))

        if len(box_sizes) > 1 and len(counts) > 1:
            return np.polyfit(np.log(box_sizes), np.log(counts), 1)[0] * -1
        return None
    
    def calculate_fractal_dimension_rolling(self, data, window_size, step):
        fractal_dimensions = []
        for start_idx in range(0, len(data) - window_size + 1, step):
            window = data[start_idx : start_idx + window_size]
            fractal_dim = self.calculate_fractal_dimension(window)
            fractal_dimensions.append(fractal_dim if fractal_dim is not None else 0)  
        return np.array(fractal_dimensions)  



    def extract_features(self, data):
        T, N, F = data.shape 
        chaotic_features = []
        window_size = 7

        for stock_idx in range(N):
            stock_features = []
            for feature_idx in range(F):
                time_series = data[:, stock_idx, feature_idx]  # Extract time-series for a single stock and feature
                
                lyapunov_exponent = self.calculate_rolling_lyapunov(time_series,window_size=window_size,step=1)
                peak_counts = self.peak_count(time_series,window_size=window_size,step=1)
                fractal_dimension = self.calculate_fractal_dimension_rolling(time_series,window_size=window_size,step=1)

                padding_length = window_size - 1
                lyapunov_exponent = np.pad(lyapunov_exponent, (padding_length, 0), mode='edge') # pad with first value.
                peak_counts = np.pad(peak_counts, (padding_length, 0), mode='edge')
                fractal_dimension = np.pad(fractal_dimension, (padding_length, 0), mode='edge')

                stock_features.append(np.stack([lyapunov_exponent, peak_counts, fractal_dimension], axis=-1))
            
            chaotic_features.append(np.array(stock_features))  # (F, T, C)

        chaotic_features = np.array(chaotic_features)  # (N, F, T, C)
        self.output_dim = F * chaotic_features.shape[-1]
        
        return chaotic_features.transpose(0, 2, 1, 3).reshape(N, T, -1)  # Final shape (N, T, F*C)