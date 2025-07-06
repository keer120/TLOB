import os
import numpy as np
import pandas as pd
import torch
from constants import DatasetType, SamplingType, N_LOB_LEVELS, LEN_SMOOTH, SPLIT_RATES, DATA_DIR
from preprocessing.dataset import Dataset

class CombinedDataBuilder:
    def __init__(self, data_dir, date_trading_days, split_rates, sampling_type, sampling_time=None, sampling_quantity=None):
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.split_rates = split_rates
        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity
        self.data_path = os.path.join(data_dir, "COMBINED", "combined_output_20.csv")

    def prepare_save_datasets(self):
        # Load and preprocess the dataset
        df = pd.read_csv(self.data_path)
        
        # Select first 10 levels of bid/ask prices and quantities
        bid_price_cols = [f"bid_price_{i}" for i in range(1, N_LOB_LEVELS + 1)]
        bid_quantity_cols = [f"bid_quantity_{i}" for i in range(1, N_LOB_LEVELS + 1)]
        ask_price_cols = [f"ask_price_{i}" for i in range(1, N_LOB_LEVELS + 1)]
        ask_quantity_cols = [f"ask_quantity_{i}" for i in range(1, N_LOB_LEVELS + 1)]
        feature_cols = bid_price_cols + bid_quantity_cols + ask_price_cols + ask_quantity_cols
        
        # Extract features
        features = df[feature_cols].values.astype(np.float32)
        
        # Compute mid-price for labeling
        mid_price = (df["bid_price_1"] + df["ask_price_1"]) / 2
        
        # Normalize features
        mean_price = np.mean(df[bid_price_cols + ask_price_cols].values)
        std_price = np.std(df[bid_price_cols + ask_price_cols].values)
        mean_size = np.mean(df[bid_quantity_cols + ask_quantity_cols].values)
        std_size = np.std(df[bid_quantity_cols + ask_quantity_cols].values)
        
        features[:, :N_LOB_LEVELS] = (features[:, :N_LOB_LEVELS] - mean_price) / std_price  # Bid prices
        features[:, N_LOB_LEVELS:2*N_LOB_LEVELS] = (features[:, N_LOB_LEVELS:2*N_LOB_LEVELS] - mean_size) / std_size  # Bid quantities
        features[:, 2*N_LOB_LEVELS:3*N_LOB_LEVELS] = (features[:, 2*N_LOB_LEVELS:3*N_LOB_LEVELS] - mean_price) / std_price  # Ask prices
        features[:, 3*N_LOB_LEVELS:] = (features[:, 3*N_LOB_LEVELS:] - mean_size) / std_size  # Ask quantities
        
        # Apply smoothing (moving average)
        smoothed_features = np.zeros_like(features)
        for i in range(features.shape[1]):
            smoothed_features[:, i] = np.convolve(features[:, i], np.ones(LEN_SMOOTH)/LEN_SMOOTH, mode='valid')
        features = smoothed_features[LEN_SMOOTH-1:]
        mid_price = mid_price[LEN_SMOOTH-1:]
        
        # Split data
        n_samples = len(features)
        train_end = int(n_samples * self.split_rates[0])
        val_end = train_end + int(n_samples * self.split_rates[1])
        
        train_features = features[:train_end]
        val_features = features[train_end:val_end]
        test_features = features[val_end:]
        train_mid_price = mid_price[:train_end]
        val_mid_price = mid_price[train_end:val_end]
        test_mid_price = mid_price[val_end:]
        
        # Save datasets
        os.makedirs(os.path.join(self.data_dir, "COMBINED"), exist_ok=True)
        np.save(os.path.join(self.data_dir, "COMBINED", "train.npy"), train_features)
        np.save(os.path.join(self.data_dir, "COMBINED", "val.npy"), val_features)
        np.save(os.path.join(self.data_dir, "COMBINED", "test.npy"), test_features)
        
        # Save normalization constants (for reference)
        np.save(os.path.join(self.data_dir, "COMBINED", "norm_params.npy"), 
                np.array([mean_price, std_price, mean_size, std_size]))

def combined_load(path, len_smooth, horizon, seq_size):
    data = np.load(path)
    labels = np.zeros(len(data) - seq_size - horizon + 1, dtype=np.int64)
    
    # Generate labels based on mid-price trend
    for i in range(len(labels)):
        mid_price_t = (data[i, 0] + data[i, 2*N_LOB_LEVELS]) / 2  # bid_price_1 + ask_price_1
        mid_price_future = (data[i + horizon, 0] + data[i + horizon, 2*N_LOB_LEVELS]) / 2
        if mid_price_future > mid_price_t * 1.0005:
            labels[i] = 0  # Up
        elif mid_price_future < mid_price_t * 0.9995:
            labels[i] = 2  # Down
        else:
            labels[i] = 1  # Stationary
    
    inputs = np.zeros((len(data) - seq_size - horizon + 1, seq_size, data.shape[1]), dtype=np.float32)
    for i in range(len(inputs)):
        inputs[i] = data[i:i+seq_size]
    
    return torch.tensor(inputs), torch.tensor(labels)