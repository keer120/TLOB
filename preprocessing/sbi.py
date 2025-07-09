import pandas as pd
import numpy as np
import torch
from utils.utils_data import z_score_orderbook, labeling

def sbi_load(csv_path, seq_size, horizon, all_features):
    # Read the SBI CSV
    df = pd.read_csv(csv_path)
    
    # Extract features: mimic FI-2010 (first 144 columns if all_features, else 40)
    # SBI columns: skip first 4 (timestamps, exchange_segment, security_id)
    feature_cols = df.columns[4:]
    if all_features:
        features = df[feature_cols[:144]].copy()
    else:
        features = df[feature_cols[:40]].copy()
    
    # Normalize features (z-score)
    features, mean_size, mean_prices, std_size, std_prices = z_score_orderbook(features)
    X = features.values.astype(np.float32)
    
    # Labeling: use mid-price columns (ask_price_1, bid_price_1)
    # Find the correct columns for ask_price_1 and bid_price_1
    ask_price_1_col = [col for col in df.columns if col.startswith('ask_price_1')][0]
    bid_price_1_col = [col for col in df.columns if col.startswith('bid_price_1')][0]
    # For labeling, create a 2-column array: [ask_price_1, bid_price_1]
    X_label = df[[ask_price_1_col, bid_price_1_col]].values.astype(np.float32)
    # The labeling function expects a 2D array with ask and bid prices at least
    # But the original labeling uses X[:, 0] (ask) and X[:, 2] (bid), so we need to match that
    # We'll create a dummy array with shape (N, 3): [ask, dummy, bid]
    X_label_full = np.zeros((X_label.shape[0], 3), dtype=np.float32)
    X_label_full[:, 0] = X_label[:, 0]  # ask
    X_label_full[:, 2] = X_label[:, 1]  # bid
    # Generate labels
    labels = labeling(X_label_full, len=seq_size, h=horizon)
    # Align features and labels (due to windowing in labeling)
    X = X[seq_size-1:len(labels)+seq_size-1]
    # Convert to torch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(labels).long()
    return X_tensor, y_tensor 