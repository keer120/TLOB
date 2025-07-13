import numpy as np
import pandas as pd
import constants as cst
import os
from torch.utils import data
import torch


def sbi_load(path, seq_size, horizon, all_features):
    """
    Load and preprocess SBI dataset from CSV files.
    
    Args:
        path: Path to the SBI data directory
        seq_size: Sequence size for the model
        horizon: Prediction horizon
        all_features: Whether to use all features or only price features
    
    Returns:
        train_input, train_labels, val_input, val_labels, test_input, test_labels
    """
    # Load the SBI CSV file
    csv_file = os.path.join(path, "sbi_data.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"SBI data file not found at {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract bid and ask price columns (first 20 levels)
    bid_price_cols = [f'bid_price_{i}' for i in range(1, 21)]
    ask_price_cols = [f'ask_price_{i}' for i in range(1, 21)]
    bid_quantity_cols = [f'bid_quantity_{i}' for i in range(1, 21)]
    ask_quantity_cols = [f'ask_quantity_{i}' for i in range(1, 21)]
    bid_orders_cols = [f'bid_orders_{i}' for i in range(1, 21)]
    ask_orders_cols = [f'ask_orders_{i}' for i in range(1, 21)]
    
    # Create feature matrix
    price_features = []
    quantity_features = []
    order_features = []
    
    # Extract bid features
    for i in range(20):
        price_features.append(df[bid_price_cols[i]].values)
        quantity_features.append(df[bid_quantity_cols[i]].values)
        order_features.append(df[bid_orders_cols[i]].values)
    
    # Extract ask features
    for i in range(20):
        price_features.append(df[ask_price_cols[i]].values)
        quantity_features.append(df[ask_quantity_cols[i]].values)
        order_features.append(df[ask_orders_cols[i]].values)
    
    # Stack features
    price_data = np.column_stack(price_features)  # 40 features (20 bid + 20 ask prices)
    quantity_data = np.column_stack(quantity_features)  # 40 features (20 bid + 20 ask quantities)
    order_data = np.column_stack(order_features)  # 40 features (20 bid + 20 ask orders)
    
    # Combine all features
    if all_features:
        # Use all features: prices, quantities, and orders (120 features total)
        feature_data = np.column_stack([price_data, quantity_data, order_data])
    else:
        # Use only price features (40 features)
        feature_data = price_data
    
    # Transpose to match the expected format (features x time)
    feature_data = feature_data.T
    
    # Calculate mid-price for labeling
    mid_price = (df[bid_price_cols[0]].values + df[ask_price_cols[0]].values) / 2
    
    # Create labels based on price movement
    # For horizon 10, we'll use a simple approach: compare current mid-price with future mid-price
    labels = np.zeros(len(mid_price) - horizon)
    
    for i in range(len(mid_price) - horizon):
        current_price = mid_price[i]
        future_price = mid_price[i + horizon]
        
        # Calculate percentage change
        pct_change = (future_price - current_price) / current_price
        
        # Define thresholds for classification
        if pct_change > 0.001:  # Up movement (>0.1%)
            labels[i] = 0
        elif pct_change < -0.001:  # Down movement (<-0.1%)
            labels[i] = 2
        else:  # Stationary
            labels[i] = 1
    
    # Split data into train/val/test
    total_samples = feature_data.shape[1]
    train_end = int(total_samples * cst.SPLIT_RATES[0])
    val_end = int(total_samples * (cst.SPLIT_RATES[0] + cst.SPLIT_RATES[1]))
    
    # Split features
    train_features = feature_data[:, :train_end]
    val_features = feature_data[:, train_end:val_end]
    test_features = feature_data[:, val_end:]
    
    # Split labels (accounting for horizon offset)
    train_labels = labels[:train_end - horizon]
    val_labels = labels[train_end:val_end - horizon]
    test_labels = labels[val_end:len(labels)]
    
    # Convert to torch tensors
    train_input = torch.from_numpy(train_features.T).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_input = torch.from_numpy(val_features.T).float()
    val_labels = torch.from_numpy(val_labels).long()
    test_input = torch.from_numpy(test_features.T).float()
    test_labels = torch.from_numpy(test_labels).long()
    
    return train_input, train_labels, val_input, val_labels, test_input, test_labels 