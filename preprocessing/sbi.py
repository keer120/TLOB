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
        path: Path to the SBI data directory or direct path to CSV file
        seq_size: Sequence size for the model
        horizon: Prediction horizon
        all_features: Whether to use all features or only price features
    
    Returns:
        train_input, train_labels, val_input, val_labels, test_input, test_labels
    """
    # Handle both directory path and direct file path
    if os.path.isdir(path):
        # If path is a directory, look for sbi_data.csv inside it
        csv_file = os.path.join(path, "sbi_data.csv")
    else:
        # If path is a direct file path, use it as is
        csv_file = path
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"SBI data file not found at {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if we have enough data
    if len(df) < 100:  # Minimum required samples
        raise ValueError(f"SBI dataset too small: {len(df)} samples. Need at least 100 samples.")
    
    print(f"SBI Dataset loaded: {len(df)} samples")
    
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
        
        # Add 24 dummy market features to match FI-2010's 144 features
        # Create dummy features with same number of samples as feature_data
        dummy_market_features = np.zeros((feature_data.shape[0], 24))
        feature_data = np.column_stack([feature_data, dummy_market_features])
    else:
        # Use only price features (40 features)
        feature_data = price_data
        
        # Add 104 dummy features to match FI-2010's 144 features
        dummy_features = np.zeros((feature_data.shape[0], 104))
        feature_data = np.column_stack([feature_data, dummy_features])
    
    # Transpose to match the expected format (features x time)
    feature_data = feature_data.T
    
    # Calculate mid-price for labeling
    mid_price = (df[bid_price_cols[0]].values.astype(np.float64) + df[ask_price_cols[0]].values.astype(np.float64)) / 2
    
    # Create labels based on price movement
    # For horizon 10, we'll use a simple approach: compare current mid-price with future mid-price
    labels = np.zeros(len(mid_price) - horizon)
    
    # Calculate some statistics for debugging
    price_changes = []
    for i in range(len(mid_price) - horizon):
        current_price = mid_price[i]
        future_price = mid_price[i + horizon]
        pct_change = (future_price - current_price) / current_price
        price_changes.append(pct_change)
    
    # Print some statistics about price changes
    price_changes = np.array(price_changes)
    print(f"SBI Dataset Price Change Statistics:")
    print(f"  Mean: {np.mean(price_changes):.6f}")
    print(f"  Std: {np.std(price_changes):.6f}")
    print(f"  Min: {np.min(price_changes):.6f}")
    print(f"  Max: {np.max(price_changes):.6f}")
    print(f"  Percentiles: 1%={np.percentile(price_changes, 1):.6f}, 99%={np.percentile(price_changes, 99):.6f}")
    
    # Use more reasonable thresholds based on the data
    # Use 1st and 99th percentiles as thresholds
    up_threshold = np.percentile(price_changes, 66)  # Top 33%
    down_threshold = np.percentile(price_changes, 34)  # Bottom 33%
    
    print(f"  Using thresholds: up > {up_threshold:.6f}, down < {down_threshold:.6f}")
    
    for i in range(len(mid_price) - horizon):
        current_price = mid_price[i]
        future_price = mid_price[i + horizon]
        
        # Calculate percentage change
        pct_change = (future_price - current_price) / current_price
        
        # Define thresholds for classification
        if pct_change > up_threshold:  # Up movement
            labels[i] = 0
        elif pct_change < down_threshold:  # Down movement
            labels[i] = 2
        else:  # Stationary
            labels[i] = 1
    
    # Print label distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique_labels, label_counts))}")
    
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
    
    # Align input and label lengths for rolling window
    train_input = train_features.T
    val_input = val_features.T
    test_input = test_features.T

    train_max_len = min(train_input.shape[0], len(train_labels) + seq_size - 1)
    train_input = train_input[:train_max_len]
    train_labels = train_labels[:train_max_len - seq_size + 1]

    val_max_len = min(val_input.shape[0], len(val_labels) + seq_size - 1)
    val_input = val_input[:val_max_len]
    val_labels = val_labels[:val_max_len - seq_size + 1]

    test_max_len = min(test_input.shape[0], len(test_labels) + seq_size - 1)
    test_input = test_input[:test_max_len]
    test_labels = test_labels[:test_max_len - seq_size + 1]

    train_input = torch.from_numpy(train_input).float().contiguous()
    train_labels = torch.from_numpy(train_labels).long().contiguous()
    val_input = torch.from_numpy(val_input).float().contiguous()
    val_labels = torch.from_numpy(val_labels).long().contiguous()
    test_input = torch.from_numpy(test_input).float().contiguous()
    test_labels = torch.from_numpy(test_labels).long().contiguous()
    
    # Print tensor shapes for debugging
    print(f"SBI Dataset Tensor Shapes:")
    print(f"  train_input: {train_input.shape}")
    print(f"  train_labels: {train_labels.shape}")
    print(f"  val_input: {val_input.shape}")
    print(f"  val_labels: {val_labels.shape}")
    print(f"  test_input: {test_input.shape}")
    print(f"  test_labels: {test_labels.shape}")
    
    # Check if we have enough samples
    min_samples = min(train_input.shape[0], val_input.shape[0], test_input.shape[0])
    if min_samples < seq_size:
        print(f"Warning: Not enough samples ({min_samples}) for sequence size {seq_size}")
        print("Adjusting sequence size to fit available data...")
        # Use a smaller sequence size that fits the data
        adjusted_seq_size = min(seq_size, min_samples - 1)
        print(f"Using adjusted sequence size: {adjusted_seq_size}")
        return train_input, train_labels, val_input, val_labels, test_input, test_labels, adjusted_seq_size
    
    return train_input, train_labels, val_input, val_labels, test_input, test_labels
