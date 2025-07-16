import numpy as np
import pandas as pd
import constants as cst
import os
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_threshold_for_horizon(horizon):
    # Use a larger threshold for larger horizons (example: 0.001 for 10, 0.005 for 100)
    if horizon <= 10:
        return 0.001
    elif horizon <= 20:
        return 0.002
    elif horizon <= 50:
        return 0.003
    elif horizon <= 100:
        return 0.005
    else:
        return 0.01

def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Up", "Stable", "Down"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)

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

    # --- DEBUG: Print price diff stats and label distribution for various thresholds ---
    price_diff = pd.Series(mid_price).shift(-horizon) - pd.Series(mid_price)
    print("Price diff stats:")
    print(price_diff.describe())
    print("Max abs diff:", price_diff.abs().max())

    def create_labels_for_threshold(price_col, horizon, threshold):
        price_diff = pd.Series(price_col).shift(-horizon) - pd.Series(price_col)
        labels = np.zeros(len(price_col) - horizon)
        for i in range(len(price_col) - horizon):
            current_price = price_col[i]
            future_price = price_col[i + horizon]
            pct_change = (future_price - current_price) / current_price
            if pct_change > threshold:
                labels[i] = 0  # Up
            elif pct_change < -threshold:
                labels[i] = 2  # Down
            else:
                labels[i] = 1  # Stable
        return labels

    print("Label distribution for various thresholds:")
    for thresh in [0.0, 0.00005, 0.0001, 0.0005, 0.001, 0.005]:
        labels_dbg = create_labels_for_threshold(mid_price, horizon, thresh)
        unique_dbg, counts_dbg = np.unique(labels_dbg, return_counts=True)
        print(f"  Threshold {thresh}: {{" + ", ".join([f'{int(u)}: {c}' for u, c in zip(unique_dbg, counts_dbg)]) + "}}")
    print("--- End debug ---\n")
    # --- END DEBUG ---

    # Create labels based on price movement
    # For horizon 10, we'll use a simple approach: compare current mid-price with future mid-price
    labels = np.zeros(len(mid_price) - horizon)
    
    # Use a fixed threshold of 0.0001 for balanced up/stat/down classes
    threshold = 0.0001  # Chosen for balanced class distribution (see analyze_thresholds_for_labeling)
    print(f"Using horizon={horizon}, threshold for up/down labeling: {threshold}")

    for i in range(len(mid_price) - horizon):
        current_price = mid_price[i]
        future_price = mid_price[i + horizon]
        pct_change = (future_price - current_price) / current_price
        if pct_change > threshold:
            labels[i] = 0  # Up
        elif pct_change < -threshold:
            labels[i] = 2  # Down
        else:
            labels[i] = 1  # Stable
    
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

# --- Add this function to your evaluation script or after test ---
def save_confusion_matrix(y_true, y_pred, out_path):
    plot_confusion_matrix(y_true, y_pred, save_path=out_path)

def analyze_thresholds_for_labeling(mid_price, horizon, true_labels=None):
    """
    Analyze class distribution and confusion matrix for a range of thresholds.
    If true_labels are provided, also print confusion matrix between new labels and true_labels.
    """
    thresholds = [0.0, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    for thresh in thresholds:
        labels = np.zeros(len(mid_price) - horizon)
        for i in range(len(mid_price) - horizon):
            current_price = mid_price[i]
            future_price = mid_price[i + horizon]
            pct_change = (future_price - current_price) / current_price
            if pct_change > thresh:
                labels[i] = 0  # Up
            elif pct_change < -thresh:
                labels[i] = 2  # Down
            else:
                labels[i] = 1  # Stable
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nThreshold {thresh}:")
        print("  Label distribution:", dict(zip(["Up", "Stable", "Down"], counts)))
        if true_labels is not None:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            cm = confusion_matrix(true_labels, labels, labels=[0, 1, 2])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Up", "Stable", "Down"])
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
            plt.title(f'Confusion Matrix (True vs. threshold={thresh})')
            plt.show()
