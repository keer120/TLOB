#!/usr/bin/env python3
"""
Test script for SBI dataset loading
"""

import sys
import os
sys.path.append('.')

from preprocessing.sbi import sbi_load

def test_sbi_loading():
    """Test SBI dataset loading"""
    try:
        # Test with a small sequence size
        path = "/content/combined_output_week_20.csv"
        seq_size = 10
        horizon = 10
        all_features = True
        
        print("Testing SBI dataset loading...")
        result = sbi_load(path, seq_size, horizon, all_features)
        
        if len(result) == 7:
            train_input, train_labels, val_input, val_labels, test_input, test_labels, adjusted_seq_size = result
            print(f"✓ SBI dataset loaded successfully with adjusted sequence size: {adjusted_seq_size}")
        else:
            train_input, train_labels, val_input, val_labels, test_input, test_labels = result
            print(f"✓ SBI dataset loaded successfully with original sequence size: {seq_size}")
        
        print(f"  Train input shape: {train_input.shape}")
        print(f"  Train labels shape: {train_labels.shape}")
        print(f"  Val input shape: {val_input.shape}")
        print(f"  Val labels shape: {val_labels.shape}")
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Test labels shape: {test_labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading SBI dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_sbi_loading()
    if success:
        print("\nSBI dataset loading test passed!")
    else:
        print("\nSBI dataset loading test failed!")
        sys.exit(1) 