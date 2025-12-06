#!/usr/bin/env python

"""Test script for Inception-RNN model - Configuration v3.

Strategy: Maximum RNN capacity
- Deep RNN stack (4 layers)
- Large RNN units per layer
- Bidirectional LSTM for comprehensive sequence modeling
"""

import sys
import os
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from reference_model_rnn_tunable import (
    train_rnn_model
)

# Configuration v3: Maximum RNN capacity
params_v3 = {
    # Inception CNN parameters (moderate, let RNN do more work)
    'n_conv_layers': 4,  # Moderate CNN layers
    'n_filters': 48,  # Moderate filters
    'filter_sizes': [3, 5, 7],
    'skip_dropout_in_first_conv_layer': False,
    
    # RNN parameters (maximum capacity)
    'n_rnn_layers': 4,  # Deep RNN stack (2→4)
    'rnn_units': 128,  # Large units per layer (64→128)
    'rnn_type': 'lstm',  # LSTM for better long-term memory
    'bidirectional': True,  # Bidirectional for context from both directions
    
    # Dense head parameters
    'n_dense_layers': 1,
    'n_dense_units': 64,
    
    # Training parameters
    'learning_rate': 0.001,
    'l2_lambda': 0.05,
    'dropout_rate': 0.15,
    'batch_size': 128,
}

if __name__ == "__main__":
    print("Testing Inception-RNN model - Configuration v3")
    print("Strategy: Maximum RNN capacity")
    print("\nParameters:")
    for k, v in sorted(params_v3.items()):
        print(f"  {k}: {v}")
    
    # Start timing
    start_time = time.time()
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Train with v3 params
    results = train_rnn_model(
        params_v3.copy(),
        output_dir=None,
        verbose=1
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✓ Model trained successfully!")
    print(f"  Min Dev Loss: {results['min_dev_loss']:.6f}")
    print(f"  Test Loss (MSE): {results['test_loss']:.6f}")
    print(f"  Test R²: {results['test_r2']:.6f}")
    print(f"\nRuntime: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} seconds)")
    print("=" * 70)


