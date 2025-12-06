#!/usr/bin/env python

"""Test script for Inception-Transformer model - Configuration v2.

Strategy: More transformer capacity, less CNN
- Fewer CNN layers (let transformer do more sequential modeling)
- More filters for larger embedding dimension
- More transformer layers
- Optimized num_heads and ff_dim
"""

import sys
import os
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from reference_model_transformer_tunable import (
    train_transformer_model
)

# Configuration v2: More transformer capacity
# Embedding dim: 48 (from n_filters=48, filter_sizes=[3,5,7] → 3 branches × 16 filters)
params_v2 = {
    # Inception CNN parameters (reduced to let transformer do more)
    'n_conv_layers': 3,  # Reduced from 5
    'n_filters': 48,  # Increased from 32 for larger embedding
    'filter_sizes': [3, 5, 7],
    'skip_dropout_in_first_conv_layer': False,
    
    # Transformer parameters (increased capacity)
    'n_transformer_layers': 4,  # Increased from 2
    'num_heads': 6,  # Divides evenly into embed_dim=48 (48/6=8 per head)
    'ff_dim': 192,  # 4x embedding dimension (48*4)
    
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
    print("Testing Inception-Transformer model - Configuration v2")
    print("Strategy: More transformer capacity, less CNN")
    print("\nParameters:")
    for k, v in sorted(params_v2.items()):
        print(f"  {k}: {v}")
    
    # Calculate embedding dimension
    n_branches = len(params_v2['filter_sizes'])
    n_filters_per_branch = max(1, params_v2['n_filters'] // n_branches)
    embed_dim = n_branches * n_filters_per_branch
    print(f"\n  Calculated embedding dimension: {embed_dim}")
    print(f"  num_heads={params_v2['num_heads']} → {embed_dim // params_v2['num_heads']} dims per head")
    
    # Start timing
    start_time = time.time()
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Train with v2 params
    results = train_transformer_model(
        params_v2.copy(),
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

