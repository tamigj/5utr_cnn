#!/usr/bin/env python

"""Test script for Inception-Transformer model - Configuration v4.

Strategy: Maximum transformer capacity
- Deeper transformer stack (6 layers)
- Larger embedding dimension with more attention heads
- Optimized head count for better attention diversity
- Larger feed-forward network
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

# Configuration v4: Maximum transformer capacity
# Embedding dim: 72 (from n_filters=72, filter_sizes=[3,5,7] → 3 branches × 24 filters)
# This allows for 8 heads with 9 dims per head (more diverse attention patterns)
params_v4 = {
    # Inception CNN parameters (keep moderate like v3)
    'n_conv_layers': 4,  # Same as v3
    'n_filters': 72,  # Larger embedding for maximum capacity
    'filter_sizes': [3, 5, 7],
    'skip_dropout_in_first_conv_layer': False,
    
    # Transformer parameters (maximum capacity)
    'n_transformer_layers': 6,  # Even deeper than v3 (5→6)
    'num_heads': 8,  # More heads than v3 (6→8) for diverse attention patterns
    # embed_dim=72, 72/8 = 9 dims per head (good balance)
    'ff_dim': 360,  # 5x embedding dimension (72*5) - larger than v3's 4x
    
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
    print("Testing Inception-Transformer model - Configuration v4")
    print("Strategy: Maximum transformer capacity")
    print("\nParameters:")
    for k, v in sorted(params_v4.items()):
        print(f"  {k}: {v}")
    
    # Calculate embedding dimension
    n_branches = len(params_v4['filter_sizes'])
    n_filters_per_branch = max(1, params_v4['n_filters'] // n_branches)
    embed_dim = n_branches * n_filters_per_branch
    print(f"\n  Calculated embedding dimension: {embed_dim}")
    print(f"  num_heads={params_v4['num_heads']} → {embed_dim // params_v4['num_heads']} dims per head")
    
    # Start timing
    start_time = time.time()
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Train with v4 params
    results = train_transformer_model(
        params_v4.copy(),
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

