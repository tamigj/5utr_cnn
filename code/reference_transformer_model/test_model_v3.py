#!/usr/bin/env python

"""Test script for Inception-Transformer model - Configuration v3.

Strategy: Balanced approach with deeper transformer
- Moderate CNN layers
- Larger embedding dimension
- Deep transformer stack
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

# Configuration v3: Balanced with deeper transformer
# Embedding dim: 60 (from n_filters=60, filter_sizes=[3,5,7] → 3 branches × 20 filters)
params_v3 = {
    # Inception CNN parameters (moderate)
    'n_conv_layers': 4,  # Moderate between v1 and v2
    'n_filters': 60,  # Larger embedding for more capacity
    'filter_sizes': [3, 5, 7],
    'skip_dropout_in_first_conv_layer': False,
    
    # Transformer parameters (deep stack)
    'n_transformer_layers': 5,  # Deep transformer
    'num_heads': 6,  # Divides evenly into embed_dim=60 (60/6=10 per head)
    'ff_dim': 240,  # 4x embedding dimension (60*4)
    
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
    print("Testing Inception-Transformer model - Configuration v3")
    print("Strategy: Balanced approach with deeper transformer")
    print("\nParameters:")
    for k, v in sorted(params_v3.items()):
        print(f"  {k}: {v}")
    
    # Calculate embedding dimension
    n_branches = len(params_v3['filter_sizes'])
    n_filters_per_branch = max(1, params_v3['n_filters'] // n_branches)
    embed_dim = n_branches * n_filters_per_branch
    print(f"\n  Calculated embedding dimension: {embed_dim}")
    print(f"  num_heads={params_v3['num_heads']} → {embed_dim // params_v3['num_heads']} dims per head")
    
    # Start timing
    start_time = time.time()
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Train with v3 params
    results = train_transformer_model(
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

