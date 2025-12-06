#!/usr/bin/env python

"""Quick test script to verify the transformer model builds and trains."""

import sys
import os
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from reference_model_transformer_tunable import (
    train_transformer_model
)
from config import default_params

if __name__ == "__main__":
    print("Testing Inception-Transformer model...")
    print("\nUsing default parameters:")
    for k, v in sorted(default_params.items()):
        print(f"  {k}: {v}")
    
    # Start timing
    start_time = time.time()
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Train with default params
    results = train_transformer_model(
        default_params.copy(),
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

