#!/usr/bin/env python

"""
Pre-tuning model: Train inception model with initial parameters.

This script trains the inception model with initial_params from config,
creates training/validation loss plots, and reports the minimum dev loss.
This serves as a baseline before hyperparameter tuning.
"""

import os
import sys

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

from config import (
    DATA_DIR,
    OUTPUT_TUNING_DIR,
    initial_params,
)
from reference_model_inception_tunable import train_inception_model

if __name__ == "__main__":
    print("=" * 70)
    print("Pre-Tuning Model: Training with Initial Parameters")
    print("=" * 70)
    print("\nInitial Parameters:")
    for k, v in sorted(initial_params.items()):
        print(f"  {k}: {v}")
    print("=" * 70)
    
    # Train model with initial parameters
    results = train_inception_model(
        initial_params.copy(),
        output_dir=OUTPUT_TUNING_DIR,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("Pre-tuning model training complete!")
    print(f"Minimum dev loss: {results['min_dev_loss']:.6f}")
    print(f"Best epoch: {results['best_epoch']}")
    print("=" * 70)

