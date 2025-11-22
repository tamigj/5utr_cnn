#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import ast

from config import DATA_DIR, initial_params, get_tuning_output_dir
from utils import build_cnn_model, tune_hyperparameter

# Load preprocessed data
data = np.load(f'{DATA_DIR}/preprocessed_data.npz')

X_train = data['X_train']
Y_train = data['Y_train']
X_dev = data['X_dev']
Y_dev = data['Y_dev']
X_test = data['X_test']
Y_test = data['Y_test']

# Extract parameters
batch_size = initial_params['batch_size']
steps_per_epoch = len(X_train) // batch_size

### Command-line interface
if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python hyperparameter_tuning.py <strategy> <param_name> <value1> <value2> ...")
        print("Example: python hyperparameter_tuning.py wide n_filters 32 64 128")
        print("Example: python hyperparameter_tuning.py coarse learning_rate 1e-4 1e-3 1e-2")
        sys.exit(1)

    strategy = sys.argv[1]
    param_name = sys.argv[2]
    param_values_str = sys.argv[3:]

    # Convert string values to appropriate types
    param_values = []
    for val_str in param_values_str:
        try:
            # Try to evaluate as Python literal (handles ints, floats, scientific notation)
            val = ast.literal_eval(val_str)
            param_values.append(val)
        except (ValueError, SyntaxError):
            print(f"Error: Could not parse value '{val_str}'")
            sys.exit(1)

    print(f"Starting hyperparameter tuning:")
    print(f"  Strategy: {strategy}")
    print(f"  Parameter: {param_name}")
    print(f"  Values: {param_values}")

    # Get strategy-specific output directory
    output_dir = get_tuning_output_dir(strategy)

    # Run tuning
    results = tune_hyperparameter(param_name, param_values,
                                  initial_params, steps_per_epoch,
                                  X_train, Y_train, X_dev, Y_dev,
                                  output_dir, strategy)

    print("\nTuning complete!")
