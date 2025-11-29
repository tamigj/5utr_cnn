#!/usr/bin/env python
"""Combinatorial hyperparameter tuning script.

This script performs combinatorial hyperparameter tuning using the
`tune_hyperparameter_combinations` helper. Multiple tuning modes (learning vs
architecture) are defined in config.py.

Usage:
    python combinatorial_tuning.py <strategy> <mode> <job_param_value>

Examples:
    python combinatorial_tuning.py combo_learning learning 0.01
    python combinatorial_tuning.py combo_architecture architecture 3
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import ast

from config import (
    DATA_DIR,
    updated_params_1,
    updated_params_2,
    combinatorial_tuning_grid_learning,
    combinatorial_tuning_grid_architecture,
)
from utils import tune_hyperparameter_combinations, get_tuning_output_dir

# Load preprocessed data
data = np.load(f'{DATA_DIR}/preprocessed_data.npz')

X_train = data['X_train']
Y_train = data['Y_train']
X_dev = data['X_dev']
Y_dev = data['Y_dev']
X_test = data['X_test']
Y_test = data['Y_test']

# Mode-specific metadata
MODE_CONFIGS = {
    'learning': {
        'base_params': updated_params_1,
        'grid': combinatorial_tuning_grid_learning,
        'job_param': 'learning_rate',
        'cast': float,
    },
    'architecture': {
        'base_params': updated_params_2,
        'grid': combinatorial_tuning_grid_architecture,
        'job_param': 'n_conv_layers',
        'cast': int,
    },
}

### Command-line interface
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python combinatorial_tuning.py <strategy> <mode> <job_param_value>")
        print("  Modes: {}".format(', '.join(MODE_CONFIGS.keys())))
        print("Example: python combinatorial_tuning.py combo_learning learning 0.01")
        print("Example: python combinatorial_tuning.py combo_architecture architecture 3")
        sys.exit(1)

    strategy = sys.argv[1]
    mode = sys.argv[2]
    job_value_raw = sys.argv[3]

    if mode not in MODE_CONFIGS:
        print(f"Error: Unknown mode '{mode}'. Expected one of: {', '.join(MODE_CONFIGS.keys())}")
        sys.exit(1)

    mode_config = MODE_CONFIGS[mode]
    job_param = mode_config['job_param']
    caster = mode_config.get('cast', lambda x: x)

    try:
        parsed_job_value = ast.literal_eval(job_value_raw)
    except (ValueError, SyntaxError):
        parsed_job_value = job_value_raw

    try:
        job_value = caster(parsed_job_value)
    except Exception as exc:
        print(f"Error: Could not coerce job parameter value '{job_value_raw}' for mode '{mode}': {exc}")
        sys.exit(1)

    # Build base parameters and param grid
    base_params = mode_config['base_params'].copy()
    param_grid = {k: v for k, v in mode_config['grid'].items() if k != job_param}

    if not param_grid:
        print(f"Error: Parameter grid for mode '{mode}' is empty after removing '{job_param}'.")
        sys.exit(1)

    base_params[job_param] = job_value

    # Determine steps_per_epoch based on batch size values being tuned (if any)
    if 'batch_size' in param_grid:
        min_batch_size = min(param_grid['batch_size'])
    else:
        min_batch_size = base_params['batch_size']
    steps_per_epoch = len(X_train) // max(1, int(min_batch_size))

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"Starting combinatorial hyperparameter tuning:")
    print(f"  Strategy: {strategy}")
    print(f"  Mode: {mode}")
    print(f"  Fixed {job_param}: {job_value}")
    print(f"  Parameter grid: {param_grid}")
    print(f"  Total combinations: {total_combinations}")

    # Get strategy-specific output directory
    output_dir = get_tuning_output_dir(strategy)

    # Run combinatorial tuning
    results = tune_hyperparameter_combinations(param_grid, base_params, steps_per_epoch,
                                                X_train, Y_train, X_dev, Y_dev,
                                                output_dir, strategy)

    print("\nCombinatorial tuning complete!")

