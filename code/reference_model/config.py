#!/usr/bin/env python

import os

# Paths
BASE_DIR = '/mnt/oak/users/tami/5utr_cnn'
DATA_DIR = f'{BASE_DIR}/data'
OUTPUT_DIR = f'{BASE_DIR}/output'
OUTPUT_TUNING_DIR = f'{OUTPUT_DIR}/hyperparameter_tuning'

# Constants
MAX_SEQ_LEN = 180
N_BASES = 4
INPUT_SHAPE = (MAX_SEQ_LEN, N_BASES)
NUM_EPOCH = 100

# Train/dev/test set
train_prop = 0.7
dev_prop = 0.2
test_prop = 0.1

# Starting parameters
initial_params = {
    'n_conv_layers': 3,
    'n_filters': 128,
    'filter_size': 7,
    'n_dense_layers': 2,
    'n_dense_units': 150,
    'learning_rate': 1e-3,
    'decay_rate': 0.95,
    'epoch_decay_interval': 5,
    'l2_lambda': 0.01,
    'dropout_rate': 0.1,
    'skip_dropout_in_first_conv_layer': True,
    'batch_size': 64
}

# Hyperparameter tuning grid
tuning_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'n_conv_layers': [1, 3, 5, 7],
    'n_filters': [10, 50, 100, 200],
    'filter_size': [3, 5, 7, 9, 11],
    'n_dense_layers': [1, 2, 3, 4, 5],
    'n_dense_units': [25, 50, 100, 200, 500],
    'decay_rate': [0.75, 0.8, 0.85, 0.9, 0.95],
    'epoch_decay_interval': [5, 10, 25, 50],
    'l2_lambda': [0.001, 0.01, 0.1, 1],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'skip_dropout_in_first_conv_layer': [True, False],
    'batch_size': [64, 128, 256, 512]
}

# Post-single hyperparameter tuning parameters
post_tuning_params = {
    'n_conv_layers': 2,
    'n_filters': 10,
    'filter_size': 7,
    'n_dense_layers': 3,
    'n_dense_units': 50,
    'learning_rate': 0.001,
    'decay_rate': 0.95,
    'epoch_decay_interval': 10,
    'l2_lambda': 0.05,
    'dropout_rate': 0.4,
    'skip_dropout_in_first_conv_layer': True,
    'batch_size': 64
}

def get_tuning_output_dir(strategy='wide'):
    """Get output directory for a specific tuning strategy.
    
    Args:
        strategy (str): Strategy name (wide, coarse, fine, etc.)
    
    Returns:
        str: Path to strategy-specific output directory
    """
    strategy_dir = f'{OUTPUT_TUNING_DIR}/{strategy}'
    os.makedirs(strategy_dir, exist_ok=True)
    return strategy_dir
