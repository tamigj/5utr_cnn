#!/usr/bin/env python

"""
Configuration for Inception model hyperparameter tuning.

Defines tuning grids for different phases of hyperparameter optimization.
"""

# Paths
BASE_DIR = '/mnt/oak/users/tami/5utr_cnn'
DATA_DIR = f'{BASE_DIR}/data'
OUTPUT_DIR = f'{BASE_DIR}/output'
OUTPUT_TUNING_DIR = f'{OUTPUT_DIR}/reference_inception_model/hyperparameter_tuning'

# Subdirectories for each tuning phase
OUTPUT_WIDE_DIR = f'{OUTPUT_TUNING_DIR}/wide'
OUTPUT_COMBO_LEARNING_DIR = f'{OUTPUT_TUNING_DIR}/combo_learning'
OUTPUT_COMBO_ARCHITECTURE_DIR = f'{OUTPUT_TUNING_DIR}/combo_architecture'
OUTPUT_REMAINING_PARAMS_DIR = f'{OUTPUT_TUNING_DIR}/remaining_params'

# Constants
MAX_SEQ_LEN = 180
N_BASES = 4
INPUT_SHAPE = (MAX_SEQ_LEN, N_BASES)
NUM_EPOCH = 100

# Train/dev/test set
train_prop = 0.7
dev_prop = 0.2
test_prop = 0.1

# Starting parameters (based on v2)
initial_params = {
    'learning_rate': 0.0005,
    'n_conv_layers': 3,
    'n_filters': 64,
    'n_dense_layers': 2,
    'n_dense_units': 64,
    'l2_lambda': 0.1,
    'dropout_rate': 0.4,
    'skip_dropout_in_first_conv_layer': True,
    'filter_sizes': [3, 5, 7, 9, 13],
    'batch_size': 64
}

# Phase 1: Wide grid search (single parameter tuning)
wide_tuning_grid = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'n_conv_layers': [2, 3, 4, 5],
    'n_filters': [32, 48, 64, 80, 96, 112],
    'n_dense_layers': [1, 2, 3],
    'n_dense_units': [32, 48, 64, 80, 96],
    'l2_lambda': [0.01, 0.05, 0.1, 0.2],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'batch_size': [32, 64, 128],
    'skip_dropout_in_first_conv_layer': [True, False],
    'filter_sizes': [
        [3, 5, 7],
        [3, 5, 7, 9],
        [3, 5, 7, 9, 11],
        [3, 5, 7, 9, 11, 13],
        [3, 5, 7, 11],
        [3, 7, 11],
    ],
}

# Post-wide tuning parameters (will be updated after Phase 1)
updated_params_1 = {
    'batch_size': 64,
    'dropout_rate': 0.1,
    'filter_sizes': [3, 5, 7],
    'l2_lambda': 0.1,
    'learning_rate': 0.001,
    'n_conv_layers': 5,
    'n_dense_layers': 1,
    'n_dense_units': 32,
    'n_filters': 32,
    'skip_dropout_in_first_conv_layer': False,
}

# Phase 2: Combinatorial learning parameters
combinatorial_tuning_grid_learning = {
    'learning_rate': [0.0005, 0.001, 0.0015],  # Around best 0.001
    'l2_lambda': [0.05, 0.1, 0.15],  # Around best 0.1
    'dropout_rate': [0.05, 0.1, 0.15],  # Around best 0.1
    'batch_size': [32, 64, 128],  # Test interactions around best 64
}

# Post-learning combinatorial tuning parameters
# Best combination from learning combinatorial tuning:
# learning_rate=0.001, l2_lambda=0.05, dropout_rate=0.15, batch_size=128

updated_params_2 = {
    'batch_size': 128,
    'dropout_rate': 0.15,
    'filter_sizes': [3, 5, 7],
    'l2_lambda': 0.05,
    'learning_rate': 0.001,
    'n_conv_layers': 5,
    'n_dense_layers': 1,
    'n_dense_units': 32,
    'n_filters': 32,
    'skip_dropout_in_first_conv_layer': False,
}

# Phase 3: Combinatorial architecture parameters
combinatorial_tuning_grid_architecture = {
    'n_conv_layers': [4, 5, 6],
    'n_filters': [32, 48, 64],
    'n_dense_layers': [1, 2, 3],
    'n_dense_units': [32, 48, 64],
    'filter_sizes': [
        [3, 5, 7],
        [3, 5, 7, 9],
        [3, 5, 7, 11],
    ],
}

# Post-architecture combinatorial tuning parameters
# Best combination from architecture combinatorial tuning:
# n_conv_layers=5, n_filters=32, n_dense_layers=1, n_dense_units=64, filter_sizes=[3,5,7]
updated_params_3 = {
    'batch_size': 128,
    'dropout_rate': 0.15,
    'filter_sizes': [3, 5, 7],
    'l2_lambda': 0.05,
    'learning_rate': 0.001,
    'n_conv_layers': 5,
    'n_dense_layers': 1,
    'n_dense_units': 64,
    'n_filters': 32,
    'skip_dropout_in_first_conv_layer': False,
}

# Phase 4: Remaining parameters to tune sequentially
# Testing skip_dropout_in_first_conv_layer with updated_params_3
remaining_params_grid = {
    'skip_dropout_in_first_conv_layer': [True, False],
}

