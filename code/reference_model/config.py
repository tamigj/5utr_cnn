#!/usr/bin/env python

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
    'learning_rate': 0.001,
    'n_conv_layers': 3,
    'n_filters': 128,
    'filter_size': 7,
    'n_dense_layers': 2,
    'n_dense_units': 150,
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
updated_params_1 = {
    'learning_rate': 0.001,
    'n_conv_layers': 2,
    'n_filters': 10,
    'filter_size': 7,
    'n_dense_layers': 2,
    'n_dense_units': 25,
    'decay_rate': 0.95,
    'epoch_decay_interval': 50,
    'l2_lambda': 0.05,
    'dropout_rate': 0.4,
    'skip_dropout_in_first_conv_layer': True,
    'batch_size': 64
}

# Combinatorial tuning grid (used by submit_scripts/4_generate_combinatorial_scripts.py)
combinatorial_tuning_grid_learning = {
    'learning_rate': [0.001, 0.005, 0.01, 0.015],
    'l2_lambda': [0.01, 0.05, 0.1, 0.15],
    'dropout_rate': [0.1, 0.4, 0.5],
    'batch_size': [64, 128],
}

# Base parameters for combinatorial architecture tuning
# (after fixing learning-related hyperparameters - 
# lr=0.001, lambda=0.05, dropout=0.4, batch_size=64)
updated_params_2 = {
    'learning_rate': 0.001,
    'n_conv_layers': 2,
    'n_filters': 10,
    'filter_size': 7,
    'n_dense_layers': 2,
    'n_dense_units': 25,
    'decay_rate': 0.95,
    'epoch_decay_interval': 50,
    'l2_lambda': 0.05,
    'dropout_rate': 0.4,
    'skip_dropout_in_first_conv_layer': True,
    'batch_size': 64
}

combinatorial_tuning_grid_architecture = {
    'n_conv_layers': [1, 2, 3, 7, 10],
    'n_filters': [5, 10, 15, 50],
    'filter_size': [3, 5, 7, 13],
    'n_dense_units': [25, 50, 100],
}

# Base parameters for final tuning 
# (after fixing architecture-related hyperaparameters)
updated_params_3 = {
    'learning_rate': 0.001,
    'n_conv_layers': 2,
    'n_filters': 10,
    'filter_size': 7,
    'n_dense_layers': 2,
    'n_dense_units': 25,
    'decay_rate': 0.95,
    'epoch_decay_interval': 50,
    'l2_lambda': 0.05,
    'dropout_rate': 0.4,
    'skip_dropout_in_first_conv_layer': True,
    'batch_size': 64
}

# Final hyperparameters (TO BE DETERMINED)
updated_params_4 = {
    'learning_rate': 0.001,
    'n_conv_layers': 3,
    'n_filters': 50,
    'filter_size': 7,
    'n_dense_layers': 2,
    'n_dense_units': 50,
    'decay_rate': 0.95,
    'epoch_decay_interval': 50,
    'l2_lambda': 0.05,
    'dropout_rate': 0.4,
    'skip_dropout_in_first_conv_layer': True,
    'batch_size': 64
}