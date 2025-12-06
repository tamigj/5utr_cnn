#!/usr/bin/env python

"""
Configuration for Inception-Transformer model.

Defines default parameters and paths.
"""

# Paths
BASE_DIR = '/mnt/oak/users/tami/5utr_cnn'
DATA_DIR = f'{BASE_DIR}/data'
OUTPUT_DIR = f'{BASE_DIR}/output'

# Constants
MAX_SEQ_LEN = 180
N_BASES = 4
INPUT_SHAPE = (MAX_SEQ_LEN, N_BASES)
NUM_EPOCH = 100

# Train/dev/test set
train_prop = 0.7
dev_prop = 0.2
test_prop = 0.1

# Default parameters for Inception-Transformer model
# Based on best Inception model, with transformer-specific additions
default_params = {
    # Inception CNN parameters
    'n_conv_layers': 5,
    'n_filters': 32,
    'filter_sizes': [3, 5, 7],
    'skip_dropout_in_first_conv_layer': False,
    
    # Transformer parameters
    'n_transformer_layers': 2,
    'num_heads': 4,
    'ff_dim': 128,
    
    # Dense head parameters
    'n_dense_layers': 1,
    'n_dense_units': 64,
    
    # Training parameters
    'learning_rate': 0.001,
    'l2_lambda': 0.05,
    'dropout_rate': 0.15,
    'batch_size': 128,
}

