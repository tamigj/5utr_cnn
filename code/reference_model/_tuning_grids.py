#!/usr/bin/env python
"""Different hyperparameter tuning grids for different search strategies.

This is a helper module used by submit_scripts/2_generate_tuning_scripts.py
to define different hyperparameter tuning strategies.

Grids:
    - wide: Broad search across wide parameter ranges (initial exploration)
    - coarse: Narrower search around promising regions (refinement)
    - fine: Fine-grained search for final optimization
"""

# Wide search - broad ranges to explore the parameter space
wide_grid = {
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

# Coarse search - narrower ranges around promising values
# TODO: Update these ranges based on wide search results
coarse_grid = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'n_conv_layers': [2, 3, 4],
    'n_filters': [32, 64, 128],
    'filter_size': [5, 7, 9],
    'n_dense_layers': [2, 3, 4],
    'n_dense_units': [50, 100, 150],
    'decay_rate': [0.85, 0.9, 0.95],
    'epoch_decay_interval': [5, 10, 20],
    'l2_lambda': [0.01, 0.05, 0.1],
    'dropout_rate': [0.2, 0.3, 0.4],
    'skip_dropout_in_first_conv_layer': [True, False],
    'batch_size': [64, 128, 256]
}

# Fine search - very narrow ranges for final optimization
# TODO: Update these ranges based on coarse search results
fine_grid = {
    'learning_rate': [0.0008, 0.001, 0.0012],
    'n_conv_layers': [2, 3],  # Or single best value
    'n_filters': [64, 128],
    'filter_size': [7],  # Or test around best value
    'n_dense_layers': [3],  # Or test around best value
    'n_dense_units': [100, 150],
    'decay_rate': [0.9, 0.95],
    'epoch_decay_interval': [10],
    'l2_lambda': [0.05],
    'dropout_rate': [0.3, 0.4],
    'skip_dropout_in_first_conv_layer': [True],
    'batch_size': [64, 128]
}

# Map strategy names to grids
GRIDS = {
    'wide': wide_grid,
    'coarse': coarse_grid,
    'fine': fine_grid,
}

