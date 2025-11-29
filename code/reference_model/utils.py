#!/usr/bin/env python

import os
import math

import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import product

from config import (MAX_SEQ_LEN, N_BASES, INPUT_SHAPE, NUM_EPOCH,
                    OUTPUT_TUNING_DIR)

"""
Utility functions for 5' UTR CNN model training.

This module provides:
  - Data preprocessing functions (sequence padding, one-hot encoding, data splitting)
  - CNN model building functions
  - Hyperparameter tuning utilities
"""

#----------------------#
# DIRECTORY UTILITIES  #
#----------------------#

def get_tuning_output_dir(strategy='wide'):
    """Create (if needed) and return directory for tuning artifacts."""
    strategy_dir = f'{OUTPUT_TUNING_DIR}/{strategy}'
    os.makedirs(strategy_dir, exist_ok=True)
    return strategy_dir

#----------------------#
# DATA PRE-PROCESSING  #
#----------------------#

def pad_sequence(seq, target_len=MAX_SEQ_LEN):
    """Pad DNA sequence to target length with 'N' bases on the left.
    
    Args:
        seq (str): DNA sequence string to pad.
        target_len (int, optional): Target length for the sequence.
            Defaults to MAX_SEQ_LEN.
    
    Returns:
        str: Padded sequence string of length target_len. If input is longer
            than target_len, it is truncated to target_len.
    """
    seq = seq.upper()
    if len(seq) < target_len:
        seq = "N" * (target_len - len(seq)) + seq  # pad with N's to the left
    return seq[:target_len]

def one_hot_encode(seq, target_len=MAX_SEQ_LEN, n_bases=N_BASES):
    """Convert DNA sequence to one-hot encoded array.
    
    Args:
        seq (str): DNA sequence string to encode.
        target_len (int, optional): Target length for the sequence.
            Defaults to MAX_SEQ_LEN.
        n_bases (int, optional): Number of bases to encode.
            Defaults to N_BASES (i.e., 4 for A, C, G, T).
    
    Returns:
        np.ndarray: One-hot encoded numpy array of shape (target_len, n_bases)
            with dtype float32. Bases A, C, G, T are mapped to positions
            0, 1, 2, 3 respectively. Unknown bases are encoded as all zeros.
    """
    seq = pad_sequence(seq, target_len)
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    arr = np.zeros((target_len, n_bases), dtype=np.float32)

    for i, base in enumerate(seq):
        if base in mapping:
            arr[i, mapping[base]] = 1.0
    return arr

def split_data(df, train_prop, dev_prop, test_prop, random_state=42):
    """Split dataframe into train, dev, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame to split.
        train_prop (float): Proportion of data for training set 
            (must sum to 1.0 with dev_prop and test_prop).
        dev_prop (float): Proportion of data for development/validation set.
        test_prop (float): Proportion of data for test set.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
    
    Returns:
        tuple: Tuple of (train_df, dev_df, test_df) DataFrames with proportions
            matching train_prop, dev_prop, and test_prop of the original data.
    
    Raises:
        AssertionError: If proportions do not sum to 1.0.
    """
    assert abs(train_prop + dev_prop + test_prop - 1.0) < 1e-6, "Proportions must sum to 1"

    # Split the data
    train_df, tmp_df = train_test_split(df, train_size=train_prop,
                                        random_state=random_state, shuffle=True)
    # Calculate test_size relative to tmp_df (not original df)
    test_size_in_tmp = test_prop / (dev_prop + test_prop)
    dev_df, test_df = train_test_split(tmp_df, test_size=test_size_in_tmp,
                                       random_state=random_state, shuffle=True)
    return train_df, dev_df, test_df

def prepare_xy(df):
    """Prepare input features (X) and target values (Y) from dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'ref_sequence' and
            'pw_mean_translation' columns.
    
    Returns:
        tuple: Tuple of (X, Y) where:
            - X (np.ndarray): Numpy array of one-hot encoded sequences,
                shape (n_samples, MAX_SEQ_LEN, N_BASES).
            - Y (np.ndarray): Numpy array of translation efficiency values,
                shape (n_samples,), dtype float32.
    """
    X = np.stack([one_hot_encode(seq) for seq in df['ref_sequence']])
    Y = df['pw_mean_translation'].values.astype(np.float32)
    return X, Y


#------------------------#
# BASELINE CNN MODEL     #
#------------------------#
# Layers include:
#   - n_conv_layers convolutional blocks (CONV)
#   - global pooling
#   - n_dense_layers fully connected blocks (FC)
# Implementation also includes:
#   - learning rate decay
#   - Adam optimizer

def build_cnn_model(n_conv_layers, n_filters, filter_size,
                    n_dense_layers, n_dense_units,
                    learning_rate, decay_rate, epoch_decay_interval,
                    l2_lambda, dropout_rate, skip_dropout_in_first_conv_layer,
                    steps_per_epoch):
    """Build and compile a 1D CNN model for sequence regression.
    
    Architecture:
        - n_conv_layers convolutional blocks (Conv1D -> BatchNorm -> ReLU -> Dropout)
        - GlobalAveragePooling1D
        - n_dense_layers fully connected blocks (Dense -> BatchNorm -> ReLU -> Dropout)
        - Final Dense(1) output layer for regression
    
    Args:
        n_conv_layers (int): Number of convolutional layers.
        n_filters (int): Number of filters in each convolutional layer.
        filter_size (int): Size of convolutional filters.
        n_dense_layers (int): Number of dense/fully connected layers.
        n_dense_units (int): Number of units in each dense layer.
        learning_rate (float): Initial learning rate for Adam optimizer.
        decay_rate (float): Decay rate for exponential learning rate decay.
        epoch_decay_interval (int): Number of epochs between learning rate decay steps.
        l2_lambda (float): L2 regularization strength for kernel regularizers.
        dropout_rate (float): Dropout rate (applied to all layers except
            optionally first conv).
        skip_dropout_in_first_conv_layer (bool): If True, skip dropout in
            first convolutional layer.
        steps_per_epoch (int): Number of steps per epoch (used for learning
            rate decay schedule).
    
    Returns:
        tf.keras.Model: Compiled Keras Model ready for training with MSE loss
            and MAE metric.
    """
    # --- Input ---
    input_seq = tf.keras.Input(shape=INPUT_SHAPE)
    x = input_seq

    # --- CONV blocks ---
    for i in range(n_conv_layers):
        current_dropout = 0 if (i == 0 and skip_dropout_in_first_conv_layer) else dropout_rate

        x = tf.keras.layers.Conv1D(n_filters, filter_size, padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(current_dropout)(x)

    # --- Global pooling ---
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # --- FC blocks ---
    for i in range(n_dense_layers):
        x = tf.keras.layers.Dense(n_dense_units,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    output = tf.keras.layers.Dense(1,
                                   kernel_regularizer=regularizers.l2(l2_lambda))(x)

    # --- Learning rate decay ---
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=steps_per_epoch * epoch_decay_interval,
        decay_rate=decay_rate,
        staircase=True
    )

    # --- Build model ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = tf.keras.Model(inputs=input_seq, outputs=output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

#-----------------#
# EARLY STOPPING  #
#-----------------#

def get_early_stopping_callback(patience=15, min_delta=0.001, monitor='val_loss', verbose=1):
    """Create an EarlyStopping callback for model training.
    
    Early stopping monitors a metric (typically validation loss) and stops
    training if the metric doesn't improve for a specified number of epochs.
    This prevents overfitting and saves computation time. The best model weights
    are automatically restored when training stops.
    
    Args:
        patience (int, optional): Number of epochs with no improvement after
            which training will be stopped. Defaults to 15.
        min_delta (float, optional): Minimum change in the monitored metric
            to qualify as an improvement. Defaults to 0.001.
        monitor (str, optional): Metric to monitor. Defaults to 'val_loss'.
        verbose (int, optional): Verbosity mode (0=silent, 1=verbose).
            Defaults to 1.
    
    Returns:
        tf.keras.callbacks.EarlyStopping: Configured EarlyStopping callback
            that can be passed to model.fit() callbacks argument.
    
    Example:
        >>> callback = get_early_stopping_callback(patience=10)
        >>> history = model.fit(X_train, Y_train,
        ...                     validation_data=(X_dev, Y_dev),
        ...                     callbacks=[callback])
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,  # Restore weights from best epoch
        verbose=verbose
    )

#--------------------------#
# HYPERPARAMETER TUNING    #
#--------------------------#
# Functions for tuning hyperparameters and visualizing results

def tune_hyperparameter(param_name, param_values,
                        base_params, steps_per_epoch,
                        X_train, Y_train, X_dev, Y_dev,
                        output_dir, strategy='initial'):
    """Tune a single hyperparameter by training models with different values.
    
    For each value in param_values, trains a model with that hyperparameter value
    (keeping all other parameters from base_params) and collects training history.
    Automatically generates and saves a plot comparing the results.
    
    Args:
        param_name (str): Name of the hyperparameter to tune (must be a key
            in base_params).
        param_values (list): List of values to try for the hyperparameter.
        base_params (dict): Dictionary of base hyperparameters (will be
            modified for each trial).
        steps_per_epoch (int): Number of steps per epoch for learning rate decay.
        X_train (np.ndarray): Training input features.
        Y_train (np.ndarray): Training target values.
        X_dev (np.ndarray): Development/validation input features.
        Y_dev (np.ndarray): Development/validation target values.
        output_dir (str): Directory to save the tuning plot.
        strategy (str, optional): Strategy name for filename. Defaults to 'initial'.
    
    Returns:
        list: List of dictionaries, each containing:
            - 'value': The hyperparameter value that was tested.
            - 'history': Keras History object from model training.
    """
    results = []

    print("Tuning hyperparameter " + str(param_name))

    for value in param_values:
        print("Trying value " + str(value))

        # Update only the param being tuned
        current_params = base_params.copy()
        current_params[param_name] = value

        # Extract batch size separately
        batch_size = current_params.pop("batch_size")

        # Build and train model
        model = build_cnn_model(**current_params,
                                steps_per_epoch=steps_per_epoch)

        history = model.fit(X_train, Y_train,
                           validation_data=(X_dev, Y_dev),
                           batch_size=batch_size,
                           epochs=NUM_EPOCH,
                           verbose=0)

        results.append({'value': value, 'history': history})

    # Plot grid
    plot_tuning_results(param_name, results, output_dir, strategy)
    return results

def plot_tuning_results(param_name, results, output_dir, strategy='initial'):
    """Create and save a grid plot comparing hyperparameter tuning results.
    
    Generates a subplot grid showing training and validation loss curves for each
    hyperparameter value tested. Saves the plot to the specified output directory.
    
    Args:
        param_name (str): Name of the hyperparameter that was tuned (used in
            title and filename).
        results (list): List of result dictionaries from tune_hyperparameter(),
            each containing 'value' and 'history' keys.
        output_dir (str): Directory to save the plot.
        strategy (str, optional): Strategy name for filename. Defaults to 'initial'.
    
    Note:
        The plot is saved as: {output_dir}/{strategy}_{param_name}.png
    """
    n_values = len(results)

    # Create grid: 1 row if <= 3 values, otherwise 2 rows
    if n_values <= 3:
        n_rows, n_cols = 1, n_values
    else:
        n_rows = 2
        n_cols = (n_values + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    # Handle single subplot case
    if n_values == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]

        # Plot training and validation loss
        ax.plot(result['history'].history['loss'], label='Training Loss', color='blue')
        ax.plot(result['history'].history['val_loss'], label='Validation Loss', color='orange')

        # Formatting
        ax.set_title(f"{param_name} = {result['value']}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_ylim(bottom=0, top=_get_loss_upper_bound(result['history']))
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_values, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = f'{output_dir}/{strategy}_{param_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

#-----------------------------------#
# COMBINATORIAL HYPERPARAMETER TUNING
#-----------------------------------#
# Functions for tuning multiple hyperparameters combinatorially

def tune_hyperparameter_combinations(param_grid, base_params, steps_per_epoch,
                                     X_train, Y_train, X_dev, Y_dev,
                                     output_dir, strategy='combinatorial'):
    """Tune multiple hyperparameters combinatorially by training models with all combinations.
    
    For each combination of parameter values in param_grid, trains a model with those
    hyperparameter values (keeping all other parameters from base_params) and collects
    training history. Automatically generates and saves plots comparing the results.
    
    Args:
        param_grid (dict): Dictionary mapping parameter names to lists of values to try.
            Example: {'n_filters': [32, 64, 128], 'learning_rate': [0.001, 0.01]}
        base_params (dict): Dictionary of base hyperparameters (will be modified
            for each combination).
        steps_per_epoch (int): Number of steps per epoch for learning rate decay.
        X_train (np.ndarray): Training input features.
        Y_train (np.ndarray): Training target values.
        X_dev (np.ndarray): Development/validation input features.
        Y_dev (np.ndarray): Development/validation target values.
        output_dir (str): Directory to save the tuning plots and summary.
        strategy (str, optional): Strategy name for filename. Defaults to 'combinatorial'.
    
    Returns:
        list: List of dictionaries, each containing:
            - 'params': Dictionary of all parameter values that were tested.
            - 'history': Keras History object from model training.
            - 'best_val_loss': Minimum validation loss achieved during training.
            - 'final_val_loss': Final validation loss at end of training.
    
    Note:
        The number of combinations grows exponentially. For example, 3 parameters
        with 5 values each = 125 combinations. Consider using smaller grids or
        early stopping strategies for large searches.
    """
    results = []
    param_names = list(param_grid.keys())
    param_value_lists = list(param_grid.values())
    
    # Generate all combinations
    combinations = list(product(*param_value_lists))
    n_combinations = len(combinations)
    
    print(f"Tuning {len(param_names)} hyperparameters combinatorially")
    print(f"Parameter names: {param_names}")
    print(f"Total combinations: {n_combinations}")
    
    for combo_idx, combo in enumerate(combinations, 1):
        # Create parameter dict for this combination
        current_params = base_params.copy()
        param_dict = {}
        for param_name, value in zip(param_names, combo):
            current_params[param_name] = value
            param_dict[param_name] = value
        
        print(f"[{combo_idx}/{n_combinations}] Trying combination: {param_dict}")
        
        # Extract batch size separately
        batch_size = current_params.pop("batch_size")
        
        # Build and train model
        model = build_cnn_model(**current_params,
                                steps_per_epoch=steps_per_epoch)
        
        # Use early stopping for combinatorial tuning to save time
        early_stopping = get_early_stopping_callback(patience=15, min_delta=0.001, verbose=0)
        
        history = model.fit(X_train, Y_train,
                           validation_data=(X_dev, Y_dev),
                           batch_size=batch_size,
                           epochs=NUM_EPOCH,
                           callbacks=[early_stopping],
                           verbose=0)
        
        # Extract key metrics
        best_val_loss = min(history.history['val_loss'])
        final_val_loss = history.history['val_loss'][-1]
        
        results.append({
            'params': param_dict,
            'history': history,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss
        })
    
    # Generate visualizations
    plot_combinatorial_results(param_grid, results, output_dir, strategy)
    
    # Save summary table
    save_combinatorial_summary(param_grid, results, output_dir, strategy)
    
    return results

def plot_combinatorial_results(param_grid, results, output_dir, strategy='combinatorial'):
    """Create and save plots comparing combinatorial hyperparameter tuning results.
    
    For 1-2 parameters: Creates grid plots showing training curves.
    For 3+ parameters: Creates summary heatmaps and top combinations table.
    
    Args:
        param_grid (dict): Dictionary mapping parameter names to lists of values.
        results (list): List of result dictionaries from tune_hyperparameter_combinations().
        output_dir (str): Directory to save the plots.
        strategy (str, optional): Strategy name for filename. Defaults to 'combinatorial'.
    """
    param_names = list(param_grid.keys())
    n_params = len(param_names)
    
    if n_params == 1:
        # Single parameter: use existing plotting function
        # Convert results format
        simple_results = [{'value': r['params'][param_names[0]], 'history': r['history']} 
                         for r in results]
        plot_tuning_results(param_names[0], simple_results, output_dir, strategy)
    
    elif n_params == 2:
        # Two parameters: create grid of subplots
        param1_name, param2_name = param_names
        param1_values = param_grid[param1_name]
        param2_values = param_grid[param2_name]
        
        n_rows, n_cols = len(param2_values), len(param1_values)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Create results lookup
        results_dict = {(r['params'][param1_name], r['params'][param2_name]): r 
                       for r in results}
        
        for i, p2_val in enumerate(param2_values):
            for j, p1_val in enumerate(param1_values):
                ax = axes[i, j]
                result = results_dict.get((p1_val, p2_val))
                
                if result:
                    ax.plot(result['history'].history['loss'], 
                           label='Training Loss', color='blue')
                    ax.plot(result['history'].history['val_loss'], 
                           label='Validation Loss', color='orange')
                    ax.set_title(f"{param1_name}={p1_val}\n{param2_name}={p2_val}")
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('MSE Loss')
                    ax.set_ylim(bottom=0, top=_get_loss_upper_bound(result['history']))
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
        
        plt.tight_layout()
        output_path = f'{output_dir}/{strategy}_combinatorial_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grid plot to {output_path}")
        
        # Also create heatmap of best validation losses
        val_loss_matrix = np.full((len(param2_values), len(param1_values)), np.nan)
        for i, p2_val in enumerate(param2_values):
            for j, p1_val in enumerate(param1_values):
                result = results_dict.get((p1_val, p2_val))
                if result:
                    val_loss_matrix[i, j] = result['best_val_loss']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(val_loss_matrix, aspect='auto', cmap='viridis_r')
        ax.set_xticks(range(len(param1_values)))
        ax.set_xticklabels([str(v) for v in param1_values])
        ax.set_yticks(range(len(param2_values)))
        ax.set_yticklabels([str(v) for v in param2_values])
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title('Best Validation Loss Heatmap')
        plt.colorbar(im, ax=ax, label='Validation Loss')
        
        # Add text annotations
        for i in range(len(param2_values)):
            for j in range(len(param1_values)):
                if not np.isnan(val_loss_matrix[i, j]):
                    text = ax.text(j, i, f'{val_loss_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        output_path = f'{output_dir}/{strategy}_combinatorial_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {output_path}")
    
    else:
        # Three or more parameters: create grid plot for all combinations
        n_combinations = len(results)
        if n_combinations == 0:
            print("No combinatorial results to plot.")
            return
        
        n_cols = math.ceil(math.sqrt(n_combinations))
        n_rows = math.ceil(n_combinations / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten()
        
        # Sort results by best validation (dev) loss for better visualization
        sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
        
        for idx, result in enumerate(sorted_results):
            ax = axes[idx]
            _plot_history_curves(ax, result)
        
        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = f'{output_dir}/{strategy}_combinatorial_all_combinations.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved all combinations plot to {output_path}")
        
        # Also save standalone overview plots
        plot_top_combinations(sorted_results, output_dir, strategy)
        plot_combination_histogram(sorted_results, output_dir, strategy)

def save_combinatorial_summary(param_grid, results, output_dir, strategy='combinatorial'):
    """Save a summary table of all combinatorial tuning results.
    
    Args:
        param_grid (dict): Dictionary mapping parameter names to lists of values.
        results (list): List of result dictionaries from tune_hyperparameter_combinations().
        output_dir (str): Directory to save the summary CSV.
        strategy (str, optional): Strategy name for filename. Defaults to 'combinatorial'.
    """
    # Create DataFrame with all results
    rows = []
    for result in results:
        row = result['params'].copy()
        row['best_val_loss'] = result['best_val_loss']
        row['final_val_loss'] = result['final_val_loss']
        row['best_train_loss'] = min(result['history'].history['loss'])
        row['final_train_loss'] = result['history'].history['loss'][-1]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('best_val_loss')
    
    # Save to CSV
    output_path = f'{output_dir}/{strategy}_combinatorial_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved summary table to {output_path}")
    print(f"\nTop 5 combinations by best validation loss:")
    print(df.head(5).to_string(index=False))


def plot_top_combinations(results, output_dir, strategy='combinatorial', top_k=10):
    """Save a standalone bar chart for the top-K combinations (dev loss)."""
    if not results:
        return
    
    sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
    top_results = sorted_results[:top_k]
    
    labels = [_format_params_for_label(res['params']) for res in top_results]
    best_losses = [res['best_val_loss'] for res in top_results]
    
    fig_height = max(4, 0.6 * len(top_results))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = np.arange(len(top_results))
    ax.barh(y_pos, best_losses, color='seagreen')
    ax.set_yticks(y_pos)
    bar_labels = []
    for idx, res in enumerate(top_results, 1):
        bar_labels.append(f"#{idx}: {labels[idx-1]}")
    ax.set_yticklabels(bar_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Best Dev (Validation) Loss')
    ax.set_xlim(left=0)
    ax.set_title(f'Top {len(top_results)} combinations (dev loss)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Annotate bars with best/final dev losses
    max_loss = max(best_losses) if best_losses else 0
    offset = 0.02 * max_loss if max_loss > 0 else 0.02
    for idx, res in enumerate(top_results):
        text = (f"best={res['best_val_loss']:.3f}, "
                f"final={res['final_val_loss']:.3f}")
        ax.text(best_losses[idx] + offset,
                idx, text, va='center', fontsize=7, color='black')
    
    plt.tight_layout()
    output_path = f'{output_dir}/{strategy}_combinatorial_top_combinations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top combinations plot to {output_path}")


def plot_combination_histogram(results, output_dir, strategy='combinatorial'):
    """Save histogram of best dev (validation) losses."""
    if not results:
        return
    
    best_losses = [res['best_val_loss'] for res in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(best_losses, bins=min(20, max(5, len(best_losses)//2)),
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Best Dev (Validation) Loss')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of best dev losses across combinations')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/{strategy}_combinatorial_dev_loss_hist.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dev loss histogram to {output_path}")


#----------------------#
# INTERNAL UTILITIES   #
#----------------------#

def _get_loss_upper_bound(history):
    """Return a y-axis upper bound (>0) so axes always start at zero."""
    combined = []
    for key in ('loss', 'val_loss'):
        combined.extend(history.history.get(key, []))
    if not combined:
        return 1.0
    max_loss = max(combined)
    if max_loss <= 0:
        return 1.0
    return max_loss * 1.05


def _format_params_for_label(params):
    """Format parameter dict into a single short string."""
    return ', '.join([f'{k}={v}' for k, v in params.items()])


def _plot_history_curves(ax, result):
    """Render training vs validation curves for a single combination."""
    ax.plot(result['history'].history['loss'], label='Training Loss', color='blue')
    ax.plot(result['history'].history['val_loss'], label='Validation Loss', color='orange')
    param_str = _format_params_for_label(result['params'])
    ax.set_title(param_str[:60], fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_ylim(bottom=0, top=_get_loss_upper_bound(result['history']))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


