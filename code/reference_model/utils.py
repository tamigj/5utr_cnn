#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from config import MAX_SEQ_LEN, N_BASES, INPUT_SHAPE, NUM_EPOCH

"""
Utility functions for 5' UTR CNN model training.

This module provides:
  - Data preprocessing functions (sequence padding, one-hot encoding, data splitting)
  - CNN model building functions
  - Hyperparameter tuning utilities
"""

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
        ax.set_ylim(0,3)
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
