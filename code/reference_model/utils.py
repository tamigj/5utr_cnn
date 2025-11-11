#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from config import MAX_SEQ_LEN, N_BASES, INPUT_SHAPE, NUM_EPOCH, OUTPUT_TUNING_DIR

"""
DATA PRE-PROCESSING:
  * pad_sequence() - pad sequences with NN to the left up to MAX_SEQ_LEN
  * one_hot_encode() - encoding for input sequences to shape (MAX_SEQ_LEN, N_BASES)
  * split_data() - split data into train, dev, and test set
  * prepare_xy() - prepare inputs and outputs for training and testing
"""

def pad_sequence(seq, target_len=MAX_SEQ_LEN):
    seq = seq.upper()
    if len(seq) < target_len:
        seq = "N" * (target_len - len(seq)) + seq  # pad with N's to the left
    return seq[:target_len]

def one_hot_encode(seq, target_len=MAX_SEQ_LEN, n_bases=N_BASES):
    seq = pad_sequence(seq, target_len)
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    arr = np.zeros((target_len, n_bases), dtype=np.float32)

    for i, base in enumerate(seq):
        if base in mapping:
            arr[i, mapping[base]] = 1.0
    return arr

def split_data(df, train_prop, dev_prop, test_prop, random_state=42):
  assert abs(train_prop + dev_prop + test_prop - 1.0) < 1e-6, "Proportions must sum to 1"

  # Split the data
  train_df, tmp_df = train_test_split(df, train_size=train_prop,
                                      random_state=random_state, shuffle=True)
  dev_df, test_df = train_test_split(tmp_df, test_size=test_prop,
                                     random_state=random_state, shuffle=True)
  return train_df, dev_df, test_df

def prepare_xy(df):
    X = np.stack([one_hot_encode(seq) for seq in df['ref_sequence']])
    y = df['pw_mean_translation'].values.astype(np.float32)
    return X, y


"""
BASELINE CNN MODEL

Layers include:
  - n_conv_layers convolutional blocks (CONV)
  - global pooling
  - n_dense_layers fully connected blocks (FC)

 Implementation also includes:
  - learning rate decay
  - Adam optimizer

"""

# Build a CNN model
def build_cnn_model(n_conv_layers, n_filters, filter_size,
                    n_dense_layers, n_dense_units,
                    learning_rate, decay_rate, epoch_decay_interval,
                    l2_lambda, dropout_rate, zero_dropout_in_first_layer,
                    steps_per_epoch):

  # --- Input ---
  input_seq = tf.keras.Input(shape=INPUT_SHAPE)
  x = input_seq

  # --- CONV blocks ---
  for i in range(n_conv_layers):
    current_dropout = 0 if (i == 0 and zero_dropout_in_first_layer) else dropout_rate

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

"""
HYPERPARAMETER TUNING:
  * tune_hyperparameter() - tune param_name using values param_values

"""

def tune_hyperparameter(param_name, param_values,
                        base_params, steps_per_epoch,
                        X_train, y_train, X_dev, y_dev):
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

        history = model.fit(X_train, y_train,
                           validation_data=(X_dev, y_dev),
                           batch_size=batch_size,
                           epochs=NUM_EPOCH,
                           verbose=0)

        results.append({'value': value, 'history': history})

    # Plot grid
    plot_tuning_results(param_name, results)
    return results

def plot_tuning_results(param_name, results):
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
    output_path = f'{OUTPUT_TUNING_DIR}/initial_tuning_{param_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")
