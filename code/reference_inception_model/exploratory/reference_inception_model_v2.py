#!/usr/bin/env python

"""
Reference Inception-style CNN model - Improved version.

Improvements over tuned version:
  - ReduceLROnPlateau callback for adaptive learning rate (replaces exponential decay)
  - More filters: 64 (vs 50) - gives each inception branch more capacity
  - Wider dense layers: 64 units (vs 50) - better leverage inception features
  - Same good regularization: l2=0.1, lr=0.0005

Outputs saved to `/mnt/oak/users/tami/5utr_cnn/output/inception_model_v2`.
"""

import os
import sys

# Add parent directory to path so we can import from reference_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reference_model.config import updated_params_4, DATA_DIR, OUTPUT_DIR


def build_inception_cnn_model(
    n_conv_layers,
    n_filters,
    n_dense_layers,
    n_dense_units,
    learning_rate,
    decay_rate,
    epoch_decay_interval,
    l2_lambda,
    dropout_rate,
    skip_dropout_in_first_conv_layer,
    steps_per_epoch,
):
    """
    Build an inception-style 1D CNN model for sequence regression.

    Differences from the baseline CNN:
      - Each conv block uses multiple parallel Conv1D branches with
        kernel sizes [3, 5, 7, 9, 13].
      - Branch outputs are concatenated along the channel dimension.
    """
    INPUT_SHAPE = (180, 4)  # matches reference_model.config.INPUT_SHAPE
    input_seq = tf.keras.Input(shape=INPUT_SHAPE)
    x = input_seq

    filter_sizes = [3, 5, 7, 9, 13]
    n_branches = len(filter_sizes)
    n_filters_per_branch = max(1, n_filters // n_branches)

    for i in range(n_conv_layers):
        current_dropout = (
            0.0 if (i == 0 and skip_dropout_in_first_conv_layer) else dropout_rate
        )

        branches = []
        for k in filter_sizes:
            b = tf.keras.layers.Conv1D(
                n_filters_per_branch,
                k,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
            )(x)
            b = tf.keras.layers.BatchNormalization()(b)
            b = tf.keras.layers.Activation("relu")(b)
            b = tf.keras.layers.Dropout(current_dropout)(b)
            branches.append(b)

        x = tf.keras.layers.Concatenate(axis=-1)(branches)

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense blocks
    for _ in range(n_dense_layers):
        x = tf.keras.layers.Dense(
            n_dense_units,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    output = tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(x)

    # Use float learning rate (not schedule) to allow ReduceLROnPlateau to work
    # ReduceLROnPlateau will handle adaptive learning rate reduction
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.Model(inputs=input_seq, outputs=output)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def main():
    # Output directory for v2 inception model artifacts
    inception_dir = os.path.join(OUTPUT_DIR, "inception_model_v2")
    os.makedirs(inception_dir, exist_ok=True)

    # Load preprocessed data from reference_model
    data = np.load(os.path.join(DATA_DIR, "preprocessed_data.npz"))
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_dev = data["X_dev"]
    Y_dev = data["Y_dev"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    # Extract base parameters and override with improved values
    params = updated_params_4.copy()
    batch_size = params.pop("batch_size")
    steps_per_epoch = len(X_train) // batch_size

    # Improved hyperparameters
    params["learning_rate"] = 0.0005  # Lower LR (worked well)
    params["l2_lambda"] = 0.1  # Higher L2 (worked well)
    params["n_filters"] = 64  # More filters (was 50) - gives each branch ~13 filters
    params["n_dense_units"] = 64  # Wider dense layers (was 50)

    print("Using improved parameters:")
    print(f"  learning_rate: {params['learning_rate']} (was {updated_params_4['learning_rate']})")
    print(f"  l2_lambda: {params['l2_lambda']} (was {updated_params_4['l2_lambda']})")
    print(f"  n_filters: {params['n_filters']} (was {updated_params_4['n_filters']})")
    print(f"  n_dense_units: {params['n_dense_units']} (was {updated_params_4['n_dense_units']})")
    for k, v in params.items():
        if k not in ["learning_rate", "l2_lambda", "n_filters", "n_dense_units"]:
            print(f"  {k}: {v}")

    # Build inception model
    model = build_inception_cnn_model(
        n_conv_layers=params["n_conv_layers"],
        n_filters=params["n_filters"],
        n_dense_layers=params["n_dense_layers"],
        n_dense_units=params["n_dense_units"],
        learning_rate=params["learning_rate"],
        decay_rate=params["decay_rate"],
        epoch_decay_interval=params["epoch_decay_interval"],
        l2_lambda=params["l2_lambda"],
        dropout_rate=params["dropout_rate"],
        skip_dropout_in_first_conv_layer=params["skip_dropout_in_first_conv_layer"],
        steps_per_epoch=steps_per_epoch,
    )

    model.summary()

    # Train with early stopping + ReduceLROnPlateau (max 100 epochs)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_dev, Y_dev),
        epochs=100,  # Max epochs (early stopping will stop earlier if no improvement)
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)

    # Compute R² on test set (fixed: flatten predictions)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_flat = y_pred.flatten()  # Ensure 1D array
    Y_test_flat = Y_test.flatten() if Y_test.ndim > 1 else Y_test
    
    ss_res = np.sum((Y_test_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((Y_test_flat - np.mean(Y_test_flat)) ** 2)
    
    # Safety check: avoid division by zero
    if ss_tot < 1e-10:
        r2 = np.nan
        print("Warning: ss_tot is too small, R² cannot be computed")
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    
    print(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R²: {r2:.4f}")

    # Plot training vs dev loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.ylim((0, 3))
    plt.title(f"Inception CNN (v2): Training and Validation Loss (R²={r2:.3f})")
    plt.legend()
    plt.savefig(os.path.join(inception_dir, "history.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {inception_dir}/history.png")

    # Save model
    model_path = os.path.join(inception_dir, "model.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

