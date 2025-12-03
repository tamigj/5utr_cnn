#!/usr/bin/env python

"""
Tunable Inception-style CNN model for hyperparameter tuning.

This script accepts hyperparameters via command-line arguments and trains
an inception-style model. Designed to work with the same tuning infrastructure
as the reference model.

Usage:
    python reference_model_inception_tunable.py <param_name> <value> [other_params...]
    
    Or pass all parameters as key=value pairs:
    python reference_model_inception_tunable.py n_conv_layers=3 n_filters=64 learning_rate=0.0005 ...

Outputs saved to `/mnt/oak/users/tami/5utr_cnn/output/inception_tuning/`.
"""

import os
import sys
import ast
import argparse
import csv

# Add parent directory to path so we can import from reference_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reference_model.config import updated_params_4, DATA_DIR, OUTPUT_DIR


def sanitize_value_for_path(value):
    """Convert a parameter value to a filesystem-safe string."""
    if isinstance(value, list):
        return "_".join(sanitize_value_for_path(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).replace(".", "p").replace("-", "n").replace(" ", "")


def format_parameters_for_summary(param_items):
    """Create a readable parameter summary string."""
    formatted_parts = []
    for key, value in param_items:
        if isinstance(value, list):
            value_repr = "[" + ",".join(str(v) for v in value) + "]"
        else:
            value_repr = str(value)
        formatted_parts.append(f"{key}={value_repr}")
    return ", ".join(formatted_parts)


def build_inception_cnn_model(
    n_conv_layers,
    n_filters,
    n_dense_layers,
    n_dense_units,
    learning_rate,
    l2_lambda,
    dropout_rate,
    skip_dropout_in_first_conv_layer,
    filter_sizes,
    steps_per_epoch,
):
    """
    Build an inception-style 1D CNN model for sequence regression.

    Architecture:
      - Each conv block uses multiple parallel Conv1D branches with
        specified kernel sizes.
      - Branch outputs are concatenated along the channel dimension.
      - Uses float learning rate (not schedule) to allow ReduceLROnPlateau.

    Args:
        n_conv_layers (int): Number of inception blocks.
        n_filters (int): Total number of filters (divided across branches).
        n_dense_layers (int): Number of dense/fully connected layers.
        n_dense_units (int): Number of units in each dense layer.
        learning_rate (float): Initial learning rate for Adam optimizer.
        l2_lambda (float): L2 regularization strength.
        dropout_rate (float): Dropout rate.
        skip_dropout_in_first_conv_layer (bool): If True, skip dropout in first conv.
        filter_sizes (list): List of filter sizes for inception branches (e.g., [3, 5, 7, 9, 13]).
        steps_per_epoch (int): Number of steps per epoch (for compatibility).

    Returns:
        tf.keras.Model: Compiled Keras Model ready for training.
    """
    INPUT_SHAPE = (180, 4)  # matches reference_model.config.INPUT_SHAPE
    input_seq = tf.keras.Input(shape=INPUT_SHAPE)
    x = input_seq

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.Model(inputs=input_seq, outputs=output)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def parse_params_from_args():
    """Parse hyperparameters from command-line arguments.
    
    Supports two formats:
    1. key=value pairs: n_conv_layers=3 n_filters=64
    2. Single parameter tuning: param_name value1 value2 ...
    
    Returns:
        dict: Dictionary of hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Train tunable inception model with specified hyperparameters"
    )
    
    # Add all possible hyperparameters as optional arguments
    parser.add_argument("--n_conv_layers", type=int, help="Number of inception blocks")
    parser.add_argument("--n_filters", type=int, help="Total number of filters")
    parser.add_argument("--n_dense_layers", type=int, help="Number of dense layers")
    parser.add_argument("--n_dense_units", type=int, help="Number of units per dense layer")
    parser.add_argument("--learning_rate", type=float, help="Initial learning rate")
    parser.add_argument("--l2_lambda", type=float, help="L2 regularization strength")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate")
    parser.add_argument(
        "--skip_dropout_in_first_conv_layer",
        type=lambda x: x.lower() == "true",
        help="Skip dropout in first conv layer (True/False)",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--filter_sizes",
        type=str,
        help="Filter sizes as list, e.g., '[3,5,7,9,13]' or '3,5,7,9,13'",
    )
    parser.add_argument(
        "--reduce_lr_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (default: 0.5)",
    )
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (default: 5)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results (default: output/hyperparameter_tuning/inception)",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        help="Optional global summary.csv path (all runs append here)",
    )
    parser.add_argument(
        "--primary_param_name",
        type=str,
        help="Name of the primary parameter (used for grouping outputs)",
    )
    
    # Also support key=value format for backward compatibility
    args, unknown = parser.parse_known_args()
    
    # Parse any key=value pairs from unknown args
    params = {}
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                # Try to parse as Python literal
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If that fails, keep as string
                params[key] = value
    
    # Convert args to dict and merge with params
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    params.update(args_dict)
    
    return params


def train_inception_model(params_dict, output_dir=None, primary_param_name=None, verbose=1):
    """
    Train an inception model with given parameters.
    
    Args:
        params_dict (dict): Dictionary of hyperparameters. Must include:
            - n_conv_layers, n_filters, n_dense_layers, n_dense_units
            - learning_rate, l2_lambda, dropout_rate
            - skip_dropout_in_first_conv_layer (bool)
            - filter_sizes (list): List of filter sizes
            - batch_size
            - Optional: reduce_lr_factor, reduce_lr_patience, early_stopping_patience, max_epochs
        output_dir (str, optional): Output directory for results. If None, uses default.
        verbose (int): Verbosity level (0, 1, or 2). Default 1.
    
    Returns:
        dict: Dictionary containing:
            - history: Training history object
            - model: Trained model
            - test_loss: Test MSE
            - test_mae: Test MAE
            - min_dev_loss: Minimum validation loss
            - best_epoch: Epoch with best validation loss
            - plot_path: Path to saved training curve plot
            - results_file: Path to saved results file
    """
    # Make a copy to avoid modifying the input
    params = params_dict.copy()
    
    # Extract parameters with defaults
    batch_size = params.pop("batch_size", 64)
    filter_sizes = params.pop("filter_sizes", [3, 5, 7, 9, 13])
    reduce_lr_factor = params.pop("reduce_lr_factor", 0.5)
    reduce_lr_patience = params.pop("reduce_lr_patience", 5)
    early_stopping_patience = params.pop("early_stopping_patience", 15)
    max_epochs = params.pop("max_epochs", 100)
    primary_param_name = params.pop("primary_param_name", primary_param_name)
    summary_csv_path = params.pop("summary_csv", None)
    # In case output_dir leaked into params_dict, remove it so it doesn't affect filenames
    params.pop("output_dir", None)
    primary_param_value = None
    if primary_param_name is not None:
        primary_param_value = params_dict.get(primary_param_name)
    
    # Convert filter_sizes if it's a string
    if isinstance(filter_sizes, str):
        filter_sizes = ast.literal_eval(filter_sizes)
    
    # Load preprocessed data
    data = np.load(os.path.join(DATA_DIR, "preprocessed_data.npz"))
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_dev = data["X_dev"]
    Y_dev = data["Y_dev"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    
    steps_per_epoch = len(X_train) // batch_size
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "hyperparameter_tuning", "inception")
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose >= 1:
        print("=" * 70)
        print("Training Inception Model with Parameters:")
        print("=" * 70)
        for k, v in sorted(params.items()):
            print(f"  {k}: {v}")
        print(f"  batch_size: {batch_size}")
        print(f"  filter_sizes: {filter_sizes}")
        print(f"  reduce_lr_factor: {reduce_lr_factor}")
        print(f"  reduce_lr_patience: {reduce_lr_patience}")
        print(f"  early_stopping_patience: {early_stopping_patience}")
        print(f"  max_epochs: {max_epochs}")
        print("=" * 70)
    
    # Build model
    model = build_inception_cnn_model(
        n_conv_layers=params["n_conv_layers"],
        n_filters=params["n_filters"],
        n_dense_layers=params["n_dense_layers"],
        n_dense_units=params["n_dense_units"],
        learning_rate=params["learning_rate"],
        l2_lambda=params["l2_lambda"],
        dropout_rate=params["dropout_rate"],
        skip_dropout_in_first_conv_layer=params["skip_dropout_in_first_conv_layer"],
        filter_sizes=filter_sizes,
        steps_per_epoch=steps_per_epoch,
    )
    
    if verbose >= 1:
        model.summary()
    
    # Set up callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1 if verbose >= 1 else 0,
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=1 if verbose >= 1 else 0,
    )
    
    # Train model
    if verbose >= 1:
        print("\nStarting training...")
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_dev, Y_dev),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=verbose,
    )
    
    # Evaluate on test set
    if verbose >= 1:
        print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
    
    # Extract metrics
    train_losses = history.history["loss"]
    val_losses = history.history["val_loss"]
    min_dev_loss = min(val_losses)
    best_epoch = val_losses.index(min_dev_loss) + 1
    
    if verbose >= 1:
        print(f"\nTest Results:")
        print(f"  Test MSE: {test_loss:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Minimum dev loss: {min_dev_loss:.6f} (epoch {best_epoch})")
    
    # Create training curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2, color='blue')
    plt.plot(val_losses, label="Validation Loss", linewidth=2, color='orange')
    plt.axvline(x=best_epoch - 1, color='red', linestyle='--', alpha=0.7, 
                label=f'Best epoch ({best_epoch})')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(f"Inception Model: Training and Validation Loss\n(Min Dev Loss: {min_dev_loss:.4f})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    max_loss = max(max(train_losses), max(val_losses))
    plt.ylim((0, max_loss * 1.1))
    
    file_params = dict(params)
    file_params["batch_size"] = batch_size
    file_params["filter_sizes"] = filter_sizes
    suffix_parts = [
        f"{k}_{sanitize_value_for_path(v)}"
        for k, v in sorted(file_params.items())
        if k != primary_param_name
    ]
    if not suffix_parts:
        suffix_parts.append("default")
    param_suffix = "_".join(suffix_parts)

    plot_path = os.path.join(output_dir, f"training_curve_{param_suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose >= 1:
        print(f"Saved training curve to: {plot_path}")
    
    # Save results summary
    results_summary = {
        "test_mse": float(test_loss),
        "test_mae": float(test_mae),
        "best_val_loss": float(min_dev_loss),
        "best_epoch": best_epoch,
        "final_val_loss": float(val_losses[-1]),
        "n_epochs": len(history.history["loss"]),
        **params,
        "batch_size": batch_size,
        "filter_sizes": filter_sizes,
    }
    
    # Create a filename based on parameters
    results_file = os.path.join(output_dir, f"results_{param_suffix}.txt")
    
    with open(results_file, "w") as f:
        f.write("Inception Model Training Results\n")
        f.write("=" * 50 + "\n\n")
        for k, v in sorted(results_summary.items()):
            f.write(f"{k}: {v}\n")
    
    # Local summary (per-output_dir)
    local_summary_path = os.path.join(output_dir, "summary.csv")
    local_exists = os.path.exists(local_summary_path)
    summary_params = dict(file_params)
    if primary_param_name is not None and primary_param_value is not None:
        summary_params[primary_param_name] = primary_param_value
    summary_items = sorted(summary_params.items())
    parameters_summary = format_parameters_for_summary(summary_items)
    with open(local_summary_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not local_exists:
            writer.writerow(["parameters", "min_dev_loss"])
        writer.writerow([parameters_summary, f"{min_dev_loss:.6f}"])

    # Global summary (shared across runs, if requested)
    if summary_csv_path:
        global_exists = os.path.exists(summary_csv_path)
        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        with open(summary_csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not global_exists:
                writer.writerow(["parameters", "min_dev_loss"])
            writer.writerow([parameters_summary, f"{min_dev_loss:.6f}"])

    if verbose >= 1:
        print(f"\nResults saved to: {results_file}")
        print("=" * 70)
    
    return {
        "history": history,
        "model": model,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "min_dev_loss": min_dev_loss,
        "best_epoch": best_epoch,
        "plot_path": plot_path,
        "results_file": results_file,
    }


def main():
    # Parse parameters from command line
    user_params = parse_params_from_args()
    
    # Extract special CLI-only params BEFORE merging into base_params
    output_dir = user_params.pop("output_dir", None)
    summary_csv = user_params.pop("summary_csv", None)
    primary_param_name = user_params.pop("primary_param_name", None)
    
    # Start with base parameters from v2 (good defaults)
    base_params = {
        "n_conv_layers": 3,
        "n_filters": 64,
        "n_dense_layers": 2,
        "n_dense_units": 64,
        "learning_rate": 0.0005,
        "l2_lambda": 0.1,
        "dropout_rate": 0.4,
        "skip_dropout_in_first_conv_layer": True,
        "filter_sizes": [3, 5, 7, 9, 13],  # Default filter sizes
        "batch_size": 64,
    }
    
    # Update with user-provided parameters
    base_params.update(user_params)
    
    # Attach summary_csv to params so train_inception_model can see it
    if summary_csv is not None:
        base_params["summary_csv"] = summary_csv

    train_inception_model(
        base_params,
        output_dir=output_dir,
        primary_param_name=primary_param_name,
        verbose=1,
    )


if __name__ == "__main__":
    main()

