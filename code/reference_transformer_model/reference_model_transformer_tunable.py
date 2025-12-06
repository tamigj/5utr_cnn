#!/usr/bin/env python

"""
Tunable Inception-CNN + Transformer model for hyperparameter tuning.

Architecture:
  - Inception-style CNN blocks for multi-scale feature extraction
  - Transformer encoder for sequential modeling and long-range dependencies
  - Dense regression head

Usage:
    python reference_model_transformer_tunable.py <param_name> <value> [other_params...]
    
    Or pass all parameters as key=value pairs:
    python reference_model_transformer_tunable.py n_conv_layers=3 n_filters=64 learning_rate=0.0005 ...
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
from sklearn.metrics import r2_score

from reference_model.config import DATA_DIR, OUTPUT_DIR


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


def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate, l2_lambda):
    """
    Transformer encoder block with multi-head attention and feed-forward network.
    
    Args:
        inputs: Input tensor
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        dropout_rate: Dropout rate
        l2_lambda: L2 regularization strength
    
    Returns:
        Output tensor after transformer block
    """
    # Get input dimension for key_dim calculation
    embed_dim = inputs.shape[-1]
    key_dim = embed_dim // num_heads
    
    # Multi-head self-attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(inputs, inputs)
    
    # Add & Norm
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ffn_output = tf.keras.layers.Dense(
        ff_dim,
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(out1)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    ffn_output = tf.keras.layers.Dense(
        inputs.shape[-1],  # Match input dimension
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(ffn_output)
    
    # Add & Norm
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    return out2


def build_inception_transformer_model(
    n_conv_layers,
    n_filters,
    n_transformer_layers,
    num_heads,
    ff_dim,
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
    Build an Inception-CNN + Transformer model for sequence regression.

    Architecture:
      - Inception blocks: Multiple parallel Conv1D branches with different kernel sizes
      - Transformer encoder: Multi-head self-attention for sequential modeling
      - Dense head: Final regression layers

    Args:
        n_conv_layers (int): Number of inception blocks.
        n_filters (int): Total number of filters (divided across branches).
        n_transformer_layers (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads in transformer.
        ff_dim (int): Feed-forward network dimension in transformer.
        n_dense_layers (int): Number of dense/fully connected layers.
        n_dense_units (int): Number of units in each dense layer.
        learning_rate (float): Initial learning rate for Adam optimizer.
        l2_lambda (float): L2 regularization strength.
        dropout_rate (float): Dropout rate.
        skip_dropout_in_first_conv_layer (bool): If True, skip dropout in first conv.
        filter_sizes (list): List of filter sizes for inception branches.
        steps_per_epoch (int): Number of steps per epoch (for compatibility).

    Returns:
        tf.keras.Model: Compiled Keras Model ready for training.
    """
    INPUT_SHAPE = (180, 4)  # matches reference_model.config.INPUT_SHAPE
    input_seq = tf.keras.Input(shape=INPUT_SHAPE)
    x = input_seq

    # Inception CNN feature extraction
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

    # Transformer encoder
    # Add positional encoding (learned embeddings)
    seq_len = x.shape[1]  # Should be 180
    embed_dim = x.shape[2]  # Feature dimension from CNN
    
    # Create positional embeddings as a learnable layer
    # This creates embeddings for positions 0 to seq_len-1
    position_embedding_layer = tf.keras.layers.Embedding(
        input_dim=seq_len,
        output_dim=embed_dim,
        name="position_embedding"
    )
    
    # Create position indices: [0, 1, 2, ..., seq_len-1]
    # Use a Lambda layer to create this and expand for batch dimension
    def create_positions(input_tensor):
        seq_length = tf.shape(input_tensor)[1]
        positions = tf.range(seq_length, dtype=tf.int32)
        # Expand to [1, seq_len] so embedding gives [1, seq_len, embed_dim]
        # which will broadcast correctly to [batch, seq_len, embed_dim]
        return tf.expand_dims(positions, 0)
    
    positions = tf.keras.layers.Lambda(
        create_positions,
        name="position_indices"
    )(x)
    
    # Get positional embeddings: shape will be [1, seq_len, embed_dim]
    position_embeddings = position_embedding_layer(positions)
    
    # Add positional embeddings to CNN features
    # position_embeddings will broadcast from [1, seq_len, embed_dim] to [batch, seq_len, embed_dim]
    x = tf.keras.layers.Add(name="add_position_encoding")([x, position_embeddings])
    
    # Apply transformer encoder blocks
    for _ in range(n_transformer_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate, l2_lambda)
    
    # Global pooling (average over sequence length)
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
    """Parse hyperparameters from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Inception-Transformer model with hyperparameters"
    )
    
    # Model architecture parameters
    parser.add_argument("--n_conv_layers", type=int, help="Number of inception blocks")
    parser.add_argument("--n_filters", type=int, help="Number of filters per branch")
    parser.add_argument("--n_transformer_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, help="Feed-forward dimension")
    parser.add_argument("--n_dense_layers", type=int, help="Number of dense layers")
    parser.add_argument("--n_dense_units", type=int, help="Number of units per dense layer")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--l2_lambda", type=float, help="L2 regularization strength")
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--skip_dropout_in_first_conv_layer",
        type=lambda x: x.lower() == "true",
        help="Skip dropout in first conv layer (True/False)",
    )
    parser.add_argument(
        "--filter_sizes",
        type=str,
        help="Filter sizes as list string, e.g., '[3,5,7]'",
    )
    
    # Training control
    parser.add_argument(
        "--reduce_lr_factor",
        type=float,
        default=0.5,
        help="Factor for learning rate reduction (default: 0.5)",
    )
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=5,
        help="Patience for learning rate reduction (default: 5)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Patience for early stopping (default: 15)",
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
        help="Output directory for results",
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


def train_transformer_model(params_dict, output_dir=None, primary_param_name=None, verbose=1):
    """
    Train an Inception-Transformer model with given parameters.
    
    Args:
        params_dict (dict): Dictionary of hyperparameters.
        output_dir (str, optional): Output directory for results.
        verbose (int): Verbosity level (0, 1, or 2). Default 1.
    
    Returns:
        dict: Dictionary containing training results.
    """
    # Make a copy to avoid modifying the input
    params = params_dict.copy()
    
    # Extract parameters with defaults
    batch_size = params.pop("batch_size", 64)
    filter_sizes = params.pop("filter_sizes", [3, 5, 7, 9, 13])
    n_transformer_layers = params.pop("n_transformer_layers", 2)
    num_heads = params.pop("num_heads", 4)
    ff_dim = params.pop("ff_dim", 128)
    reduce_lr_factor = params.pop("reduce_lr_factor", 0.5)
    reduce_lr_patience = params.pop("reduce_lr_patience", 5)
    early_stopping_patience = params.pop("early_stopping_patience", 15)
    max_epochs = params.pop("max_epochs", 100)
    primary_param_name = params.pop("primary_param_name", primary_param_name)
    summary_csv_path = params.pop("summary_csv", None)
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
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    
    # Build model
    model = build_inception_transformer_model(
        steps_per_epoch=steps_per_epoch,
        filter_sizes=filter_sizes,
        n_transformer_layers=n_transformer_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        **params
    )
    
    if verbose >= 1:
        print("\nModel Architecture:")
        model.summary()
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "reference_transformer_model", "hyperparameter_tuning")
    os.makedirs(output_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1 if verbose >= 1 else 0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1 if verbose >= 1 else 0,
        ),
    ]
    
    # Train model
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_data=(X_dev, Y_dev),
        callbacks=callbacks,
        verbose=verbose,
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
    min_dev_loss = min(history.history["val_loss"])
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    
    # Calculate R²
    Y_pred = model.predict(X_test, verbose=0).flatten()
    test_r2 = r2_score(Y_test, Y_pred)
    
    if verbose >= 1:
        print(f"\nTest Loss (MSE): {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test R²: {test_r2:.6f}")
        print(f"Min Dev Loss: {min_dev_loss:.6f} (epoch {best_epoch})")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Dev Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Create unique filename suffix
    file_params = dict(params)
    file_params["batch_size"] = batch_size
    file_params["filter_sizes"] = filter_sizes
    file_params["n_transformer_layers"] = n_transformer_layers
    file_params["num_heads"] = num_heads
    file_params["ff_dim"] = ff_dim
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
    
    # Save results
    results_file = os.path.join(output_dir, f"results_{param_suffix}.txt")
    with open(results_file, "w") as f:
        f.write("Training Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Hyperparameters:\n")
        for k, v in sorted(file_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTransformer-specific:\n")
        f.write(f"  n_transformer_layers: {n_transformer_layers}\n")
        f.write(f"  num_heads: {num_heads}\n")
        f.write(f"  ff_dim: {ff_dim}\n")
        f.write("\nResults:\n")
        f.write(f"  Test Loss (MSE): {test_loss:.6f}\n")
        f.write(f"  Test MAE: {test_mae:.6f}\n")
        f.write(f"  Test R²: {test_r2:.6f}\n")
        f.write(f"  Min Dev Loss: {min_dev_loss:.6f}\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
    
    # Local summary (per-output_dir)
    local_summary_path = os.path.join(output_dir, "summary.csv")
    local_exists = os.path.exists(local_summary_path)
    summary_params = dict(file_params)
    summary_params["n_transformer_layers"] = n_transformer_layers
    summary_params["num_heads"] = num_heads
    summary_params["ff_dim"] = ff_dim
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
    
    return {
        "history": history,
        "model": model,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "min_dev_loss": min_dev_loss,
        "best_epoch": best_epoch,
        "plot_path": plot_path,
        "results_file": results_file,
    }


def main():
    """Main entry point for training."""
    params = parse_params_from_args()
    
    # Required parameters with defaults (for initial testing)
    base_params = {
        "n_conv_layers": 3,
        "n_filters": 64,
        "n_dense_layers": 2,
        "n_dense_units": 64,
        "learning_rate": 0.001,
        "l2_lambda": 0.1,
        "dropout_rate": 0.3,
        "skip_dropout_in_first_conv_layer": False,
        "filter_sizes": [3, 5, 7, 9, 13],
        "batch_size": 64,
        "n_transformer_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
    }
    
    # Merge user params over base params
    base_params.update(params)
    
    # Extract special params
    output_dir = base_params.pop("output_dir", None)
    primary_param_name = base_params.pop("primary_param_name", None)
    summary_csv = base_params.pop("summary_csv", None)
    
    # Train
    results = train_transformer_model(
        base_params,
        output_dir=output_dir,
        primary_param_name=primary_param_name,
        verbose=1
    )
    
    print(f"\nTraining complete!")
    print(f"Results saved to: {results['results_file']}")
    print(f"Plot saved to: {results['plot_path']}")


if __name__ == "__main__":
    main()

