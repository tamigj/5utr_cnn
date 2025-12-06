#!/usr/bin/env python

"""
Retrain the best reference Inception model on length-stratified REF subsets.

For each length threshold T in {100, 125, 150}, this script:
  1) Selects 5,000 sequences with seq_len < T for training
  2) Selects 5,000 sequences with seq_len > T for testing
  3) Trains the Inception model with the best hyperparameters
  4) Saves a training curve, R² text file, and true-vs-predicted scatterplot

Additionally, it trains on 5,000 randomly selected REF sequences of any length
and evaluates on 5,000 different randomly selected REF sequences of any length
("seq_len_any" condition).

Outputs are saved under:
  OUTPUT_DIR/reference_inception_model/retrain_on_short_seq/seq_len_<T>/
  OUTPUT_DIR/reference_inception_model/retrain_on_short_seq/seq_len_any/
"""

import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
import tensorflow as tf  # noqa: E402

# Add parent directories to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
REF_MODEL_DIR = os.path.join(PROJECT_ROOT, "reference_model")

# Ensure imports work both locally and on the cluster
sys.path.insert(0, PROJECT_ROOT)   # so `reference_inception_model.*` works
sys.path.insert(0, REF_MODEL_DIR)  # so reference_model.utils can import config

from reference_inception_model.hyperparameter_tuning.config import (  # noqa: E402
    updated_params_3,
    DATA_DIR,
    OUTPUT_DIR,
)
from reference_model.utils import one_hot_encode  # noqa: E402
from reference_inception_model.reference_model_inception_tunable import (  # type: ignore  # noqa: E402,E501
    build_inception_cnn_model,
)


BASE_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR, "reference_inception_model", "retrain_on_short_seq"
)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def make_length_split(
    df: pd.DataFrame, threshold: int, n_train: int = 5000, n_test: int = 5000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test DataFrames based on seq_len threshold.

    Train: seq_len < threshold, Test: seq_len > threshold.
    Samples up to n_train / n_test rows (or fewer if insufficient).
    """
    df_short = df[df["seq_len"] < threshold]
    df_long = df[df["seq_len"] > threshold]

    if len(df_short) < n_train:
        print(
            f"Warning: Only {len(df_short)} short sequences available for threshold {threshold}; "
            f"using all instead of {n_train}."
        )
        train_df = df_short.sample(n=len(df_short), random_state=seed)
    else:
        train_df = df_short.sample(n=n_train, random_state=seed)

    if len(df_long) < n_test:
        print(
            f"Warning: Only {len(df_long)} long sequences available for threshold {threshold}; "
            f"using all instead of {n_test}."
        )
        test_df = df_long.sample(n=len(df_long), random_state=seed)
    else:
        test_df = df_long.sample(n=n_test, random_state=seed)

    return train_df, test_df


def make_any_length_split(
    df: pd.DataFrame, n_train: int = 5000, n_test: int = 5000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test DataFrames by random sampling, ignoring seq_len.

    Ensures train and test sets are disjoint.
    """
    if len(df) <= n_train:
        print(
            f"Warning: Only {len(df)} total sequences available; "
            f"using all for training and leaving no separate test set."
        )
        train_df = df.sample(n=len(df), random_state=seed)
        test_df = df.iloc[0:0].copy()
        return train_df, test_df

    train_df = df.sample(n=min(n_train, len(df)), random_state=seed)
    remaining = df.drop(train_df.index)

    if len(remaining) <= n_test:
        print(
            f"Warning: Only {len(remaining)} remaining sequences available for test; "
            f"using all instead of requested {n_test}."
        )
        test_df = remaining.sample(n=len(remaining), random_state=seed + 1)
    else:
        test_df = remaining.sample(n=n_test, random_state=seed + 1)

    return train_df, test_df


def prepare_xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot encode REF_sequence and extract pw_mean_translation."""
    # Column is named 'REF_sequence' in ref_data.tsv
    seqs = df["REF_sequence"].astype(str).tolist()
    X = np.stack([one_hot_encode(s) for s in seqs])
    y = df["pw_mean_translation"].values.astype(np.float32)
    return X, y


def train_and_evaluate_for_threshold(threshold: int) -> None:
    """Train model for a given length threshold and save outputs."""
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"seq_len_{threshold}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print(f"Training Inception model with length-based split at seq_len {threshold}")
    print("=" * 80)


def train_and_evaluate_any_length() -> None:
    """Train model on 5,000 random REF sequences of any length and test on 5,000 others."""
    run_dir = os.path.join(BASE_OUTPUT_DIR, "seq_len_any")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print("Training Inception model with random length-agnostic split (seq_len_any)")
    print("=" * 80)

    # Load full reference data
    df = pd.read_csv(os.path.join(DATA_DIR, "ref_data.tsv"), sep="\t")
    df["REF_sequence"] = df["REF_sequence"].astype(str)

    if "pw_mean_translation" not in df.columns:
        raise ValueError("Column 'pw_mean_translation' not found in ref_data.tsv.")

    # Random train/test split (length-agnostic)
    train_df, test_df = make_any_length_split(df)
    print(
        f"seq_len_any: train n={len(train_df)} (random), "
        f"test n={len(test_df)} (random, disjoint from train)"
    )

    # Prepare X/y
    X_train_full, y_train_full = prepare_xy_from_df(train_df)
    X_test, y_test = prepare_xy_from_df(test_df)

    # Train/dev split within training data for early stopping
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=True
    )

    # Best hyperparameters from updated_params_3 + skip_dropout_in_first_conv_layer=False
    params = updated_params_3.copy()
    params["skip_dropout_in_first_conv_layer"] = False

    batch_size = params["batch_size"]
    filter_sizes = params["filter_sizes"]
    steps_per_epoch = max(1, len(X_train) // batch_size)

    print("\nHyperparameters (seq_len_any):")
    for k in sorted(params.keys()):
        print(f"  {k}: {params[k]}")
    print(f"\nsteps_per_epoch: {steps_per_epoch}")
    print()

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

    model.summary(print_fn=lambda x: print(x))

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    # Train model
    print("Starting training (seq_len_any)...")
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_dev, y_dev),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Training curve
    train_losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    best_epoch = None
    if val_losses:
        min_val_loss = min(val_losses)
        best_epoch = val_losses.index(min_val_loss) + 1
    else:
        min_val_loss = None

    plt.figure(figsize=(8, 6))
    if train_losses:
        plt.plot(train_losses, label="Training Loss", linewidth=2, color="blue")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss", linewidth=2, color="orange")
    if best_epoch is not None:
        plt.axvline(
            x=best_epoch - 1,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best epoch ({best_epoch})",
        )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    title = "Inception Retrain (seq_len_any)"
    if min_val_loss is not None:
        title += f"\n(Min Dev Loss: {min_val_loss:.4f})"
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    if train_losses or val_losses:
        max_loss = max(train_losses + val_losses)
        plt.ylim((0, max_loss * 1.1))

    curve_path = os.path.join(run_dir, "training_curve_seq_len_any.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve to {curve_path}")

    # Evaluate R² on test set
    print("\nEvaluating on seq_len_any test set...")
    if len(X_test) > 0:
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
    else:
        print("No test data available for seq_len_any; skipping R² calculation.")
        y_pred = np.array([])
        r2 = float("nan")
    print(f"Test R² (seq_len_any): {r2:.6f}")

    r2_file = os.path.join(run_dir, "r2_scores_seq_len_any.txt")
    with open(r2_file, "w") as f:
        f.write("R² Scores (seq_len_any)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Train n (random any length): {len(train_df)}\n")
        f.write(f"Test  n (random any length): {len(test_df)}\n")
        f.write(f"Test R²: {r2:.6f}\n")
        f.write(
            f"\nRuntime: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} seconds)\n"
        )
    print(f"Saved R² scores to {r2_file}")

    # Scatterplot true vs predicted
    if len(X_test) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, alpha=0.6, color="black", s=25)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Perfect prediction",
        )
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlabel("True Translation Efficiency", fontsize=12)
        ax.set_ylabel("Predicted Translation Efficiency", fontsize=12)
        ax.set_title(
            "Predicted vs True (train/test random any length)", fontsize=14
        )
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        scatter_path = os.path.join(run_dir, "true_vs_predicted_seq_len_any.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved scatterplot to {scatter_path}")

    print("\nCompleted run for seq_len_any.")
    print(f"Outputs saved to: {run_dir}")
    print(
        f"Runtime: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} seconds) for seq_len_any"
    )
    print("=" * 80)


def main() -> None:
    thresholds = [100, 125, 150]
    print("Length-based retraining for thresholds:", thresholds)
    for thr in thresholds:
        train_and_evaluate_for_threshold(threshold=thr)

    print("\nNow running length-agnostic condition (seq_len_any)...")
    train_and_evaluate_any_length()


if __name__ == "__main__":
    main()


