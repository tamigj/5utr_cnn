#!/usr/bin/env python

"""
Run multiple retraining replicates for length-stratified REF subsets.

Conditions:
  - T = 100: train on 5,000 sequences with seq_len < 100, test on 5,000 with seq_len > 100
  - T = 125: train on 5,000 sequences with seq_len < 125, test on 5,000 with seq_len > 125
  - T = 150: train on 5,000 sequences with seq_len < 150, test on 5,000 with seq_len > 150
  - T = any: train on 5,000 random sequences (any length), test on 5,000 other random sequences

For each condition, this script:
  - Runs 30 independent training runs (different random samples).
  - Records the test R² for each run.
  - Saves a single table with columns: T, rep, R2.
  - Creates a barplot of mean R² per condition with individual replicate R² values overlaid as points.

Outputs:
  OUTPUT_DIR/reference_inception_model/retrain_on_short_seq/experiment/
    - r2_replicates_by_T.csv
    - r2_replicates_by_T_barplot.png
"""

import os
import sys
import time
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
import tensorflow as tf  # noqa: E402

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
REF_MODEL_DIR = os.path.join(PROJECT_ROOT, "reference_model")

sys.path.insert(0, PROJECT_ROOT)   # for reference_inception_model.*
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


EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "reference_inception_model", "retrain_on_short_seq", "experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def make_length_split(
    df: pd.DataFrame, threshold: int, n_train: int = 5000, n_test: int = 5000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test DataFrames based on seq_len threshold."""
    df_short = df[df["seq_len"] < threshold]
    df_long = df[df["seq_len"] > threshold]

    if len(df_short) < n_train:
        train_df = df_short.sample(n=len(df_short), random_state=seed)
    else:
        train_df = df_short.sample(n=n_train, random_state=seed)

    if len(df_long) < n_test:
        test_df = df_long.sample(n=len(df_long), random_state=seed + 1)
    else:
        test_df = df_long.sample(n=n_test, random_state=seed + 1)

    return train_df, test_df


def make_any_length_split(
    df: pd.DataFrame, n_train: int = 5000, n_test: int = 5000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test DataFrames by random sampling, ignoring seq_len.

    Ensures train and test sets are disjoint.
    """
    if len(df) <= n_train:
        train_df = df.sample(n=len(df), random_state=seed)
        test_df = df.iloc[0:0].copy()
        return train_df, test_df

    train_df = df.sample(n=min(n_train, len(df)), random_state=seed)
    remaining = df.drop(train_df.index)

    if len(remaining) <= n_test:
        test_df = remaining.sample(n=len(remaining), random_state=seed + 1)
    else:
        test_df = remaining.sample(n=n_test, random_state=seed + 1)

    return train_df, test_df


def prepare_xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot encode REF_sequence and extract pw_mean_translation."""
    seqs = df["REF_sequence"].astype(str).tolist()
    X = np.stack([one_hot_encode(s) for s in seqs])
    y = df["pw_mean_translation"].values.astype(np.float32)
    return X, y


def train_once_for_condition(
    df: pd.DataFrame, condition: str, threshold: int | None, rep_seed: int
) -> float:
    """
    Run a single training/evaluation for a given condition.

    Args:
        df: full ref_data dataframe with REF_sequence, pw_mean_translation, seq_len.
        condition: '100', '125', '150', or 'any'.
        threshold: numeric threshold for length-based splits (ignored for 'any').
        rep_seed: base seed for sampling / reproducibility.

    Returns:
        Test R² for this run.
    """
    if condition == "any":
        train_df, test_df = make_any_length_split(df, seed=rep_seed)
    else:
        assert threshold is not None
        train_df, test_df = make_length_split(df, threshold=threshold, seed=rep_seed)

    # Prepare X/y
    X_train_full, y_train_full = prepare_xy_from_df(train_df)
    X_test, y_test = prepare_xy_from_df(test_df)

    # Train/dev split within training data for early stopping
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=rep_seed, shuffle=True
    )

    # Hyperparameters: best from updated_params_3, with skip_dropout_in_first_conv_layer=False
    params = updated_params_3.copy()
    params["skip_dropout_in_first_conv_layer"] = False

    batch_size = params["batch_size"]
    filter_sizes = params["filter_sizes"]
    steps_per_epoch = max(1, len(X_train) // batch_size)

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

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=0,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_dev, y_dev),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    # Evaluate on test set
    if len(X_test) == 0:
        return float("nan")

    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    return float(r2)


def main() -> None:
    print("=" * 80)
    print("Running 30 retraining replicates for each length condition")
    print("=" * 80)

    # Load full reference data once
    df = pd.read_csv(os.path.join(DATA_DIR, "ref_data.tsv"), sep="\t")
    df["REF_sequence"] = df["REF_sequence"].astype(str)
    if "seq_len" not in df.columns:
        raise ValueError("Column 'seq_len' not found in ref_data.tsv.")
    if "pw_mean_translation" not in df.columns:
        raise ValueError("Column 'pw_mean_translation' not found in ref_data.tsv.")

    conditions: List[Dict[str, object]] = [
        {"T": "100", "threshold": 100},
        {"T": "125", "threshold": 125},
        {"T": "150", "threshold": 150},
        {"T": "any", "threshold": None},
    ]

    n_reps = 30
    rows: List[Dict[str, object]] = []

    start_all = time.time()

    for cond in conditions:
        T_label = cond["T"]
        threshold = cond["threshold"]
        print("\n" + "-" * 80)
        print(f"Condition T={T_label}: running {n_reps} replicates")
        print("-" * 80)

        for rep in range(1, n_reps + 1):
            rep_seed = 42 + rep  # simple varying seed
            print(f"T={T_label}, replicate {rep}/{n_reps} (seed={rep_seed})")
            start_rep = time.time()
            r2 = train_once_for_condition(
                df=df,
                condition=str(T_label),
                threshold=threshold if isinstance(threshold, int) else None,
                rep_seed=rep_seed,
            )
            dur = time.time() - start_rep
            print(f"  -> R² = {r2:.6f} (took {dur:.1f}s)")
            rows.append({"T": str(T_label), "rep": rep, "R2": r2})

    # Save table
    df_res = pd.DataFrame(rows)
    table_path = os.path.join(EXPERIMENT_DIR, "r2_replicates_by_T.csv")
    df_res.to_csv(table_path, index=False)
    print(f"\nSaved R² replicate table to {table_path}")

    # Plot barplot with points
    # Order T on x-axis
    T_order = ["100", "125", "150", "any"]
    means = df_res.groupby("T")["R2"].mean().reindex(T_order)

    x = np.arange(len(T_order))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, means.values, width=bar_width, color="lightgray", edgecolor="black", label="Mean R²")

    rng = np.random.default_rng(seed=123)
    for i, T_label in enumerate(T_order):
        sub = df_res[df_res["T"] == T_label]
        if sub.empty:
            continue
        jitter = (rng.random(len(sub)) - 0.5) * (bar_width * 0.6)
        ax.scatter(
            i + jitter,
            sub["R2"].values,
            color="black",
            alpha=0.8,
            s=20,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(T_order)
    ax.set_xlabel("Condition (T)", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_title("Length-Based Retraining: R² over 30 Replicates per Condition", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(EXPERIMENT_DIR, "r2_replicates_by_T_barplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved barplot to {plot_path}")

    total_dur = time.time() - start_all
    print(f"\nTotal experiment time: {total_dur/60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()



