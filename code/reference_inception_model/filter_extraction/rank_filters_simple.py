#!/usr/bin/env python
"""
Simplified ranking of first-layer filters.

For each first-layer filter, this script:
  1) Computes mean_activation over all test sequences and positions.
  2) Computes a consensus_sequence based on absolute kernel weights:
       - At each position, pick the base with the largest absolute weight.
  3) Computes p_consensus_sequence:
       - Sum of absolute weights along the consensus sequence
         divided by the sum of absolute weights over all bases/positions.
  4) Ranks filters by mean_activation and saves:
       rank, filter_size, filter_idx, mean_activation,
       consensus_sequence, p_consensus_sequence

Outputs:
  OUTPUT_DIR/reference_inception_model/filter_extraction/filter_ranking/filter_ranking_simple.csv
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from hyperparameter_tuning.config import OUTPUT_DIR, DATA_DIR, updated_params_3  # noqa: E402

# Paths
MODEL_PATH = os.path.join(
    OUTPUT_DIR, "reference_inception_model", "best_model", "best_model_reference_inception.h5"
)
OUTPUT_DIR_FILTERS = os.path.join(
    OUTPUT_DIR, "reference_inception_model", "filter_extraction", "extracted_filters"
)
OUTPUT_DIR_RANKING = os.path.join(
    OUTPUT_DIR, "reference_inception_model", "filter_extraction", "filter_ranking"
)
os.makedirs(OUTPUT_DIR_RANKING, exist_ok=True)

# Channel order is A, C, G, T based on one-hot encoding in reference_model.utils
BASES = ["A", "C", "G", "T"]


def get_first_layer_activation_model(
    model: tf.keras.Model,
) -> Tuple[tf.keras.Model, Dict[int, Tuple[int, int]]]:
    """
    Create a model that outputs first-layer concatenated activations and
    return index ranges per filter size.

    Returns:
        activation_model: Keras Model mapping input -> first concat output.
        index_ranges: dict mapping filter_size -> (start_idx, end_idx) in channel axis.
    """
    # Find the first Concatenate layer (output of first Inception block)
    first_concat = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Concatenate):
            first_concat = layer
            break

    if first_concat is None:
        raise ValueError("No Concatenate layer found in the model.")

    activation_model = tf.keras.Model(inputs=model.input, outputs=first_concat.output)

    # Derive channel index ranges from hyperparameters
    filter_sizes = updated_params_3["filter_sizes"]
    n_filters = updated_params_3["n_filters"]
    n_branches = len(filter_sizes)
    n_filters_per_branch = max(1, n_filters // n_branches)

    index_ranges: Dict[int, Tuple[int, int]] = {}
    start_idx = 0
    for size in filter_sizes:
        end_idx = start_idx + n_filters_per_branch
        index_ranges[size] = (start_idx, end_idx)
        start_idx = end_idx

    return activation_model, index_ranges


def compute_mean_activations_by_filter(
    model: tf.keras.Model, X_test: np.ndarray, batch_size: int = 128
) -> Dict[int, np.ndarray]:
    """
    Compute mean activation for each first-layer filter, grouped by filter size.

    Returns:
        dict mapping filter_size -> 1D array of shape (n_filters_per_branch,)
        with mean_activation for each filter.
    """
    activation_model, index_ranges = get_first_layer_activation_model(model)

    print("Computing first-layer activations for mean_activation...")
    activations = activation_model.predict(X_test, batch_size=batch_size, verbose=1)
    # activations shape: (n_samples, seq_len, total_filters)

    mean_activations_by_size: Dict[int, np.ndarray] = {}
    for size, (start_idx, end_idx) in index_ranges.items():
        act_block = activations[:, :, start_idx:end_idx]  # (n_samples, seq_len, n_filters)
        # Mean over samples and positions -> (n_filters,)
        mean_per_filter = act_block.mean(axis=(0, 1))
        mean_activations_by_size[size] = mean_per_filter

    return mean_activations_by_size


def load_filters_by_size() -> Dict[int, np.ndarray]:
    """
    Load first-layer filters from filters_layer1.npz.

    Returns:
      dict mapping filter_size -> filters array of shape (n_filters, kernel_size, 4)
    """
    filters_path = os.path.join(OUTPUT_DIR_FILTERS, "filters_layer1.npz")
    if not os.path.exists(filters_path):
        raise FileNotFoundError(f"Filters not found at {filters_path}")

    data = np.load(filters_path, allow_pickle=True)
    filter_sizes = data["filter_sizes"]

    filters_by_size: Dict[int, np.ndarray] = {}
    for size in filter_sizes:
        key = f"filters_size_{size}"
        if key not in data:
            raise KeyError(f"Filters for size {size} not found in {filters_path}")
        filters_by_size[int(size)] = data[key]

    return filters_by_size


def consensus_from_abs_weights(filter_weights: np.ndarray) -> Tuple[str, float]:
    """
    Compute consensus sequence and p_consensus_sequence from raw filter weights.

    Args:
        filter_weights: array of shape (kernel_size, 4) with raw weights.

    Returns:
        consensus_sequence (str)
        p_consensus_sequence (float): sum |w_consensus| / sum |w_all|
    """
    # Absolute weights across bases
    abs_weights = np.abs(filter_weights)  # (kernel_size, 4)

    # For each position, choose base with max absolute weight
    max_indices = abs_weights.argmax(axis=1)  # (kernel_size,)
    consensus_bases = [BASES[idx] for idx in max_indices]
    consensus_sequence = "".join(consensus_bases)

    # Total absolute weight along consensus vs all
    rows = np.arange(filter_weights.shape[0])
    abs_consensus = abs_weights[rows, max_indices].sum()
    abs_total = abs_weights.sum()

    if abs_total > 0:
        p_consensus = float(abs_consensus / abs_total)
    else:
        p_consensus = float("nan")

    return consensus_sequence, p_consensus


def main() -> None:
    print("=" * 70)
    print("Simplified Ranking of First-Layer Filters")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded")

    # Load test data
    print(f"\nLoading test data from: {DATA_DIR}")
    data = np.load(os.path.join(DATA_DIR, "preprocessed_data.npz"))
    X_test = data["X_test"]
    print(f"✓ Loaded {len(X_test)} test sequences")

    # Compute mean activations per filter
    mean_activations_by_size = compute_mean_activations_by_filter(model, X_test)

    # Load filters to compute consensus statistics
    filters_by_size = load_filters_by_size()

    # Assemble rows
    rows = []
    for filter_size, mean_acts in mean_activations_by_size.items():
        if filter_size not in filters_by_size:
            raise KeyError(
                f"Filter weights for size {filter_size} not found in filters_layer1.npz"
            )
        filters = filters_by_size[filter_size]  # (n_filters, kernel_size, 4)
        n_filters = filters.shape[0]

        if n_filters != len(mean_acts):
            raise ValueError(
                f"Mismatch for size {filter_size}: {n_filters} filters in weights "
                f"but {len(mean_acts)} mean activations"
            )

        for filter_idx in range(n_filters):
            w = filters[filter_idx]  # (kernel_size, 4)
            consensus_seq, p_consensus = consensus_from_abs_weights(w)
            rows.append(
                {
                    "filter_size": int(filter_size),
                    "filter_idx": int(filter_idx),
                    "mean_activation": float(mean_acts[filter_idx]),
                    "consensus_sequence": consensus_seq,
                    "p_consensus_sequence": float(p_consensus),
                }
            )

    df = pd.DataFrame(rows)

    # Rank by mean_activation (descending)
    df = df.sort_values("mean_activation", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # Reorder columns
    df = df[
        [
            "rank",
            "filter_size",
            "filter_idx",
            "mean_activation",
            "consensus_sequence",
            "p_consensus_sequence",
        ]
    ]

    # Save
    output_path = os.path.join(OUTPUT_DIR_RANKING, "filter_ranking_simple.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Simplified ranking saved to: {output_path}")

    # Show top 10
    print("\nTop 10 filters by mean_activation:")
    print(df.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("Simplified Ranking Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()


