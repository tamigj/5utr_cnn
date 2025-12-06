#!/usr/bin/env python
"""
Simplified ranking of first-layer filters for Siamese model.

For each first-layer filter, this script:
  1) Computes mean_activation over all test sequences and positions.
     - For REF branch: uses REF sequences
     - For ALT branch: uses ALT sequences
  2) Computes a consensus_sequence based on absolute kernel weights:
       - At each position, pick the base with the largest absolute weight.
  3) Computes p_consensus_sequence:
       - Sum of absolute weights along the consensus sequence
         divided by the sum of absolute weights over all bases/positions.
  4) Ranks filters by mean_activation and saves:
       rank, filter_size, filter_idx, mean_activation,
       consensus_sequence, p_consensus_sequence

Outputs:
  OUTPUT_DIR/variation_inception_model/filter_extraction/siamese_ref/filter_ranking/filter_ranking_simple.csv
  OUTPUT_DIR/variation_inception_model/filter_extraction/siamese_alt/filter_ranking/filter_ranking_simple.csv
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

# Import config from reference model to get model parameters
sys.path.insert(0, os.path.join(PARENT_DIR, '..', 'reference_inception_model', 'hyperparameter_tuning'))
from config import OUTPUT_DIR, updated_params_3, BASE_DIR

# Paths
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, 'variation_inception_model', 'richer_model', 
                                  'unfrozen_0.1_0.1_0.8_20p', 'model.h5')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'naptrap_full_data.tsv')

OUTPUT_DIR_FILTERS_REF = os.path.join(
    OUTPUT_DIR, "variation_inception_model", "filter_extraction", "siamese_ref", "extracted_filters"
)
OUTPUT_DIR_FILTERS_ALT = os.path.join(
    OUTPUT_DIR, "variation_inception_model", "filter_extraction", "siamese_alt", "extracted_filters"
)
OUTPUT_DIR_RANKING_REF = os.path.join(
    OUTPUT_DIR, "variation_inception_model", "filter_extraction", "siamese_ref", "filter_ranking"
)
OUTPUT_DIR_RANKING_ALT = os.path.join(
    OUTPUT_DIR, "variation_inception_model", "filter_extraction", "siamese_alt", "filter_ranking"
)
os.makedirs(OUTPUT_DIR_RANKING_REF, exist_ok=True)
os.makedirs(OUTPUT_DIR_RANKING_ALT, exist_ok=True)

# Channel order is A, C, G, T based on one-hot encoding
BASES = ["A", "C", "G", "T"]


def extract_encoder_from_siamese(siamese_model):
    """Extract the encoder from the Siamese model."""
    # Method 1: Look for Model layers
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                if layer.input_shape == (None, 180, 4) or layer.input_shape[1:] == (180, 4):
                    return layer
            except:
                pass
    
    # Method 2: Extract by tracing from seq_ref input to emb_ref output
    concat_layer = None
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.layers.Concatenate):
            concat_layer = layer
            break
    
    if concat_layer is None:
        raise ValueError("Could not find Concatenate layer in Siamese model")
    
    concat_inputs = concat_layer.input
    if isinstance(concat_inputs, list) and len(concat_inputs) >= 2:
        emb_ref_output = concat_inputs[0]
        ref_input = siamese_model.input[0]
        encoder_model = tf.keras.Model(inputs=ref_input, outputs=emb_ref_output)
        return encoder_model
    
    raise ValueError("Could not extract encoder from Siamese model")


def get_first_layer_activation_model(
    encoder_model: tf.keras.Model,
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
    for layer in encoder_model.layers:
        if isinstance(layer, tf.keras.layers.Concatenate):
            first_concat = layer
            break

    if first_concat is None:
        raise ValueError("No Concatenate layer found in the encoder model.")

    activation_model = tf.keras.Model(inputs=encoder_model.input, outputs=first_concat.output)

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
    encoder_model: tf.keras.Model, 
    X_test: np.ndarray, 
    batch_size: int = 128
) -> Dict[int, np.ndarray]:
    """
    Compute mean activation for each first-layer filter, grouped by filter size.

    Returns:
        dict mapping filter_size -> 1D array of shape (n_filters_per_branch,)
        with mean_activation for each filter.
    """
    activation_model, index_ranges = get_first_layer_activation_model(encoder_model)

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


def load_filters_by_size(output_dir: str) -> Dict[int, np.ndarray]:
    """
    Load first-layer filters from filters_layer1.npz.

    Returns:
      dict mapping filter_size -> filters array of shape (n_filters, kernel_size, 4)
    """
    filters_path = os.path.join(output_dir, "filters_layer1.npz")
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


def load_test_sequences(data_path: str, random_state: int = 230):
    """Load REF and ALT test sequences."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    df = df.dropna(subset=['REF_sequence', 'ALT_sequence', 'pw_mean_translation', 'pw_mean_log2_delta_t'])
    
    df['REF_sequence'] = df['REF_sequence'].astype(str)
    df['ALT_sequence'] = df['ALT_sequence'].astype(str)
    
    # One-hot encoding function
    def pad_sequence(seq, target_len=180):
        seq = seq.upper()
        if len(seq) < target_len:
            seq = "N" * (target_len - len(seq)) + seq
        return seq[:target_len]
    
    def one_hot_encode(seq, target_len=180):
        seq = pad_sequence(seq, target_len)
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        arr = np.zeros((target_len, 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in mapping:
                arr[i, mapping[base]] = 1.0
        return arr
    
    # Encode sequences
    print("Encoding sequences...")
    seq_ref_array = np.stack(df['REF_sequence'].apply(one_hot_encode).values)
    seq_alt_array = np.stack(df['ALT_sequence'].apply(one_hot_encode).values)
    
    # Split: 70% train, 15% dev, 15% test (matching training script)
    X_ref_train, X_ref_tmp, X_alt_train, X_alt_tmp = train_test_split(
        seq_ref_array, seq_alt_array, test_size=0.3, random_state=random_state, shuffle=True
    )
    X_ref_dev, X_ref_test, X_alt_dev, X_alt_test = train_test_split(
        X_ref_tmp, X_alt_tmp, test_size=0.5, random_state=random_state
    )
    
    print(f"Test sequences: {len(X_ref_test)} REF, {len(X_alt_test)} ALT")
    
    return X_ref_test, X_alt_test


def rank_filters_for_branch(
    branch_name: str,
    encoder_model: tf.keras.Model,
    X_test: np.ndarray,
    filters_by_size: Dict[int, np.ndarray],
    output_dir_ranking: str
):
    """Rank filters for a specific branch (REF or ALT)."""
    print(f"\n{'='*70}")
    print(f"Ranking filters for {branch_name} branch")
    print(f"{'='*70}")
    
    # Compute mean activations per filter
    mean_activations_by_size = compute_mean_activations_by_filter(encoder_model, X_test)
    
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
    output_path = os.path.join(output_dir_ranking, "filter_ranking_simple.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Ranking saved to: {output_path}")

    # Show top 10
    print(f"\nTop 10 filters by mean_activation ({branch_name} branch):")
    print(df.head(10).to_string(index=False))


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description='Rank filters from Siamese model')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to trained Siamese model')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='Path to data TSV file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Simplified Ranking of First-Layer Filters (Siamese Model)")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    siamese_model = tf.keras.models.load_model(args.model_path, compile=False)
    print("✓ Siamese model loaded")
    
    # Extract encoder
    print("\nExtracting encoder from Siamese model...")
    encoder_model = extract_encoder_from_siamese(siamese_model)
    print("✓ Encoder extracted")

    # Load test sequences
    X_ref_test, X_alt_test = load_test_sequences(args.data_path)

    # Load filters
    filters_by_size_ref = load_filters_by_size(OUTPUT_DIR_FILTERS_REF)
    filters_by_size_alt = load_filters_by_size(OUTPUT_DIR_FILTERS_ALT)

    # Rank filters for REF branch
    rank_filters_for_branch(
        "REF",
        encoder_model,
        X_ref_test,
        filters_by_size_ref,
        OUTPUT_DIR_RANKING_REF
    )

    # Rank filters for ALT branch
    rank_filters_for_branch(
        "ALT",
        encoder_model,
        X_alt_test,
        filters_by_size_alt,
        OUTPUT_DIR_RANKING_ALT
    )

    print("\n" + "=" * 70)
    print("Simplified Ranking Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

