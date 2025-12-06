#!/usr/bin/env python
"""5UTR Step 2 Model - Richer Architecture (Tunable)

Siamese network with:
- Unfrozen encoder (fine-tuned) from inception model
- Richer feature representation (ref, mut, diff, product)
- Wider/deeper dense head
- Tunable hyperparameters via command-line arguments

Usage:
    python draft_training_richer_tunable.py [--learning_rate 1e-4] [--batch_size 128] ...

    Or pass all parameters as key=value pairs:
    python draft_training_richer_tunable.py learning_rate=1e-4 batch_size=128 ...
"""

import os
import sys
import ast
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Subtract, Multiply, Concatenate, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# =============================================================================
# Default hyperparameters
# =============================================================================

DEFAULT_PARAMS = {
    'learning_rate': 1e-4,
    'batch_size': 128,
    'dense_units_1': 128,
    'dense_units_2': 64,
    'dense_units_3': 32,
    'l2_lambda': 0.001,
    'dropout_rate': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6,
    'max_epochs': 100,
    'encoder_trainable': True,
    'squared': False,
    'data_fraction': 1.0,
    'random_state': 230,
    'loss_weight_ref': 1.0,
    'loss_weight_alt': 2.0,
    'loss_weight_delta': 4.0,
}

# =============================================================================
# Paths
# =============================================================================

BASE_DIR = '/mnt/oak/users/tami/5utr_cnn'
DATA_PATH = f'{BASE_DIR}/data/naptrap_full_data.tsv'
ENCODER_MODEL_PATH = f'{BASE_DIR}/output/reference_inception_model/best_model/best_model_reference_inception.h5'
OUTPUT_DIR = f'{BASE_DIR}/output/variation_inception_model/richer_model'

# =============================================================================
# Parse command-line arguments
# =============================================================================

def parse_params_from_args():
    """Parse hyperparameters from command-line arguments.

    Supports two formats:
    1. --key value: --learning_rate 1e-4
    2. key=value pairs: learning_rate=1e-4 batch_size=128
    """
    parser = argparse.ArgumentParser(
        description='Train tunable richer siamese model with specified hyperparameters'
    )

    # Add all hyperparameters as optional arguments
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--dense_units_1', type=int, help='Units in first dense layer')
    parser.add_argument('--dense_units_2', type=int, help='Units in second dense layer')
    parser.add_argument('--dense_units_3', type=int, help='Units in third dense layer')
    parser.add_argument('--l2_lambda', type=float, help='L2 regularization strength')
    parser.add_argument('--dropout_rate', type=float, help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, help='Early stopping patience')
    parser.add_argument('--reduce_lr_patience', type=int, help='ReduceLROnPlateau patience')
    parser.add_argument('--reduce_lr_factor', type=float, help='ReduceLROnPlateau factor')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--encoder_trainable', type=lambda x: x.lower() == 'true',
                       help='Whether encoder is trainable (True/False)')
    parser.add_argument('--squared', action='store_true',
                       help='If set, predict squared delta TE (magnitude only, direction-agnostic)')
    parser.add_argument('--data_fraction', type=float,
                       help='Fraction of data to use (0.0 to 1.0, default: 1.0 for 100%%)')
    parser.add_argument('--random_state', type=int, help='Random state for data splitting')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--loss_weight_ref', type=float, help='Loss weight for REF translation prediction')
    parser.add_argument('--loss_weight_alt', type=float, help='Loss weight for ALT translation prediction')
    parser.add_argument('--loss_weight_delta', type=float, help='Loss weight for variant effect (ΔTE) prediction')

    # Also support key=value format for backward compatibility
    args, unknown = parser.parse_known_args()

    # Parse any key=value pairs from unknown args
    params = {}
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
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


def find_encoder_output_layer(model):
    """Find the encoder output layer (before final Dense(1) output).

    The encoder should be the output before the final prediction layer.
    For inception models, this extracts everything up to (but not including) the final Dense(1).
    """
    # Find the last Dense layer with units=1 (output layer)
    last_dense_idx = None
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1:
            last_dense_idx = i
            break

    if last_dense_idx is not None and last_dense_idx > 0:
        # Return the layer before the final Dense(1)
        # This should be the last dropout/batchnorm/activation before the output
        return model.layers[last_dense_idx - 1].output

    # Fallback: use second-to-last layer
    if len(model.layers) >= 2:
        return model.layers[-2].output

    # Last resort: use the model output (shouldn't happen)
    return model.output


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_and_preprocess_data(data_path, random_state=230, squared=False, data_fraction=1.0):
    """Load and preprocess mutation data."""
    print("Loading data...")
    df = pd.read_csv(data_path, sep="\t")
    df = df.dropna(subset=['gene', 'REF_sequence', 'ALT_sequence', 
                            'REF_pw_mean_translation', 'ALT_pw_mean_translation',
                            'pw_mean_log2_delta_t'])

    # Sample fraction of data if specified
    if data_fraction < 1.0:
        n_samples = int(len(df) * data_fraction)
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        print(f"Using {data_fraction*100:.1f}% of data ({len(df)} samples)")

    df['gene'] = df['gene'].astype(str)
    df['REF_sequence'] = df['REF_sequence'].astype(str)
    df['ALT_sequence'] = df['ALT_sequence'].astype(str)
    df['REF_pw_mean_translation'] = df['REF_pw_mean_translation'].values.astype(np.float32)
    df['ALT_pw_mean_translation'] = df['ALT_pw_mean_translation'].values.astype(np.float32)
    df['variant_effect'] = df['pw_mean_log2_delta_t'].values.astype(np.float32)

    # Padding + one-hot encoding
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

    # Handle squared target
    if squared:
        df['modified_variant_effect'] = (df['variant_effect'] ** 2).astype(np.float32)
        print(f"Using SQUARED target (magnitude only, direction-agnostic)")
        print(f"Original delta TE range: [{df['variant_effect'].min():.4f}, {df['variant_effect'].max():.4f}]")
        print(f"Squared delta TE range: [{df['modified_variant_effect'].min():.4f}, {df['modified_variant_effect'].max():.4f}]")
    else:
        df['modified_variant_effect'] = df['variant_effect'].values.astype(np.float32)
        print(f"Delta TE range: [{df['modified_variant_effect'].min():.4f}, {df['modified_variant_effect'].max():.4f}]")

    # # Split: 70% train, 15% dev, 15% test (50% test from unseen genes, 50% test from seen genes)
    np.random.seed(random_state)

    # Determine total test size
    n_total = len(df)
    n_test = int(0.15 * n_total)
    n_unseen_test = n_test // 2     # Numbers of samples for "unseen genes"
    n_seen_test = n_test - n_unseen_test # Numbers of samples for "unseen variants"

    # Select unseen genes until reaching n_unseen_test samples
    genes = df['gene'].unique()
    np.random.shuffle(genes)

    unseen_genes = []
    unseen_count = 0
    for g in genes:
        gene_rows = df[df['gene'] == g]
        # If adding this gene would exceed target, only add if we haven't reached target yet
        if unseen_count + len(gene_rows) <= n_unseen_test:
            unseen_genes.append(g)
            unseen_count += len(gene_rows)
        elif unseen_count < n_unseen_test:
            # If we haven't reached target and this gene would exceed it, add it anyway to get closer
            unseen_genes.append(g)
            unseen_count += len(gene_rows)
        if unseen_count >= n_unseen_test:
            break

    df_unseen_test = df[df['gene'].isin(unseen_genes)]
    df_remaining = df[~df['gene'].isin(unseen_genes)]

    # Split remaining rows into train/dev/seen-test
    df_train_dev, df_seen_test = train_test_split(df_remaining, test_size=n_seen_test, random_state=random_state, shuffle=True)
    train_size = 0.7 / (0.7 + 0.15)  # to get 70/15 split from train_dev
    df_train, df_dev = train_test_split(df_train_dev, train_size=train_size, random_state=random_state, shuffle=True)

    # Final test set
    df_test = pd.concat([df_unseen_test, df_seen_test]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)   

    # Reference sequences
    X_ref_train = np.stack(df_train['REF_sequence'].apply(one_hot_encode).values)
    X_ref_dev = np.stack(df_dev['REF_sequence'].apply(one_hot_encode).values)
    X_ref_test = np.stack(df_test['REF_sequence'].apply(one_hot_encode).values)

    # Mutant sequences
    X_mut_train = np.stack(df_train['ALT_sequence'].apply(one_hot_encode).values)
    X_mut_dev = np.stack(df_dev['ALT_sequence'].apply(one_hot_encode).values)
    X_mut_test = np.stack(df_test['ALT_sequence'].apply(one_hot_encode).values)

    # REF translation
    yREF_train = df_train['REF_pw_mean_translation'].values.astype(np.float32).reshape(-1, 1)
    yREF_dev = df_dev['REF_pw_mean_translation'].values.astype(np.float32).reshape(-1, 1)
    yREF_test = df_test['REF_pw_mean_translation'].values.astype(np.float32).reshape(-1, 1)

    # ALT translation
    yALT_train = df_train['ALT_pw_mean_translation'].values.astype(np.float32).reshape(-1, 1)
    yALT_dev = df_dev['ALT_pw_mean_translation'].values.astype(np.float32).reshape(-1, 1)
    yALT_test = df_test['ALT_pw_mean_translation'].values.astype(np.float32).reshape(-1, 1)

    # ΔTE
    yVAR_train = df_train['modified_variant_effect'].values.astype(np.float32).reshape(-1, 1)
    yVAR_dev = df_dev['modified_variant_effect'].values.astype(np.float32).reshape(-1, 1)
    yVAR_test = df_test['modified_variant_effect'].values.astype(np.float32).reshape(-1, 1)

    print(f"Train: {len(yVAR_train)}, Dev: {len(yVAR_dev)}, Test: {len(yVAR_test)}")

    return (X_ref_train, X_mut_train, yREF_train, yALT_train, yVAR_train,
            X_ref_dev, X_mut_dev, yREF_dev, yALT_dev, yVAR_dev,
            X_ref_test, X_mut_test, yREF_test, yALT_test, yVAR_test,
            df_train, df_test)


# =============================================================================
# Model building
# =============================================================================

def build_siamese_model(encoder, params):
    """Build the siamese model with richer features."""

    # Inputs for reference and mutant sequences
    seq_ref = Input(shape=(180, 4), name="seq_ref")
    seq_mut = Input(shape=(180, 4), name="seq_mut")

    # Pass through shared encoder
    emb_ref = encoder(seq_ref)
    emb_mut = encoder(seq_mut)

    # Predict translation values
    pred_ref = Dense(1, name="pred_ref")(emb_ref)
    pred_alt = Dense(1, name="pred_alt")(emb_mut)

    # RICHER feature representation
    diff = Subtract()([emb_mut, emb_ref])      # difference: mut - ref
    prod = Multiply()([emb_mut, emb_ref])      # element-wise product

    # Concatenate all features
    feat = Concatenate()([emb_ref, emb_mut, diff, prod])

    # Dense head
    x = BatchNormalization()(feat)
    x = Dense(params['dense_units_1'], activation='relu',
              kernel_regularizer=regularizers.l2(params['l2_lambda']))(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(params['dense_units_2'], activation='relu',
              kernel_regularizer=regularizers.l2(params['l2_lambda']))(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(params['dense_units_3'], activation='relu',
              kernel_regularizer=regularizers.l2(params['l2_lambda']))(x)
    x = Dropout(params['dropout_rate'])(x)

    # Use ReLU activation for squared target (non-negative predictions)
    output_activation = 'relu' if params.get('squared', False) else None
    pred_var = Dense(1, activation=output_activation, name="pred_delta")(x)

    # Build Siamese model
    siamese_model = Model(inputs=[seq_ref, seq_mut], outputs=[pred_ref, pred_alt, pred_var])

    # # Compile
    optimizer = Adam(learning_rate=params['learning_rate'])
    siamese_model.compile(
        optimizer=optimizer,
        loss={
            "pred_ref": "mse",
            "pred_alt": "mse",
            "pred_delta": "mse"
        },
        loss_weights={
            "pred_ref": params['loss_weight_ref'],
            "pred_alt": params['loss_weight_alt'],
            "pred_delta": params['loss_weight_delta']
        }
    )

    return siamese_model


# =============================================================================
# Training
# =============================================================================

def train_model(siamese_model, X_ref_train, X_mut_train, yREF_train, yALT_train, yVAR_train,
                X_ref_dev, X_mut_dev, yREF_dev, yALT_dev, yVAR_dev, params):
    """Train the siamese model."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=params['early_stopping_patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=params['reduce_lr_factor'],
            patience=params['reduce_lr_patience'],
            min_lr=params['min_lr']
        ),
    ]

    print("\nStarting training...")
    history = siamese_model.fit(
        [X_ref_train, X_mut_train], [yREF_train, yALT_train, yVAR_train],
        validation_data=([X_ref_dev, X_mut_dev], [yREF_dev, yALT_dev, yVAR_dev]),
        epochs=params['max_epochs'],
        batch_size=params['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    return history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(siamese_model, X_ref_test, X_mut_test, yREF_test, yALT_test, yVAR_test):
    """Evaluate model on test set and return metrics."""
    print("\nEvaluating on test set...")    
    y_test = [yREF_test, yALT_test, yVAR_test]
    
    # Multi-output evaluation
    # For multi-output models, evaluate() returns a list: [loss1, loss2, loss3, total_loss]
    # The total weighted loss is typically the last element
    test_loss_result = siamese_model.evaluate([X_ref_test, X_mut_test], y_test, verbose=0)
    
    # Ensure test_loss is a scalar float
    # For multi-output models, evaluate() returns a list of losses
    if isinstance(test_loss_result, (list, tuple)):
        # Take the total loss (last element) which is the weighted sum
        test_loss = float(test_loss_result[-1])
    elif isinstance(test_loss_result, np.ndarray):
        # If it's a numpy array, take the last element and convert to float
        test_loss = float(test_loss_result.flat[-1])
    else:
        # Already a scalar
        test_loss = float(test_loss_result)
    
    # Get predictions
    y_pred_ref, y_pred_alt, y_pred_var = siamese_model.predict([X_ref_test, X_mut_test], verbose=0)
    
    # Flatten predictions and targets for R² calculation (handle both 1D and 2D)
    yREF_test_flat = yREF_test.flatten()
    yALT_test_flat = yALT_test.flatten()
    yVAR_test_flat = yVAR_test.flatten()
    y_pred_ref_flat = y_pred_ref.flatten()
    y_pred_alt_flat = y_pred_alt.flatten()
    y_pred_var_flat = y_pred_var.flatten()
    
    # Compute R² per output
    r2_ref = r2_score(yREF_test_flat, y_pred_ref_flat)
    r2_alt = r2_score(yALT_test_flat, y_pred_alt_flat)
    r2_var = r2_score(yVAR_test_flat, y_pred_var_flat)
    
    return test_loss, r2_ref, r2_alt, r2_var, y_pred_ref_flat, y_pred_alt_flat, y_pred_var_flat

# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()

    # Parse parameters
    user_params = parse_params_from_args()

    # Merge with defaults
    params = DEFAULT_PARAMS.copy()
    params.update(user_params)

    # Extract output_dir if provided, otherwise build from mode
    output_dir = params.pop('output_dir', None)
    if output_dir is None:
        # Build output directory name: [frozen/unfrozen]_[loss_weights]_[data_fraction]
        mode_parts = []
        
        # Add frozen/unfrozen
        if not params['encoder_trainable']:
            mode_parts.append('frozen')
        else:
            mode_parts.append('unfrozen')
        
        # Add loss weights: format as "0.25_0.25_0.5"
        loss_weights_str = f"{params['loss_weight_ref']}_{params['loss_weight_alt']}_{params['loss_weight_delta']}"
        mode_parts.append(loss_weights_str)
        
        # Add data fraction: "full" if 100%, otherwise "20p" for 20%
        data_fraction = params['data_fraction']
        if data_fraction >= 1.0:
            mode_parts.append('full')
        else:
            # Convert to percentage and add "p" suffix
            percentage = int(data_fraction * 100)
            mode_parts.append(f"{percentage}p")
        
        mode_suffix = '_'.join(mode_parts)
        output_dir = f"{OUTPUT_DIR}/{mode_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Training Richer Siamese Model (Inception Encoder)")
    if params['squared']:
        print("Mode: SQUARED target (magnitude only)")
    else:
        print("Mode: Standard target (with direction)")
    print("=" * 70)
    print("\nHyperparameters:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")
    print("=" * 70)

    # Load and preprocess data
    (X_ref_train, X_mut_train, yREF_train, yALT_train, yVAR_train,
     X_ref_dev, X_mut_dev, yREF_dev, yALT_dev, yVAR_dev,
     X_ref_test, X_mut_test, yREF_test, yALT_test, yVAR_test,
     df_train, df_test) = load_and_preprocess_data(
         DATA_PATH, random_state=params['random_state'],
         squared=params['squared'], data_fraction=params['data_fraction']
     )
    
    # Load encoder model
    print("\nLoading encoder from inception model...")
    model = tf.keras.models.load_model(
        ENCODER_MODEL_PATH,
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

    # Find encoder output layer
    encoder_output = find_encoder_output_layer(model)
    encoder = Model(inputs=model.input, outputs=encoder_output)

    # Set encoder trainability
    encoder.trainable = params['encoder_trainable']

    # Find which layer was used (for debugging)
    last_dense_idx = None
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 1:
            last_dense_idx = i
            break
    encoder_layer_name = model.layers[last_dense_idx - 1].name if last_dense_idx else "unknown"

    print(f"Encoder layer: {encoder_layer_name}")
    print(f"Encoder trainable: {encoder.trainable}")
    print(f"Encoder output shape: {encoder.output_shape}")
    print(f"Encoder trainable weights: {len(encoder.trainable_weights)}")

    # Build siamese model
    print("\nBuilding siamese model...")
    siamese_model = build_siamese_model(encoder, params)
    siamese_model.summary()

    # Train
    history = train_model(
        siamese_model, X_ref_train, X_mut_train, yREF_train, yALT_train, yVAR_train,
                X_ref_dev, X_mut_dev, yREF_dev, yALT_dev, yVAR_dev, params)

    # Evaluate
    test_loss, r2_ref, r2_alt, r2_var, y_pred_ref, y_pred_alt, y_pred_var = evaluate_model(siamese_model, X_ref_test, X_mut_test, yREF_test, yALT_test, yVAR_test)

    # Calculate runtime
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test R² (ΔTE): {r2_var:.6f}")
    print(f"Test R² (REF translation): {r2_ref:.6f}")
    print(f"Test R² (ALT translation): {r2_alt:.6f}")
    print(f"Runtime: {hours}h {minutes}m {seconds}s ({elapsed_time:.2f} seconds)")
    print("=" * 70)

    # Save plots
    print("\nSaving plots...")
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Richer Model: Training and Validation Loss (Test R²={r2_var:.3f})')
    plt.legend()
    plt.savefig(f"{output_dir}/history.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Scatter plots
    # Flatten target arrays for plotting (predictions are already flattened)
    yREF_test_flat = yREF_test.flatten()
    yALT_test_flat = yALT_test.flatten()
    yVAR_test_flat = yVAR_test.flatten()
    
    genes_in_train = set(df_train['gene'])
    seen_mask = df_test['gene'].isin(genes_in_train).values
    unseen_mask = ~seen_mask

    # ========== Variant Effect (ΔTE) Scatter Plots ==========
    plt.figure(figsize=(6, 6))
    plt.scatter(yVAR_test_flat, y_pred_var, alpha=0.3, s=5)
    min_val = min(yVAR_test_flat.min(), y_pred_var.min())
    max_val = max(yVAR_test_flat.max(), y_pred_var.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual ΔTE')
    plt.ylabel('Predicted ΔTE')
    plt.title(f'Test Set: Predicted vs Actual ΔTE (R²={r2_var:.3f})')
    plt.savefig(f"{output_dir}/scatter_test.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Seen genes - Variant Effect
    yVAR_seen = yVAR_test_flat[seen_mask]
    y_pred_var_seen = y_pred_var[seen_mask]
    r2_var_seen = r2_score(yVAR_seen, y_pred_var_seen)
    plt.figure(figsize=(6, 6))
    plt.scatter(yVAR_seen, y_pred_var_seen, alpha=0.3, s=5)
    min_val = min(yVAR_seen.min(), y_pred_var_seen.min())
    max_val = max(yVAR_seen.max(), y_pred_var_seen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual ΔTE')
    plt.ylabel('Predicted ΔTE')
    plt.title(f'Seen Genes: Predicted vs Actual ΔTE (R²={r2_var_seen:.3f})')
    plt.savefig(f"{output_dir}/scatter_seen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Unseen genes - Variant Effect
    yVAR_unseen = yVAR_test_flat[unseen_mask]
    y_pred_var_unseen = y_pred_var[unseen_mask]
    r2_var_unseen = r2_score(yVAR_unseen, y_pred_var_unseen)
    plt.figure(figsize=(6, 6))
    plt.scatter(yVAR_unseen, y_pred_var_unseen, alpha=0.3, s=5)
    min_val = min(yVAR_unseen.min(), y_pred_var_unseen.min())
    max_val = max(yVAR_unseen.max(), y_pred_var_unseen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual ΔTE')
    plt.ylabel('Predicted ΔTE')
    plt.title(f'Unseen Genes: Predicted vs Actual ΔTE (R²={r2_var_unseen:.3f})')
    plt.savefig(f"{output_dir}/scatter_unseen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ========== REF Translation Scatter Plots ==========
    plt.figure(figsize=(6, 6))
    plt.scatter(yREF_test_flat, y_pred_ref, alpha=0.3, s=5)
    min_val = min(yREF_test_flat.min(), y_pred_ref.min())
    max_val = max(yREF_test_flat.max(), y_pred_ref.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual REF Translation')
    plt.ylabel('Predicted REF Translation')
    plt.title(f'Test Set: Predicted vs Actual REF Translation (R²={r2_ref:.3f})')
    plt.savefig(f"{output_dir}/scatter_ref_test.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Seen genes - REF Translation
    yREF_seen = yREF_test_flat[seen_mask]
    y_pred_ref_seen = y_pred_ref[seen_mask]
    r2_ref_seen = r2_score(yREF_seen, y_pred_ref_seen)
    plt.figure(figsize=(6, 6))
    plt.scatter(yREF_seen, y_pred_ref_seen, alpha=0.3, s=5)
    min_val = min(yREF_seen.min(), y_pred_ref_seen.min())
    max_val = max(yREF_seen.max(), y_pred_ref_seen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual REF Translation')
    plt.ylabel('Predicted REF Translation')
    plt.title(f'Seen Genes: Predicted vs Actual REF Translation (R²={r2_ref_seen:.3f})')
    plt.savefig(f"{output_dir}/scatter_ref_seen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Unseen genes - REF Translation
    yREF_unseen = yREF_test_flat[unseen_mask]
    y_pred_ref_unseen = y_pred_ref[unseen_mask]
    r2_ref_unseen = r2_score(yREF_unseen, y_pred_ref_unseen)
    plt.figure(figsize=(6, 6))
    plt.scatter(yREF_unseen, y_pred_ref_unseen, alpha=0.3, s=5)
    min_val = min(yREF_unseen.min(), y_pred_ref_unseen.min())
    max_val = max(yREF_unseen.max(), y_pred_ref_unseen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual REF Translation')
    plt.ylabel('Predicted REF Translation')
    plt.title(f'Unseen Genes: Predicted vs Actual REF Translation (R²={r2_ref_unseen:.3f})')
    plt.savefig(f"{output_dir}/scatter_ref_unseen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ========== ALT Translation Scatter Plots ==========
    plt.figure(figsize=(6, 6))
    plt.scatter(yALT_test_flat, y_pred_alt, alpha=0.3, s=5)
    min_val = min(yALT_test_flat.min(), y_pred_alt.min())
    max_val = max(yALT_test_flat.max(), y_pred_alt.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual ALT Translation')
    plt.ylabel('Predicted ALT Translation')
    plt.title(f'Test Set: Predicted vs Actual ALT Translation (R²={r2_alt:.3f})')
    plt.savefig(f"{output_dir}/scatter_alt_test.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Seen genes - ALT Translation
    yALT_seen = yALT_test_flat[seen_mask]
    y_pred_alt_seen = y_pred_alt[seen_mask]
    r2_alt_seen = r2_score(yALT_seen, y_pred_alt_seen)
    plt.figure(figsize=(6, 6))
    plt.scatter(yALT_seen, y_pred_alt_seen, alpha=0.3, s=5)
    min_val = min(yALT_seen.min(), y_pred_alt_seen.min())
    max_val = max(yALT_seen.max(), y_pred_alt_seen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual ALT Translation')
    plt.ylabel('Predicted ALT Translation')
    plt.title(f'Seen Genes: Predicted vs Actual ALT Translation (R²={r2_alt_seen:.3f})')
    plt.savefig(f"{output_dir}/scatter_alt_seen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Unseen genes - ALT Translation
    yALT_unseen = yALT_test_flat[unseen_mask]
    y_pred_alt_unseen = y_pred_alt[unseen_mask]
    r2_alt_unseen = r2_score(yALT_unseen, y_pred_alt_unseen)
    plt.figure(figsize=(6, 6))
    plt.scatter(yALT_unseen, y_pred_alt_unseen, alpha=0.3, s=5)
    min_val = min(yALT_unseen.min(), y_pred_alt_unseen.min())
    max_val = max(yALT_unseen.max(), y_pred_alt_unseen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual ALT Translation')
    plt.ylabel('Predicted ALT Translation')
    plt.title(f'Unseen Genes: Predicted vs Actual ALT Translation (R²={r2_alt_unseen:.3f})')
    plt.savefig(f"{output_dir}/scatter_alt_unseen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print detailed results
    print("\n" + "=" * 70)
    print("Detailed Results by Gene Split")
    print("=" * 70)
    print(f"\nVariant Effect (ΔTE):")
    print(f"  All test: R² = {r2_var:.6f}")
    print(f"  Seen genes: R² = {r2_var_seen:.6f}")
    print(f"  Unseen genes: R² = {r2_var_unseen:.6f}")
    print(f"\nREF Translation:")
    print(f"  All test: R² = {r2_ref:.6f}")
    print(f"  Seen genes: R² = {r2_ref_seen:.6f}")
    print(f"  Unseen genes: R² = {r2_ref_unseen:.6f}")
    print(f"\nALT Translation:")
    print(f"  All test: R² = {r2_alt:.6f}")
    print(f"  Seen genes: R² = {r2_alt_seen:.6f}")
    print(f"  Unseen genes: R² = {r2_alt_unseen:.6f}")
    print("=" * 70)

    # Save model
    model_path = f"{output_dir}/model.h5"
    # Remove existing model file if it exists to avoid HDF5 conflicts
    if os.path.exists(model_path):
        os.remove(model_path)
    siamese_model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save results
    results = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'test_mse': float(test_loss),
        'test_r2_var': float(r2_var),
        'test_r2_ref': float(r2_ref),
        'test_r2_alt': float(r2_alt),
        'test_r2_var_seen': float(r2_var_seen),
        'test_r2_var_unseen': float(r2_var_unseen),
        'test_r2_ref_seen': float(r2_ref_seen),
        'test_r2_ref_unseen': float(r2_ref_unseen),
        'test_r2_alt_seen': float(r2_alt_seen),
        'test_r2_alt_unseen': float(r2_alt_unseen),
        'runtime_seconds': float(elapsed_time),
        'runtime_formatted': f"{hours}h {minutes}m {seconds}s",
        **params
    }
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/results.json")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
