#!/usr/bin/env python
"""Extract first-layer filters from the trained Siamese Inception CNN model.

This script:
1. Loads the trained Siamese model
2. Extracts the encoder (shared between REF and ALT branches)
3. Extracts filter weights from the first convolutional layer (all Inception branches)
4. Saves filters separately for REF and ALT branches (same filters, organized separately)
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import config from reference model to get model parameters
sys.path.insert(0, os.path.join(parent_dir, '..', 'reference_inception_model', 'hyperparameter_tuning'))
from config import OUTPUT_DIR, updated_params_3

# Model path - user should specify which trained model to use
# Default to a recent model, but can be overridden
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, 'variation_inception_model', 'richer_model', 
                                  'unfrozen_0.1_0.1_0.8_20p', 'model.h5')

# Output directories
OUTPUT_DIR_REF = os.path.join(OUTPUT_DIR, 'variation_inception_model', 'filter_extraction', 'siamese_ref', 'extracted_filters')
OUTPUT_DIR_ALT = os.path.join(OUTPUT_DIR, 'variation_inception_model', 'filter_extraction', 'siamese_alt', 'extracted_filters')
os.makedirs(OUTPUT_DIR_REF, exist_ok=True)
os.makedirs(OUTPUT_DIR_ALT, exist_ok=True)


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
        return model.layers[last_dense_idx - 1].output
    
    # Fallback: use second-to-last layer
    if len(model.layers) >= 2:
        return model.layers[-2].output
    
    # Last resort: use the model output (shouldn't happen)
    return model.output


def extract_encoder_from_siamese(siamese_model):
    """
    Extract the encoder from the Siamese model.
    
    The Siamese model has two inputs: seq_ref and seq_mut.
    Both use the same encoder. We extract it from the REF branch.
    
    Args:
        siamese_model: Trained Siamese Keras model
        
    Returns:
        encoder: Keras Model that is the encoder
    """
    # Method 1: Look for Model layers (the encoder might be stored as a Model layer)
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.Model):
            # Check if this model has the right input shape (180, 4)
            try:
                if layer.input_shape == (None, 180, 4) or layer.input_shape[1:] == (180, 4):
                    print(f"  Found encoder as Model layer: {layer.name}")
                    return layer
            except:
                pass
    
    # Method 2: Extract by tracing from seq_ref input to emb_ref output
    # Find the Concatenate layer that combines features
    concat_layer = None
    for layer in siamese_model.layers:
        if isinstance(layer, tf.keras.layers.Concatenate):
            concat_layer = layer
            break
    
    if concat_layer is None:
        raise ValueError("Could not find Concatenate layer in Siamese model")
    
    # The concat layer takes [emb_ref, emb_mut, diff, prod] as inputs
    # Get the inputs to concat_layer
    concat_inputs = concat_layer.input
    
    # emb_ref should be the first input
    if isinstance(concat_inputs, list) and len(concat_inputs) >= 2:
        emb_ref_output = concat_inputs[0]  # First input is emb_ref
        
        # Build a submodel from seq_ref input to emb_ref output
        ref_input = siamese_model.input[0]  # First input is seq_ref
        encoder_model = tf.keras.Model(inputs=ref_input, outputs=emb_ref_output)
        
        print(f"  Extracted encoder by tracing from seq_ref to emb_ref")
        return encoder_model
    
    raise ValueError("Could not extract encoder from Siamese model")


def extract_first_layer_filters(encoder_model):
    """
    Extract filter weights from the first convolutional layer of the encoder.
    
    The Inception model has parallel Conv1D branches in the first layer,
    each with different filter sizes (e.g., 3, 5, 7).
    
    Args:
        encoder_model: Encoder Keras model (extracted from Siamese)
        
    Returns:
        dict: Dictionary with keys:
            - 'filters_by_size': dict mapping filter_size -> array of filters
            - 'filter_sizes': list of filter sizes
            - 'n_filters_per_branch': number of filters per branch
            - 'model_params': parameters used to build the model
    """
    # Get model parameters
    filter_sizes = updated_params_3['filter_sizes']
    n_filters = updated_params_3['n_filters']
    n_branches = len(filter_sizes)
    n_filters_per_branch = max(1, n_filters // n_branches)
    
    # Find all Conv1D layers before the first Concatenate layer
    # In the Inception architecture, the first layer has parallel Conv1D branches
    # that are concatenated together
    first_conv_layers = []
    found_first_concat = False
    
    for i, layer in enumerate(encoder_model.layers):
        if isinstance(layer, tf.keras.layers.Concatenate):
            found_first_concat = True
            break
        elif isinstance(layer, tf.keras.layers.Conv1D):
            first_conv_layers.append(layer)
    
    if not first_conv_layers:
        raise ValueError("No Conv1D layers found in the encoder model")
    
    print(f"Found {len(first_conv_layers)} Conv1D layers in first layer")
    
    # Extract filters by size
    filters_by_size = {}
    
    for layer in first_conv_layers:
        # Get kernel weights: shape is (kernel_size, input_channels, num_filters)
        kernel = layer.get_weights()[0]  # [0] is kernel, [1] is bias
        kernel_size = kernel.shape[0]
        num_filters = kernel.shape[2]
        
        print(f"  Filter size {kernel_size}: {num_filters} filters")
        
        # Transpose to (num_filters, kernel_size, input_channels) for easier handling
        # Shape: (num_filters, kernel_size, 4) where 4 is A, T, G, C
        filters = np.transpose(kernel, (2, 0, 1))
        filters_by_size[kernel_size] = filters
    
    return {
        'filters_by_size': filters_by_size,
        'filter_sizes': filter_sizes,
        'n_filters_per_branch': n_filters_per_branch,
        'model_params': updated_params_3.copy()
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract filters from Siamese model encoder')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to trained Siamese model')
    args = parser.parse_args()
    
    model_path = args.model_path
    
    print("=" * 70)
    print("Extracting First-Layer Filters from Siamese Inception CNN Model")
    print("=" * 70)
    print(f"\nModel path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load Siamese model (compile=False to avoid deserialization issues)
    print("\nLoading Siamese model...")
    siamese_model = tf.keras.models.load_model(model_path, compile=False)
    print("✓ Siamese model loaded successfully")
    
    # Extract encoder from Siamese model
    print("\nExtracting encoder from Siamese model...")
    encoder_model = extract_encoder_from_siamese(siamese_model)
    print("✓ Encoder extracted successfully")
    print(f"  Encoder input shape: {encoder_model.input_shape}")
    print(f"  Encoder output shape: {encoder_model.output_shape}")
    
    # Extract filters
    print("\nExtracting first-layer filters...")
    filter_data = extract_first_layer_filters(encoder_model)
    
    # Save filters for REF branch
    output_path_ref = os.path.join(OUTPUT_DIR_REF, 'filters_layer1.npz')
    save_dict = {
        'filter_sizes': np.array(filter_data['filter_sizes']),
        'n_filters_per_branch': np.array(filter_data['n_filters_per_branch']),
    }
    # Add filters for each size
    for size, filters in filter_data['filters_by_size'].items():
        save_dict[f'filters_size_{size}'] = filters
    # Add model parameters
    for k, v in filter_data['model_params'].items():
        if isinstance(v, (int, float, bool)):
            save_dict[f'param_{k}'] = np.array([v])
        elif isinstance(v, list):
            save_dict[f'param_{k}'] = np.array(v)
        else:
            save_dict[f'param_{k}'] = np.array([str(v)])
    
    np.savez_compressed(output_path_ref, **save_dict)
    
    # Save summary for REF
    summary_path_ref = os.path.join(OUTPUT_DIR_REF, 'filter_summary.txt')
    with open(summary_path_ref, 'w') as f:
        f.write("First-Layer Filter Extraction Summary (REF Branch)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters:\n")
        for k, v in sorted(filter_data['model_params'].items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFilter sizes: {filter_data['filter_sizes']}\n")
        f.write(f"Filters per branch: {filter_data['n_filters_per_branch']}\n\n")
        f.write("Filters extracted:\n")
        for size, filters in filter_data['filters_by_size'].items():
            f.write(f"  Size {size}: {filters.shape[0]} filters, shape {filters.shape}\n")
    
    # Save filters for ALT branch (same filters, but organized separately)
    output_path_alt = os.path.join(OUTPUT_DIR_ALT, 'filters_layer1.npz')
    np.savez_compressed(output_path_alt, **save_dict)
    
    # Save summary for ALT
    summary_path_alt = os.path.join(OUTPUT_DIR_ALT, 'filter_summary.txt')
    with open(summary_path_alt, 'w') as f:
        f.write("First-Layer Filter Extraction Summary (ALT Branch)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters:\n")
        for k, v in sorted(filter_data['model_params'].items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFilter sizes: {filter_data['filter_sizes']}\n")
        f.write(f"Filters per branch: {filter_data['n_filters_per_branch']}\n\n")
        f.write("Filters extracted:\n")
        for size, filters in filter_data['filters_by_size'].items():
            f.write(f"  Size {size}: {filters.shape[0]} filters, shape {filters.shape}\n")
    
    print(f"\n✓ Filters saved to:")
    print(f"  REF: {output_path_ref}")
    print(f"  ALT: {output_path_alt}")
    print(f"✓ Summaries saved to:")
    print(f"  REF: {summary_path_ref}")
    print(f"  ALT: {summary_path_alt}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Extraction Summary")
    print("=" * 70)
    for size, filters in filter_data['filters_by_size'].items():
        print(f"  Filter size {size}: {filters.shape[0]} filters")
    print("=" * 70)


if __name__ == "__main__":
    main()
