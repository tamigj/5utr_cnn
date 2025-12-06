#!/usr/bin/env python
"""Extract first-layer filters from the trained Inception CNN model.

This script:
1. Loads the trained model
2. Extracts filter weights from the first convolutional layer (all Inception branches)
3. Saves filters in a structured format for visualization and analysis
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from hyperparameter_tuning.config import OUTPUT_DIR, updated_params_3

# Model path
MODEL_PATH = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'best_model', 
                          'best_model_reference_inception.h5')
OUTPUT_DIR_FILTERS = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'filter_extraction', 'extracted_filters')
os.makedirs(OUTPUT_DIR_FILTERS, exist_ok=True)


def extract_first_layer_filters(model):
    """
    Extract filter weights from the first convolutional layer.
    
    The Inception model has parallel Conv1D branches in the first layer,
    each with different filter sizes (e.g., 3, 5, 7).
    
    Args:
        model: Trained Keras model
        
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
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Concatenate):
            found_first_concat = True
            break
        elif isinstance(layer, tf.keras.layers.Conv1D):
            first_conv_layers.append(layer)
    
    if not first_conv_layers:
        raise ValueError("No Conv1D layers found in the model")
    
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
    print("=" * 70)
    print("Extracting First-Layer Filters from Inception CNN Model")
    print("=" * 70)
    print(f"\nModel path: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Load model (compile=False to avoid deserialization issues)
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully")
    
    # Extract filters
    print("\nExtracting first-layer filters...")
    filter_data = extract_first_layer_filters(model)
    
    # Save filters
    output_path = os.path.join(OUTPUT_DIR_FILTERS, 'filters_layer1.npz')
    # Save each filter size separately
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
    
    np.savez_compressed(output_path, **save_dict)
    
    # Also save a summary
    summary_path = os.path.join(OUTPUT_DIR_FILTERS, 'filter_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("First-Layer Filter Extraction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters:\n")
        for k, v in sorted(filter_data['model_params'].items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFilter sizes: {filter_data['filter_sizes']}\n")
        f.write(f"Filters per branch: {filter_data['n_filters_per_branch']}\n\n")
        f.write("Filters extracted:\n")
        for size, filters in filter_data['filters_by_size'].items():
            f.write(f"  Size {size}: {filters.shape[0]} filters, shape {filters.shape}\n")
    
    print(f"\n✓ Filters saved to: {output_path}")
    print(f"✓ Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Extraction Summary")
    print("=" * 70)
    for size, filters in filter_data['filters_by_size'].items():
        print(f"  Filter size {size}: {filters.shape[0]} filters")
    print("=" * 70)


if __name__ == "__main__":
    main()

