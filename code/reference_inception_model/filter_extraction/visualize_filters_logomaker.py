#!/usr/bin/env python
"""Visualize filters using logomaker with contribution scores (raw weights).

This script uses logomaker to create sequence logos directly from filter weights,
treating them as contribution scores.
"""

import os
import sys
import numpy as np
import logomaker

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from hyperparameter_tuning.config import OUTPUT_DIR

OUTPUT_DIR_FILTERS = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'filter_extraction', 'extracted_filters')
OUTPUT_DIR_VIS = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'filter_extraction', 'visualizations_logomaker')
os.makedirs(OUTPUT_DIR_VIS, exist_ok=True)

# Base mapping - one-hot encoding order is A, C, G, T (indices 0, 1, 2, 3)
# Logomaker can accept any order, we'll use A, C, G, T
BASES = ['A', 'C', 'G', 'T']


def weights_to_contribution_matrix(filter_weights):
    """
    Convert filter weights to contribution score matrix for logomaker.
    
    Logomaker expects a DataFrame with:
    - Rows: positions (0-indexed or 1-indexed)
    - Columns: A, T, G, C
    - Values: contribution scores (can be positive or negative)
    
    Args:
        filter_weights: Array of shape (filter_size, 4) with raw weights
                       Order: [A, T, G, C] based on one-hot encoding
        
    Returns:
        pd.DataFrame: Contribution matrix for logomaker
    """
    import pandas as pd
    
    filter_size = filter_weights.shape[0]
    
    # Create DataFrame with positions as index and bases as columns
    contrib_matrix = pd.DataFrame(
        filter_weights,
        index=range(1, filter_size + 1),  # 1-indexed positions
        columns=BASES
    )
    
    return contrib_matrix


def create_logomaker_logo(filter_weights, filter_idx, filter_size, output_path):
    """
    Create sequence logo using logomaker with contribution scores.
    
    Args:
        filter_weights: Array of shape (filter_size, 4) with raw weights
        filter_idx: Index of the filter
        filter_size: Size of the filter
        output_path: Path to save the logo
    """
    import matplotlib.pyplot as plt
    
    # Convert weights to contribution matrix
    contrib_matrix = weights_to_contribution_matrix(filter_weights)
    
    # Create logo using logomaker with contribution scores
    fig, ax = plt.subplots(figsize=(max(8, filter_size * 1.2), 4))
    
    # Create logo plot
    # Logomaker can handle contribution scores - they show as positive/negative bars
    logo = logomaker.Logo(
        contrib_matrix,
        ax=ax,
        color_scheme='classic',  # A=green, T=red, G=orange, C=blue
        font_name='Arial Rounded MT Bold',
        vpad=0.05,
        width=0.8
    )
    
    # Style the logo
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_xlabel('Position in Filter', fontsize=13, fontweight='bold')
    logo.ax.set_ylabel('Contribution Score', fontsize=13, fontweight='bold')
    logo.ax.set_title(f'Filter {filter_idx} Motif (size {filter_size} bp)', 
                      fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_combined_logomaker_figure(filters_by_size, output_dir):
    """Create a combined figure with all filters using logomaker."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    for filter_size, filters in filters_by_size.items():
        n_filters = filters.shape[0]
        
        # Determine grid size
        n_cols = min(4, n_filters)
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
        if n_filters == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for filter_idx in range(n_filters):
            filter_weights = filters[filter_idx]
            ax = axes[filter_idx]
            
            # Convert to contribution matrix
            contrib_matrix = weights_to_contribution_matrix(filter_weights)
            
            # Create logo
            logo = logomaker.Logo(
                contrib_matrix,
                ax=ax,
                color_scheme='classic',
                font_name='Arial Rounded MT Bold',
                vpad=0.02,
                width=0.7
            )
            
            # Style
            logo.style_spines(visible=False)
            logo.style_spines(spines=['left'], visible=True)
            logo.ax.set_title(f'Filter {filter_idx}', fontsize=10)
            logo.ax.set_xlabel('') if filter_idx < (n_rows - 1) * n_cols else logo.ax.set_xlabel('Position')
            logo.ax.set_ylabel('') if filter_idx % n_cols != 0 else logo.ax.set_ylabel('Score')
        
        # Hide unused subplots
        for idx in range(n_filters, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'All Filters (size {filter_size}) - Logomaker', fontsize=16, y=0.995)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'all_filters_size_{filter_size}_logomaker.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved combined figure: {output_path}")


def main():
    print("=" * 70)
    print("Visualizing Filters with Logomaker (Contribution Scores)")
    print("=" * 70)
    
    # Load filters
    filters_path = os.path.join(OUTPUT_DIR_FILTERS, 'filters_layer1.npz')
    if not os.path.exists(filters_path):
        raise FileNotFoundError(
            f"Filters not found at {filters_path}. "
            f"Please run extract_filters.py first."
        )
    
    print(f"\nLoading filters from: {filters_path}")
    data = np.load(filters_path, allow_pickle=True)
    
    # Reconstruct filters_by_size dict
    filters_by_size = {}
    filter_sizes = data['filter_sizes']
    
    for size in filter_sizes:
        key = f'filters_size_{size}'
        if key in data:
            filters_by_size[size] = data[key]
        else:
            raise KeyError(f"Filters for size {size} not found in saved file")
    
    print(f"Loaded filters for sizes: {list(filters_by_size.keys())}")
    
    # Create individual visualizations
    print("\nCreating individual logomaker logos...")
    for filter_size, filters in filters_by_size.items():
        n_filters = filters.shape[0]
        print(f"\n  Filter size {filter_size}: {n_filters} filters")
        
        size_dir = os.path.join(OUTPUT_DIR_VIS, f'size_{filter_size}')
        os.makedirs(size_dir, exist_ok=True)
        
        for filter_idx in range(n_filters):
            filter_weights = filters[filter_idx]  # (filter_size, 4)
            
            logo_path = os.path.join(size_dir, f'filter_{filter_idx}_logomaker.png')
            create_logomaker_logo(filter_weights, filter_idx, filter_size, logo_path)
    
    # Create combined figures
    print("\nCreating combined figures...")
    create_combined_logomaker_figure(filters_by_size, OUTPUT_DIR_VIS)
    
    print("\n" + "=" * 70)
    print("Logomaker Visualization Complete!")
    print("=" * 70)
    print(f"\nVisualizations saved to: {OUTPUT_DIR_VIS}")
    print("=" * 70)


if __name__ == "__main__":
    main()

