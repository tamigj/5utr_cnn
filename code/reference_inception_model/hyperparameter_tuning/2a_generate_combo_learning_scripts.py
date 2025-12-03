#!/usr/bin/env python
"""Generate and submit tuning scripts for combinatorial learning parameter search.

This script creates bash submit scripts for each combination of learning parameters
in combinatorial_tuning_grid_learning and submits them to SLURM. Each script trains
the model with a specific combination of learning_rate, l2_lambda, dropout_rate, and batch_size.

Usage:
    python 2_generate_combo_learning_scripts.py
"""

import sys
import os
import subprocess
import itertools

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import (
    combinatorial_tuning_grid_learning,
    updated_params_1,
    OUTPUT_COMBO_LEARNING_DIR,
)

# Script directory
script_dir = current_dir
generated_scripts_dir = os.path.join(script_dir, 'generated_scripts')
os.makedirs(generated_scripts_dir, exist_ok=True)

# Path to tunable script
tunable_script = os.path.join(parent_dir, 'reference_model_inception_tunable.py')

# Log directory (relative to hyperparameter_tuning/)
logs_dir = os.path.join(current_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

def format_param_value(value):
    """Format a parameter value for command-line argument.
    
    Args:
        value: Parameter value (can be int, float, bool, list, etc.)
    
    Returns:
        str: Formatted string for command-line argument
    """
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, list):
        # Convert list to string representation that can be parsed by ast.literal_eval
        return str(value).replace(' ', '')
    else:
        return str(value)

def build_param_args(base_params, combo_params, output_dir, primary_param_name=None, summary_csv_path=None):
    """Build command-line arguments for the tunable script.
    
    Args:
        base_params (dict): Base parameters (updated_params_1)
        combo_params (dict): Parameters for this combination
        output_dir (str): Output directory for results
    
    Returns:
        list: List of command-line argument strings
    """
    args = []
    
    # Start with base parameters
    current_params = base_params.copy()
    # Override with combinatorial parameters
    current_params.update(combo_params)
    
    # Build arguments for all parameters
    for key, val in current_params.items():
        if key == 'filter_sizes':
            args.append(f'--filter_sizes={format_param_value(val)}')
        elif isinstance(val, bool):
            args.append(f'--{key}={str(val).lower()}')
        else:
            args.append(f'--{key}={val}')
    
    # Add output directory and metadata
    args.append(f'--output_dir={output_dir}')
    if primary_param_name:
        args.append(f'--primary_param_name={primary_param_name}')
    if summary_csv_path:
        args.append(f'--summary_csv={summary_csv_path}')
    
    return args

def create_value_id(value):
    """Create a safe identifier string for a parameter value."""
    if isinstance(value, list):
        return '_'.join(str(v) for v in value)
    elif isinstance(value, bool):
        return str(value).lower()
    else:
        return str(value).replace('.', 'p').replace('-', 'n')

# Generate all combinations
param_names = list(combinatorial_tuning_grid_learning.keys())
param_value_lists = list(combinatorial_tuning_grid_learning.values())
combinations = list(itertools.product(*param_value_lists))
n_combinations = len(combinations)

print("="*70)
print("Generating combinatorial learning parameter tuning scripts...")
print("="*70)
print(f"Parameters: {param_names}")
print(f"Total combinations: {n_combinations}")
print("="*70)

total_scripts = 0
for combo_idx, combo in enumerate(combinations, 1):
    # Create parameter dict for this combination
    combo_params = dict(zip(param_names, combo))
    
    # Determine primary parameter grouping (first parameter in grid)
    primary_param = param_names[0]
    primary_value = combo_params[primary_param]
    primary_value_id = create_value_id(primary_value)
    primary_output_dir = os.path.join(
        OUTPUT_COMBO_LEARNING_DIR, f'{primary_param}_{primary_value_id}'
    )
    
    # Create identifier for logging
    combo_id_parts = []
    for param_name, value in zip(param_names, combo):
        value_id = create_value_id(value)
        combo_id_parts.append(f'{param_name}_{value_id}')
    combo_id = '_'.join(combo_id_parts)
    
    # Build command-line arguments
    global_summary_csv = os.path.join(OUTPUT_COMBO_LEARNING_DIR, "summary.csv")
    param_args = build_param_args(
        updated_params_1,
        combo_params,
        primary_output_dir,
        primary_param_name=primary_param,
        summary_csv_path=global_summary_csv,
    )
    args_str = ' '.join(param_args)
    
    script_name = f'submit_combo_learning_{combo_id}.sh'
    script_path = os.path.join(generated_scripts_dir, script_name)
    
    # Properly quote arguments that contain special characters (like lists)
    args_list = args_str.split()
    quoted_args = []
    for arg in args_list:
        if '--filter_sizes=' in arg and '[' in arg:
            # Quote the entire argument to protect brackets from shell
            key, val = arg.split('=', 1)
            quoted_args.append(f'{key}="{val}"')
        else:
            quoted_args.append(arg)
    safe_args_str = ' '.join(quoted_args)
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=combo_lrn_{combo_id}
#SBATCH --output={logs_dir}/combo_learning_{combo_id}_%j.log
#SBATCH --error={logs_dir}/combo_learning_{combo_id}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

mkdir -p {logs_dir}

source activate cnn_env
python {tunable_script} {safe_args_str}
"""
    
    # Write the script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Submit the script
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"[{combo_idx}/{n_combinations}] ✓ Generated and submitted {script_name} (Job ID: {job_id})")
        print(f"    Parameters: {combo_params}")
        total_scripts += 1
    else:
        print(f"[{combo_idx}/{n_combinations}] ✗ Error submitting {script_name}: {result.stderr}")

print(f"\n{'='*70}")
print(f"Generated and submitted {total_scripts} combinatorial learning tuning scripts")
print(f"Output directory: {OUTPUT_COMBO_LEARNING_DIR}")
print(f"Generated scripts saved to: {generated_scripts_dir}")
print(f"{'='*70}")

