#!/usr/bin/env python
"""Generate and submit tuning scripts for wide grid search.

This script creates bash submit scripts for each hyperparameter value in the
wide_tuning_grid and submits them to SLURM. Each script trains the model with
one parameter varied while keeping others at initial_params.

Usage:
    python 1_generate_wide_tuning_scripts.py
"""

import sys
import os
import subprocess
import ast

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import wide_tuning_grid, initial_params, OUTPUT_WIDE_DIR

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
        # Format as '[3,5,7]' - the shell will handle quoting when we use it in the script
        return str(value).replace(' ', '')
    else:
        return str(value)

def build_param_args(base_params, param_name, param_value):
    """Build command-line arguments for the tunable script.
    
    Args:
        base_params (dict): Base parameters (initial_params)
        param_name (str): Name of parameter being tuned
        param_value: Value for the parameter being tuned
    
    Returns:
        list: List of command-line argument strings
    """
    args = []
    
    # Add all base parameters
    for key, val in base_params.items():
        if key == param_name:
            # Skip the parameter being tuned - we'll add it with the new value
            continue
        if key == 'filter_sizes':
            # Special handling for filter_sizes (list) - quote it
            args.append(f'--filter_sizes={format_param_value(val)}')
        elif isinstance(val, bool):
            args.append(f'--{key}={str(val).lower()}')
        else:
            args.append(f'--{key}={val}')
    
    # Add the parameter being tuned with its new value
    if param_name == 'filter_sizes':
        args.append(f'--filter_sizes={format_param_value(param_value)}')
    elif isinstance(param_value, bool):
        args.append(f'--{param_name}={str(param_value).lower()}')
    else:
        args.append(f'--{param_name}={param_value}')
    
    # Add output directory
    args.append(f'--output_dir={OUTPUT_WIDE_DIR}')
    
    return args

# Generate and submit scripts for each parameter
total_scripts = 0
for param_name, param_values in wide_tuning_grid.items():
    print(f"\nGenerating scripts for parameter: {param_name}")
    print(f"  Values to test: {param_values}")
    
    for idx, param_value in enumerate(param_values):
        # Build command-line arguments
        param_args = build_param_args(initial_params, param_name, param_value)
        args_str = ' '.join(param_args)
        
        # Create a safe identifier for the value (for filename and output dir)
        if isinstance(param_value, list):
            value_id = '_'.join(str(v) for v in param_value)
        elif isinstance(param_value, bool):
            value_id = str(param_value).lower()
        else:
            value_id = str(param_value).replace('.', 'p')  # Replace . with p for floats
        
        script_name = f'submit_wide_{param_name}_{value_id}.sh'
        script_path = os.path.join(generated_scripts_dir, script_name)
        
        # Create unique output subdirectory for this parameter combination
        unique_output_dir = os.path.join(OUTPUT_WIDE_DIR, f'{param_name}_{value_id}')
        
        # Properly quote arguments that contain special characters (like lists)
        # Split args_str and quote individual arguments that need it
        args_list = args_str.split()
        quoted_args = []
        for arg in args_list:
            if '--filter_sizes=' in arg and '[' in arg:
                # Quote the entire argument to protect brackets from shell
                key, val = arg.split('=', 1)
                quoted_args.append(f'{key}="{val}"')
            elif '--output_dir=' in arg:
                # Replace output_dir with unique subdirectory
                quoted_args.append(f'--output_dir={unique_output_dir}')
            else:
                quoted_args.append(arg)
        safe_args_str = ' '.join(quoted_args)
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=wide_{param_name}_{value_id}
#SBATCH --output={logs_dir}/wide_{param_name}_{value_id}_%j.log
#SBATCH --error={logs_dir}/wide_{param_name}_{value_id}_%j.err
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
            print(f"  ✓ Generated and submitted {script_name} (Job ID: {job_id})")
            total_scripts += 1
        else:
            print(f"  ✗ Error submitting {script_name}: {result.stderr}")

print(f"\n{'='*70}")
print(f"Generated and submitted {total_scripts} tuning scripts for wide grid search")
print(f"Output directory: {OUTPUT_WIDE_DIR}")
print(f"Generated scripts saved to: {generated_scripts_dir}")
print(f"{'='*70}")

