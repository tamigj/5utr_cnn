#!/usr/bin/env python
"""Generate and submit tuning scripts for remaining parameters.

This script creates bash submit scripts for testing skip_dropout_in_first_conv_layer
with updated_params_3 as the base. Tests both True and False values.

Usage:
    python 4_generate_remaining_params_scripts.py
"""

import sys
import os
import subprocess

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import remaining_params_grid, updated_params_3, OUTPUT_REMAINING_PARAMS_DIR

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
        return str(value).replace(' ', '')
    else:
        return str(value)

def build_param_args(base_params, param_name, param_value, output_dir):
    """Build command-line arguments for the tunable script.
    
    Args:
        base_params (dict): Base parameters (updated_params_3)
        param_name (str): Name of parameter being tuned
        param_value: Value for the parameter being tuned
        output_dir (str): Output directory for results
    
    Returns:
        list: List of command-line argument strings
    """
    args = []
    
    # Start with base parameters
    current_params = base_params.copy()
    # Override with the parameter being tuned
    current_params[param_name] = param_value
    
    # Build arguments for all parameters
    for key, val in current_params.items():
        if key == 'filter_sizes':
            args.append(f'--filter_sizes={format_param_value(val)}')
        elif isinstance(val, bool):
            args.append(f'--{key}={str(val).lower()}')
        else:
            args.append(f'--{key}={val}')
    
    # Add output directory
    args.append(f'--output_dir={output_dir}')
    
    return args

def create_value_id(value):
    """Create a safe identifier string for a parameter value."""
    if isinstance(value, list):
        return '_'.join(str(v) for v in value)
    elif isinstance(value, bool):
        return str(value).lower()
    else:
        return str(value).replace('.', 'p').replace('-', 'n')

# Generate and submit scripts for each parameter
print("="*70)
print("Generating remaining parameter tuning scripts...")
print("="*70)
print(f"Base parameters: {updated_params_3}")
print(f"Parameters to test: {remaining_params_grid}")
print("="*70)

total_scripts = 0
for param_name, param_values in remaining_params_grid.items():
    print(f"\nGenerating scripts for parameter: {param_name}")
    print(f"  Values to test: {param_values}")
    
    for idx, param_value in enumerate(param_values):
        # Create unique output directory for this parameter value
        value_id = create_value_id(param_value)
        unique_output_dir = os.path.join(OUTPUT_REMAINING_PARAMS_DIR, f'{param_name}_{value_id}')
        
        # Build command-line arguments
        param_args = build_param_args(
            updated_params_3,
            param_name,
            param_value,
            unique_output_dir
        )
        args_str = ' '.join(param_args)
        
        # Create script name
        script_name = f'submit_remaining_{param_name}_{value_id}.sh'
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
#SBATCH --job-name=remaining_{param_name}_{value_id}
#SBATCH --output={logs_dir}/remaining_{param_name}_{value_id}_%j.log
#SBATCH --error={logs_dir}/remaining_{param_name}_{value_id}_%j.err
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
            print(f"  [{idx+1}/{len(param_values)}] ✓ Generated and submitted {script_name} (Job ID: {job_id})")
            print(f"      Value: {param_value}")
            total_scripts += 1
        else:
            print(f"  [{idx+1}/{len(param_values)}] ✗ Error submitting {script_name}: {result.stderr}")

print(f"\n{'='*70}")
print(f"Generated and submitted {total_scripts} remaining parameter tuning scripts")
print(f"Output directory: {OUTPUT_REMAINING_PARAMS_DIR}")
print(f"Generated scripts saved to: {generated_scripts_dir}")
print(f"{'='*70}")

