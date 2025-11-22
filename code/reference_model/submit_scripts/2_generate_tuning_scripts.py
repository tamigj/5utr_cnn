#!/usr/bin/env python
"""Generate and submit tuning scripts from tuning grids.

This script creates bash submit scripts for each hyperparameter in the tuning
grid and submits them to SLURM.

Usage:
    python 2_generate_tuning_scripts.py [strategy]
    
    strategy: 'wide', 'coarse', 'fine' (defaults to 'wide')
             Or 'config' to use tuning_grid from config.py
"""

import sys
import os
import subprocess
import argparse

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import tuning_grid as config_tuning_grid
from _tuning_grids import GRIDS

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_scripts_dir = os.path.join(script_dir, 'generated_scripts')
os.makedirs(generated_scripts_dir, exist_ok=True)

def format_values_for_bash(values):
    """Convert Python list to space-separated string for bash.
    
    Args:
        values (list): List of values to convert.
    
    Returns:
        str: Space-separated string of values for use in bash commands.
    """
    return ' '.join(str(v) for v in values)

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate and submit hyperparameter tuning scripts.'
)
parser.add_argument(
    'strategy',
    nargs='?',
    default='wide',
    choices=list(GRIDS.keys()) + ['config'],
    help='Tuning strategy: wide (default), coarse, fine, or config (use config.py)'
)
args = parser.parse_args()

# Select tuning grid based on strategy
if args.strategy == 'config':
    tuning_grid = config_tuning_grid
    print(f"Using tuning_grid from config.py")
else:
    tuning_grid = GRIDS[args.strategy]
    print(f"Using {args.strategy} tuning grid")

# Generate and submit scripts for each parameter
for param_name, param_values in tuning_grid.items():
    values_str = format_values_for_bash(param_values)
    script_path = os.path.join(generated_scripts_dir, f'submit_tune_{param_name}.sh')
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=tune_{param_name}
#SBATCH --output=../logs/tune_{param_name}_%j.log
#SBATCH --error=../logs/tune_{param_name}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py {args.strategy} {param_name} {values_str}
"""
    
    # Write the script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Submit the script
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Generated and submitted {script_path}")
    else:
        print(f"Error submitting {script_path}: {result.stderr}")

print(f"\nGenerated and submitted {len(tuning_grid)} tuning scripts using '{args.strategy}' strategy")

