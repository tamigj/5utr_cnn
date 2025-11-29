#!/usr/bin/env python
"""Generate and submit combinatorial hyperparameter tuning scripts.

This script creates SLURM submit scripts for combinatorial hyperparameter tuning,
grouping by learning rate (one job per learning rate value).

Usage:
    python 4_generate_learning_combinatorial_scripts.py
"""

import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import combinatorial_tuning_grid_learning

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_scripts_dir = os.path.join(script_dir, 'generated_scripts')
os.makedirs(generated_scripts_dir, exist_ok=True)
logs_dir = os.path.abspath(os.path.join(script_dir, '..', 'logs'))
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, 'generate_combinatorial_scripts.log')

def _log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(line + '\n')

# Define parameter grids (sourced from config.py)
learning_grid = combinatorial_tuning_grid_learning
learning_rates = learning_grid['learning_rate']
l2_lambda_values = learning_grid['l2_lambda']
dropout_values = learning_grid['dropout_rate']
batch_size_values = learning_grid['batch_size']

base_strategy = 'combo_learning'
mode = 'learning'

# Calculate total combinations per job
combinations_per_job = len(l2_lambda_values) * len(dropout_values) * len(batch_size_values)
_log(f"Each job will test {combinations_per_job} combinations")
_log(f"Total combinations across all jobs: {len(learning_rates) * combinations_per_job}")

# Generate and submit scripts for each learning rate
for lr in learning_rates:
    # Create a safe filename for the learning rate
    lr_str = str(lr).replace('.', '_')
    strategy = f'{base_strategy}_lr_{lr_str}'
    script_path = os.path.join(generated_scripts_dir, f'submit_combo_tune_lr_{lr_str}.sh')
    output_path = os.path.join(logs_dir, f'combo_lr_{lr_str}_%j.out')
    error_path = os.path.join(logs_dir, f'combo_lr_{lr_str}_%j.log')
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=combo_lr_{lr_str}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source activate cnn_env
python ../combinatorial_tuning.py {strategy} {mode} {lr}
"""
    
    # Write the script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Submit the script
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        _log(f"Generated and submitted {script_path} (strategy={strategy}, learning_rate={lr})")
        if result.stdout:
            _log(result.stdout.strip())
    else:
        _log(f"Error submitting {script_path}: {result.stderr.strip()}")

_log(f"Generated and submitted {len(learning_rates)} combinatorial tuning scripts")
_log(f"Each script tests {combinations_per_job} combinations")


