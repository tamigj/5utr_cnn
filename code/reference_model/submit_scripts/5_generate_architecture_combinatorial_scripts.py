#!/usr/bin/env python
"""Generate and submit combinatorial architecture tuning scripts.

Creates one SLURM job per n_conv_layers value, sweeping the remaining
architecture parameters defined in config.combinatorial_tuning_grid_architecture.
"""

import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import combinatorial_tuning_grid_architecture

# Script directories
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_scripts_dir = os.path.join(script_dir, 'generated_scripts')
os.makedirs(generated_scripts_dir, exist_ok=True)
logs_dir = os.path.abspath(os.path.join(script_dir, '..', 'logs'))
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, 'generate_architecture_combinatorial_scripts.log')


def _log(message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(line + '\n')


# Parameter grids (from config.py)
arch_grid = combinatorial_tuning_grid_architecture
n_conv_layers_values = arch_grid['n_conv_layers']
n_filters_values = arch_grid['n_filters']
filter_size_values = arch_grid['filter_size']
n_dense_units_values = arch_grid['n_dense_units']

base_strategy = 'combo_architecture'
mode = 'architecture'

# Calculate total combinations per job (exclude the per-job n_conv_layers value)
combinations_per_job = (
    len(n_filters_values) *
    len(filter_size_values) *
    len(n_dense_units_values)
)
_log(f"Each architecture job will test {combinations_per_job} combinations")
_log(f"Total combinations across all jobs: {len(n_conv_layers_values) * combinations_per_job}")

# Generate and submit scripts for each n_conv_layers value
for n_conv in n_conv_layers_values:
    n_conv_str = str(n_conv)
    strategy = f'{base_strategy}_n{n_conv_str}'
    script_path = os.path.join(generated_scripts_dir, f'submit_combo_arch_nconv_{n_conv_str}.sh')
    output_path = os.path.join(logs_dir, f'combo_arch_nconv_{n_conv_str}_%j.out')
    error_path = os.path.join(logs_dir, f'combo_arch_nconv_{n_conv_str}_%j.log')

    script_content = f"""#!/bin/bash
#SBATCH --job-name=combo_arch_n{n_conv_str}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source activate cnn_env
python ../combinatorial_tuning.py {strategy} {mode} {n_conv}
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)

    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        _log(f"Generated and submitted {script_path} (strategy={strategy}, n_conv_layers={n_conv})")
        if result.stdout:
            _log(result.stdout.strip())
    else:
        _log(f"Error submitting {script_path}: {result.stderr.strip()}")

_log(f"Generated and submitted {len(n_conv_layers_values)} architecture tuning scripts")
_log(f"Each script tests {combinations_per_job} combinations")


