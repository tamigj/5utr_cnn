#!/bin/bash

#SBATCH --job-name=generate_combo_scripts
#SBATCH --output=../logs/generate_combo_scripts_%j.out
#SBATCH --error=../logs/generate_combo_scripts_%j.log
#SBATCH --time=00:20:00
#SBATCH --mem=8G

source activate cnn_env

# Record start time
start_time=$(date +%s)
echo "Script started at: $(date)"

# Generate and submit combinatorial learning-rate tuning scripts (config-driven counts)
python 4_generate_learning_combinatorial_scripts.py

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "Script completed at: $(date)"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"

