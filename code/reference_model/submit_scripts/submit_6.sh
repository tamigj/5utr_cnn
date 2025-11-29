#!/bin/bash

#SBATCH --job-name=sequential_finetune
#SBATCH --output=../logs/sequential_finetune_%j.out
#SBATCH --error=../logs/sequential_finetune_%j.log
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source activate cnn_env

# Record start time
start_time=$(date +%s)
echo "Script started at: $(date)"

echo "Running sequential fine-tuning of remaining hyperparameters (6_sequential_finetune_remaining_hparams.py)"

python 6_sequential_finetune_remaining_hparams.py

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "Script completed at: $(date)"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"


