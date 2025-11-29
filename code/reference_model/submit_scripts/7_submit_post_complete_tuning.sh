#!/bin/bash

#SBATCH --job-name=post_complete_tuning
#SBATCH --output=../logs/post_complete_tuning_%j.out
#SBATCH --error=../logs/post_complete_tuning_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source activate cnn_env

# Record start time
start_time=$(date +%s)
echo "Script started at: $(date)"

echo "Training final model with best hyperparameters and evaluating on test set"
echo "This script will:"
echo "  1. Train model with early stopping"
echo "  2. Create train vs dev loss plot"
echo "  3. Save best model to output/tuned_model/"
echo "  4. Evaluate on test set and output R2"
echo "  5. Save predictions to data/model_predictions.tsv"
echo ""

python ../post_complete_tuning_model.py

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "Script completed at: $(date)"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"

