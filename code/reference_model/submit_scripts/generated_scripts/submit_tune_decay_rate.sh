#!/bin/bash
#SBATCH --job-name=tune_decay_rate
#SBATCH --output=../logs/tune_decay_rate_%j.log
#SBATCH --error=../logs/tune_decay_rate_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py decay_rate 0.75 0.8 0.85 0.9 0.95
