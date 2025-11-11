#!/bin/bash
#SBATCH --job-name=tune_dropout_rate
#SBATCH --output=../logs/tune_dropout_rate_%j.log
#SBATCH --error=../logs/tune_dropout_rate_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py dropout_rate 0.1 0.2 0.3 0.4 0.5
