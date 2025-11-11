#!/bin/bash
#SBATCH --job-name=tune_learning_rate
#SBATCH --output=../logs/tune_learning_rate_%j.log
#SBATCH --error=../logs/tune_learning_rate_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py learning_rate 0.0001 0.001 0.01 0.1
