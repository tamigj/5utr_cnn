#!/bin/bash
#SBATCH --job-name=tune_l2_lambda
#SBATCH --output=../logs/tune_l2_lambda_%j.log
#SBATCH --error=../logs/tune_l2_lambda_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py l2_lambda 0.001 0.01 0.1 1
