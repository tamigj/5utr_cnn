#!/bin/bash
#SBATCH --job-name=tune_zero_dropout_in_first_layer
#SBATCH --output=../logs/tune_zero_dropout_in_first_layer_%j.log
#SBATCH --error=../logs/tune_zero_dropout_in_first_layer_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py zero_dropout_in_first_layer True False
