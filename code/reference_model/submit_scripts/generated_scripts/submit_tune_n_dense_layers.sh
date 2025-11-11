#!/bin/bash
#SBATCH --job-name=tune_n_dense_layers
#SBATCH --output=../logs/tune_n_dense_layers_%j.log
#SBATCH --error=../logs/tune_n_dense_layers_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py n_dense_layers 1 2 3 4 5
