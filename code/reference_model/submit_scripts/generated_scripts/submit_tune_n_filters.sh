#!/bin/bash
#SBATCH --job-name=tune_n_filters
#SBATCH --output=../logs/tune_n_filters_%j.log
#SBATCH --error=../logs/tune_n_filters_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py n_filters 10 50 100 200
