#!/bin/bash
#SBATCH --job-name=tune_filter_size
#SBATCH --output=../logs/tune_filter_size_%j.log
#SBATCH --error=../logs/tune_filter_size_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py filter_size 3 5 7 9 11
