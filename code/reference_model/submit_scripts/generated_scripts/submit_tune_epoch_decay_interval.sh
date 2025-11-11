#!/bin/bash
#SBATCH --job-name=tune_epoch_decay_interval
#SBATCH --output=../logs/tune_epoch_decay_interval_%j.log
#SBATCH --error=../logs/tune_epoch_decay_interval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py epoch_decay_interval 5 10 25 50
