#!/bin/bash
#SBATCH --job-name=tune_batch_size
#SBATCH --output=../logs/tune_batch_size_%j.log
#SBATCH --error=../logs/tune_batch_size_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py batch_size 64 128 256 512
