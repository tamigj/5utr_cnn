#!/bin/bash

#SBATCH --job-name=baseline_cnn
#SBATCH --output=../logs/baseline_%j.log
#SBATCH --error=../logs/baseline_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

conda activate cnn_env

python ../data_preprocessing.py
python ../baseline_model.py
