#!/bin/bash

#SBATCH --job-name=post_tune_cnn
#SBATCH --output=../logs/post_tune_%j.log
#SBATCH --error=../logs/post_tune_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

source activate cnn_env

# Pass argument for output filename suffix
python ../post_tuning_model.py wide
