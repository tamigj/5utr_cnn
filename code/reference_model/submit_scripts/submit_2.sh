#!/bin/bash

conda activate cnn_env

# Generate and submit tuning scripts from config.py
python generate_tuning_scripts.py
