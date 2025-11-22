#!/bin/bash

conda activate cnn_env

# Generate and submit tuning scripts using wide search strategy
python 2_generate_tuning_scripts.py wide
