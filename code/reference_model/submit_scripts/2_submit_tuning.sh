#!/bin/bash

conda activate cnn_env

mkdir -p generated_scripts

# Define tuning grid
declare -A tuning_grid

tuning_grid[learning_rate]="0.0001 0.001 0.01 0.1"
tuning_grid[n_conv_layers]="1 3 5 7"
tuning_grid[n_filters]="10 50 100 200"
tuning_grid[filter_size]="3 5 7 9 11"
tuning_grid[n_dense_layers]="1 2 3 4 5"
tuning_grid[n_dense_units]="25 50 100 200 500"
tuning_grid[decay_rate]="0.75 0.8 0.85 0.9 0.95"
tuning_grid[epoch_decay_interval]="5 10 25 50"
tuning_grid[l2_lambda]="0.001 0.01 0.1 1"
tuning_grid[dropout_rate]="0.1 0.2 0.3 0.4 0.5"
tuning_grid[zero_dropout_in_first_layer]="True False"
tuning_grid[batch_size]="64 128 256 512"

# Create and submit a job script for each parameter
for param in "${!tuning_grid[@]}"
do
    values="${tuning_grid[$param]}"
    script_name="generated_scripts/submit_tune_${param}.sh"

    # Create the submit script
    cat > ${script_name} << EOF
#!/bin/bash
#SBATCH --job-name=tune_${param}
#SBATCH --output=../logs/tune_${param}_%j.log
#SBATCH --error=../logs/tune_${param}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

conda activate cnn_env
python ../hyperparameter_tuning.py ${param} ${values}
EOF

    # Submit the script
    sbatch ${script_name}
    echo "Submitted ${script_name}"
done
