#!/usr/bin/env python
"""Sequentially fine-tune remaining hyperparameters starting from updated_params_4.

This script:
    - Loads updated_params_4 from config.py (you will update its values after
      architecture combinatorial tuning is finished).
    - Defines a set of "remaining" hyperparameters that were NOT tuned in the
      previous combinatorial learning-rate and architecture searches.
    - For each such hyperparameter, in sequence:
        * runs tuning for up to 4 candidate values,
        * selects the value with the lowest dev (validation) loss,
        * updates the current parameter set with this best value,
        * and proceeds to the next hyperparameter.
    - Uses existing utilities:
        * utils.tune_hyperparameter
        * utils.get_tuning_output_dir
    - Produces the same style of plots and intermediate artifacts as the
      earlier single-parameter tuning (hyperparameter_tuning.py).

Usage:
    python 6_sequential_finetune_remaining_hparams.py
"""

import os
import sys
import numpy as np

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_DIR, tuning_grid, updated_params_4  # type: ignore
from utils import tune_hyperparameter, get_tuning_output_dir


def main():
    # Load preprocessed data
    data = np.load(f"{DATA_DIR}/preprocessed_data.npz")
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_dev = data["X_dev"]
    Y_dev = data["Y_dev"]

    # Start from updated_params_4 (post-architecture best settings)
    current_params = updated_params_4.copy()

    # Use the smallest batch size we ever consider to define steps_per_epoch
    min_batch_size = min(tuning_grid.get("batch_size", [current_params["batch_size"]]))
    steps_per_epoch = len(X_train) // max(1, int(min_batch_size))

    # Hyperparameters already tuned in recent combinatorial runs:
    already_tuned = {
        "learning_rate",
        "l2_lambda",
        "dropout_rate",
        "batch_size",
        "n_conv_layers",
        "n_filters",
        "filter_size",
        "n_dense_units",
    }

    # Candidate hyperparameters to consider from the global tuning grid
    candidate_hparams = [hp for hp in tuning_grid.keys() if hp not in already_tuned]

    print("Starting sequential fine-tuning of remaining hyperparameters.")
    print(f"Initial params (updated_params_4): {current_params}")
    print(f"Remaining hyperparameters to tune (in order): {candidate_hparams}")

    # Strategy / output naming
    base_strategy = "sequential_finetune"
    _ = get_tuning_output_dir(base_strategy)  # ensure base dir exists

    # Sequentially tune each remaining hyperparameter
    for idx, param_name in enumerate(candidate_hparams, start=1):
        values = tuning_grid.get(param_name)
        if not values:
            print(f"Skipping '{param_name}': no candidate values found in tuning_grid.")
            continue

        # Take up to 4 candidate values as requested (first 4 from tuning_grid)
        candidate_values = values[:4]
        if len(candidate_values) < 2:
            print(f"Skipping '{param_name}': need at least 2 values, found {candidate_values}")
            continue

        # Strategy name per-parameter (keeps artifacts separate but grouped)
        strategy = f"{base_strategy}_{idx}_{param_name}"
        output_dir = get_tuning_output_dir(strategy)

        print("\n========================================")
        print(f"[{idx}/{len(candidate_hparams)}] Tuning '{param_name}'")
        print(f"Candidate values: {candidate_values}")

        # Run tuning using the current parameters as base
        results = tune_hyperparameter(
            param_name,
            candidate_values,
            current_params,
            steps_per_epoch,
            X_train,
            Y_train,
            X_dev,
            Y_dev,
            output_dir,
            strategy,
        )

        # Pick the value with the lowest dev (validation) loss
        best_value = None
        best_val_loss = None
        for res in results:
            history = res["history"]
            val_losses = history.history.get("val_loss", [])
            if not val_losses:
                continue
            cur_best = min(val_losses)
            if best_val_loss is None or cur_best < best_val_loss:
                best_val_loss = cur_best
                best_value = res["value"]

        if best_value is None:
            print(f"Could not determine best value for '{param_name}' (no val_loss found).")
            continue

        # Update the running parameter set
        current_params[param_name] = best_value
        print(f"Best value for '{param_name}': {best_value} (best dev loss = {best_val_loss:.4f})")
        print(f"Updated params after tuning '{param_name}': {current_params}")

    print("\nSequential fine-tuning complete.")
    print("Final tuned parameters:")
    for k in sorted(current_params.keys()):
        print(f"  {k}: {current_params[k]}")


if __name__ == "__main__":
    main()


