#!/usr/bin/env python
"""Calculate mean R² and standard error (SE) for each T value.

Reads r2_replicates_by_T.csv and computes statistics for each threshold T.
"""

import os
import sys
import csv
import statistics
import math

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from hyperparameter_tuning.config import OUTPUT_DIR

# Path to CSV file
CSV_PATH = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'retrain_on_short_seq', 
                        'experiment', 'r2_replicates_by_T.csv')


def main():
    # Read the CSV file
    data = {}
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = row['T']
            R2 = float(row['R2'])
            if T not in data:
                data[T] = []
            data[T].append(R2)

    # Calculate mean and SE for each T
    print("T\tMean R²\tSE\tN")
    print("-" * 50)
    
    # Sort: numeric T values first, then 'any'
    sorted_Ts = sorted([T for T in data.keys() if T != 'any'], key=int) + (['any'] if 'any' in data else [])
    
    for T in sorted_Ts:
        r2_values = data[T]
        mean_r2 = statistics.mean(r2_values)
        std_r2 = statistics.stdev(r2_values) if len(r2_values) > 1 else 0.0
        n = len(r2_values)
        se = std_r2 / math.sqrt(n)
        print(f"{T}\t{mean_r2:.6f}\t{se:.6f}\t{n}")


if __name__ == "__main__":
    main()
