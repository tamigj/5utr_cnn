#!/usr/bin/env python3
"""Extract R² values from variation model log files.

For each weight combination, selects the log file with the highest log number
and extracts R² values for variant effect (overall, seen genes, unseen genes).
"""

import os
import re
import glob
import pandas as pd
from pathlib import Path

#------------#
# VARIABLES  #
#------------#
# Use absolute path to workspace root
WORKSPACE_ROOT = '/mnt/oak/users/tami/5utr_cnn'

DIR_LOGS = os.path.join(WORKSPACE_ROOT, 'code/variation_inception_model/logs')
OUTPUT_CSV = os.path.join(WORKSPACE_ROOT, 'output/variation_inception_model/data_analysis/r2_from_logs.csv')

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

#------------#
# FUNCTIONS  #
#------------#
def extract_r2_from_log(log_file):
    """Extract R² values from a log file.
    
    Returns:
        dict with keys: r2_all, r2_seen, r2_unseen, or None if not found
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the "Detailed Results by Gene Split" section
    start_idx = None
    for i, line in enumerate(lines):
        if "Detailed Results by Gene Split" in line:
            start_idx = i
            break
    
    if start_idx is None:
        return None
    
    # Find "Variant Effect (ΔTE):" section
    var_effect_idx = None
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        if "Variant Effect" in lines[i]:
            var_effect_idx = i
            break
    
    if var_effect_idx is None:
        return None
    
    # Extract R² values from the next few lines
    r2_all = None
    r2_seen = None
    r2_unseen = None
    
    for i in range(var_effect_idx + 1, min(var_effect_idx + 5, len(lines))):
        line = lines[i]
        if "All test" in line and "R²" in line:
            match = re.search(r'R²\s*=\s*([0-9.]+)', line)
            if match:
                r2_all = float(match.group(1))
        elif "Seen genes" in line and "R²" in line:
            match = re.search(r'R²\s*=\s*([0-9.]+)', line)
            if match:
                r2_seen = float(match.group(1))
        elif "Unseen genes" in line and "R²" in line:
            match = re.search(r'R²\s*=\s*([0-9.]+)', line)
            if match:
                r2_unseen = float(match.group(1))
    
    if r2_all is None and r2_seen is None and r2_unseen is None:
        return None
    
    return {
        'r2_all': r2_all,
        'r2_seen': r2_seen,
        'r2_unseen': r2_unseen
    }

#------------#
# MAIN       #
#------------#
# Get all log files matching pattern: variation_inception_unfrozen_*_*_*_20p_*.log or *_full_*.log
pattern_20p = os.path.join(DIR_LOGS, 'variation_inception_unfrozen_*_20p_*.log')
pattern_full = os.path.join(DIR_LOGS, 'variation_inception_unfrozen_*_full_*.log')
all_files = glob.glob(pattern_20p) + glob.glob(pattern_full)

print(f"Found {len(all_files)} log files")

# Parse filenames to extract weights and log number
# Pattern: variation_inception_unfrozen_0.05_0.05_0.9_20p_19060.log
# or: variation_inception_unfrozen_0.05_0.05_0.9_full_19075.log
file_info = []

for file_path in all_files:
    filename = os.path.basename(file_path)
    # Try 20p pattern first
    match = re.match(r'variation_inception_unfrozen_([0-9.]+_[0-9.]+_[0-9.]+)_20p_([0-9]+)\.log', filename)
    if match:
        weights = match.group(1)
        log_num = int(match.group(2))
        data_type = "20p"
    else:
        # Try full pattern
        match = re.match(r'variation_inception_unfrozen_([0-9.]+_[0-9.]+_[0-9.]+)_full_([0-9]+)\.log', filename)
        if match:
            weights = match.group(1)
            log_num = int(match.group(2))
            data_type = "full"
        else:
            continue
    
    file_info.append({
        'file': file_path,
        'filename': filename,
        'weights': weights,
        'log_num': log_num,
        'data_type': data_type
    })

# Convert to DataFrame for easier manipulation
df_files = pd.DataFrame(file_info)

if len(df_files) == 0:
    print("No matching log files found!")
    exit(1)

# For each weight combination and data type, keep only the file with highest log number
df_files = df_files.sort_values(['weights', 'data_type', 'log_num'], ascending=[True, True, False])
df_files = df_files.drop_duplicates(subset=['weights', 'data_type'], keep='first')

print(f"After selecting highest log number per combination: {len(df_files)} files")

# Extract R² values from each log file
results_list = []

for _, row in df_files.iterrows():
    file_path = row['file']
    weights = row['weights']
    
    # Parse weights
    weight_parts = weights.split('_')
    if len(weight_parts) >= 3:
        loss_weight_ref = float(weight_parts[0])
        loss_weight_alt = float(weight_parts[1])
        loss_weight_delta = float(weight_parts[2])
        
        # Extract R² values
        r2_data = extract_r2_from_log(file_path)
        
        if r2_data is not None:
            results_list.append({
                'combo': weights,
                'loss_weight_ref': loss_weight_ref,
                'loss_weight_alt': loss_weight_alt,
                'loss_weight_delta': loss_weight_delta,
                'data_type': row['data_type'],
                'r2_test': r2_data['r2_all'],
                'r2_seen': r2_data['r2_seen'],
                'r2_unseen': r2_data['r2_unseen']
            })
            print(f"Extracted from {row['filename']} ({row['data_type']}) - R² test: {r2_data['r2_all']}, seen: {r2_data['r2_seen']}, unseen: {r2_data['r2_unseen']}")
        else:
            print(f"Warning: Could not extract R² from {row['filename']}")

# Combine into DataFrame
if len(results_list) == 0:
    print("No R² values extracted from log files!")
    exit(1)

df = pd.DataFrame(results_list)

# Sort by loss weights (delta descending, then ref, then alt)
df = df.sort_values(['loss_weight_delta', 'loss_weight_ref', 'loss_weight_alt'], 
                    ascending=[False, True, True])

print("\nFinal data frame:")
print(df)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to: {OUTPUT_CSV}")
