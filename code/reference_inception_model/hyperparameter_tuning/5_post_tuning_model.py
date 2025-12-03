#!/usr/bin/env python
"""Train final model with optimal hyperparameters and generate evaluation outputs.

This script:
1. Trains the model with optimal parameters (updated_params_3 + skip_dropout=False)
2. Saves the best model to output/reference_inception_model/best_model/
3. Calculates R2 and saves it
4. Creates scatterplot of true vs predicted with separate R2 for positive/negative
5. Saves predictions file with ref data
6. Saves all evaluation outputs to output/reference_inception_model/best_model_evaluation/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import updated_params_3, DATA_DIR, OUTPUT_DIR
from reference_model_inception_tunable import train_inception_model

# Output directories
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'best_model')
EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'reference_inception_model', 'best_model_evaluation')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Best parameters (updated_params_3 with skip_dropout_in_first_conv_layer=False confirmed from Phase 4)
best_params = updated_params_3.copy()
best_params['skip_dropout_in_first_conv_layer'] = False

print("=" * 70)
print("Training Final Model with Optimal Hyperparameters")
print("=" * 70)
print("\nHyperparameters:")
for k, v in sorted(best_params.items()):
    print(f"  {k}: {v}")
print()

# Train the model
print("Training model...")
results = train_inception_model(
    best_params.copy(),
    output_dir=EVAL_OUTPUT_DIR,  # Training curve will go here
    verbose=1
)

# Get the trained model from results
model = results['model']
history = results['history']

# Save the best model
model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_model_reference_inception.h5')
model.save(model_path)
print(f"\nSaved best model to {model_path}")

# Load test data for evaluation
print("\n" + "=" * 70)
print("Evaluating on Test Set")
print("=" * 70)

data = np.load(os.path.join(DATA_DIR, 'preprocessed_data.npz'))
X_test = data['X_test']
Y_test = data['Y_test']

# Get predictions
Y_pred = model.predict(X_test, verbose=0).flatten()

# Calculate overall R2
r2_overall = r2_score(Y_test, Y_pred)
print(f"Test R²: {r2_overall:.6f}")

# Save R2 metrics to file
r2_file = os.path.join(MODEL_OUTPUT_DIR, 'r2_scores.txt')
with open(r2_file, 'w') as f:
    f.write("R² Scores\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test R²: {r2_overall:.6f}\n")
print(f"\nSaved R² scores to {r2_file}")

# Create scatterplot
print("\nCreating scatterplot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all points in black
ax.scatter(Y_test, Y_pred, alpha=0.6, color='black', s=30)

# Add diagonal line (perfect prediction)
min_val = min(Y_test.min(), Y_pred.min())
max_val = max(Y_test.max(), Y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='Perfect prediction')

# Add R² text
ax.text(0.05, 0.95, f'R² = {r2_overall:.3f}', 
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('True Translation Efficiency', fontsize=12)
ax.set_ylabel('Predicted Translation Efficiency', fontsize=12)
ax.set_title('Test Set: Predicted vs True Translation Efficiency', fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
scatter_path = os.path.join(EVAL_OUTPUT_DIR, 'true_vs_predicted_scatter.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved scatterplot to {scatter_path}")

# Load original dataframe to get ref data for test set
print("\nLoading reference data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'ref_data.tsv'), sep='\t')
df['ref_sequence'] = df['ref_sequence'].astype(str)

# Split data with same random_state as preprocessing to get matching test_df
train_prop = 0.7
dev_prop = 0.2
test_prop = 0.1
random_state = 42  # Must match data_preprocessing.py

train_df, tmp_df = train_test_split(df, train_size=train_prop,
                                    random_state=random_state, shuffle=True)
test_size_in_tmp = test_prop / (dev_prop + test_prop)
dev_df, test_df = train_test_split(tmp_df, test_size=test_size_in_tmp,
                                   random_state=random_state, shuffle=True)

# Create predictions dataframe with ref data
print("Creating predictions file...")
predictions_df = pd.DataFrame({
    'true_translation': Y_test,
    'predicted_translation': Y_pred
})

# Add columns from test_df
for col in test_df.columns:
    if col not in predictions_df.columns:
        predictions_df[col] = test_df[col].values

# Reorder columns to put ref data first, then predictions
cols = list(test_df.columns) + ['true_translation', 'predicted_translation']
predictions_df = predictions_df[cols]

# Save predictions
predictions_path = os.path.join(EVAL_OUTPUT_DIR, 'predictions_with_ref_data.tsv')
predictions_df.to_csv(predictions_path, sep='\t', index=False)
print(f"Saved predictions with ref data to {predictions_path} ({len(predictions_df)} rows)")

print("\n" + "=" * 70)
print("Final Model Training and Evaluation Complete!")
print("=" * 70)
print(f"\nModel saved to: {MODEL_OUTPUT_DIR}/")
print(f"Evaluation outputs saved to: {EVAL_OUTPUT_DIR}/")
print(f"\nOverall Test R²: {r2_overall:.6f}")
print("=" * 70)

