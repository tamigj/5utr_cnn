#!/usr/bin/env python
"""Train final model with best hyperparameters after complete tuning.

This script:
1. Trains the model with early stopping using the best hyperparameters
2. Creates train vs dev loss plot
3. Saves the best model to output/tuned_model/
4. Evaluates on test set and outputs R2 score
5. Saves predictions for all test set reporters to data/model_predictions.tsv
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

from config import updated_params_4, NUM_EPOCH, DATA_DIR, OUTPUT_DIR, OUTPUT_TUNING_DIR
from utils import build_cnn_model, get_early_stopping_callback, split_data
from sklearn.model_selection import train_test_split

# Create output directory for saved model
MODEL_OUTPUT_DIR = f'{OUTPUT_DIR}/tuned_model'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Load preprocessed data
data = np.load(f'{DATA_DIR}/preprocessed_data.npz')

X_train = data['X_train']
Y_train = data['Y_train']
X_dev = data['X_dev']
Y_dev = data['Y_dev']
X_test = data['X_test']
Y_test = data['Y_test']

# Load original dataframe to get reporter (gene) information for test set
# Need to split with same random_state to match preprocessed data indices
df = pd.read_csv(f'{DATA_DIR}/ref_data.tsv', sep='\t')
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

# Extract parameters
batch_size = updated_params_4['batch_size']
steps_per_epoch = len(X_train) // batch_size

print("=" * 70)
print("Training final model with best hyperparameters")
print("=" * 70)
print("\nHyperparameters:")
for k, v in updated_params_4.items():
    print(f"  {k}: {v}")
print()

# Build model
model_params = {k: v for k, v in updated_params_4.items() if k != 'batch_size'}
model = build_cnn_model(**model_params, steps_per_epoch=steps_per_epoch)

# Print model summary
print("\nModel architecture:")
model.summary()

# Create early stopping callback
early_stopping = get_early_stopping_callback(patience=15, min_delta=0.001, verbose=1)

# Fit the model with early stopping
print("\nStarting training with early stopping...")
history = model.fit(X_train, Y_train,
                    validation_data=(X_dev, Y_dev),
                    epochs=NUM_EPOCH,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=1)

# Create train vs dev loss plot
print("\nCreating training plot...")
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Training and Validation Loss (Final Tuned Model)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Set y-axis limits based on data
max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
plt.ylim((0, max_loss * 1.1))

plot_path = f'{MODEL_OUTPUT_DIR}/training_curve.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved training curve to {plot_path}")

# Save the best model (early stopping already restored best weights)
model_path = f'{MODEL_OUTPUT_DIR}/best_model.h5'
model.save(model_path)
print(f"Saved best model to {model_path}")

# Evaluate on test set
print("\n" + "=" * 70)
print("Evaluating on test set")
print("=" * 70)

test_loss = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss[0]:.6f}")
print(f"Test MAE: {test_loss[1]:.6f}")

# Get predictions on test set
Y_pred = model.predict(X_test, verbose=0).flatten()

# Calculate R2 score
r2 = r2_score(Y_test, Y_pred)
print(f"Test R²: {r2:.6f}")

# Save evaluation metrics to file
eval_path = f'{OUTPUT_DIR}/best_models_evals.txt'
with open(eval_path, 'w') as f:
    f.write("Final Model Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test Loss (MSE): {test_loss[0]:.6f}\n")
    f.write(f"Test MAE: {test_loss[1]:.6f}\n")
    f.write(f"Test R²: {r2:.6f}\n")
print(f"Saved evaluation metrics to {eval_path}")

# Save predictions with reporter information
print("\nSaving predictions to data/model_predictions.tsv...")

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'gene': test_df['gene'].values,
    'true_translation': Y_test,
    'predicted_translation': Y_pred
})

# Add other columns from test_df if available
for col in test_df.columns:
    if col not in predictions_df.columns:
        predictions_df[col] = test_df[col].values

# Save to TSV
predictions_path = f'{DATA_DIR}/model_predictions.tsv'
predictions_df.to_csv(predictions_path, sep='\t', index=False)
print(f"Saved predictions for {len(predictions_df)} test set reporters to {predictions_path}")

print("\n" + "=" * 70)
print("Training and evaluation complete!")
print("=" * 70)

