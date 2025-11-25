#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from config import updated_params_1, NUM_EPOCH, DATA_DIR, OUTPUT_TUNING_DIR
from utils import build_cnn_model

# Load preprocessed data
data = np.load(f'{DATA_DIR}/preprocessed_data.npz')

X_train = data['X_train']
Y_train = data['Y_train']
X_dev = data['X_dev']
Y_dev = data['Y_dev']
X_test = data['X_test']
Y_test = data['Y_test']

# Extract parameters
batch_size = updated_params_1['batch_size']
steps_per_epoch = len(X_train) // batch_size

# Get output suffix from command line argument
if len(sys.argv) < 2:
    output_suffix = 'tuning'  # default
    print("Warning: No output suffix provided, using 'tuning' as default")
    print("Usage: python post_tuning_model.py <suffix>")
    print("Example: python post_tuning_model.py round_1")
else:
    output_suffix = sys.argv[1]

# Build model
model_params = {k: v for k, v in updated_params_1.items() if k != 'batch_size'}
model = build_cnn_model(**model_params, steps_per_epoch=steps_per_epoch)

# Print model summary
model.summary()

# Fit the model
history = model.fit(X_train, Y_train,
                    validation_data=(X_dev, Y_dev),
                    epochs=NUM_EPOCH,
                    batch_size=batch_size,
                    verbose=1)

# Visualize training vs dev loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim((0,3))
plt.title('Training and Validation Loss (post-tuning parameters)')
plt.legend()
plt.savefig(f'{OUTPUT_TUNING_DIR}/post_{output_suffix}.png',
            dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved plot to {OUTPUT_TUNING_DIR}/post_{output_suffix}.png")
