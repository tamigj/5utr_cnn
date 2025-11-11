#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import post_tuning_params, NUM_EPOCH, DATA_DIR, OUTPUT_TUNING_DIR
from utils import build_cnn_model

# Load preprocessed data
data = np.load(f'{DATA_DIR}/preprocessed_data.npz')

X_train = data['X_train']
y_train = data['y_train']
X_dev = data['X_dev']
y_dev = data['y_dev']
X_test = data['X_test']
y_test = data['y_test']

# Extract parameters
batch_size = post_tuning_params['batch_size']
steps_per_epoch = len(X_train) // batch_size

# Build model
model_params = {k: v for k, v in post_tuning_params.items() if k != 'batch_size'}
model = build_cnn_model(**model_params, steps_per_epoch=steps_per_epoch)

# Print model summary
model.summary()

# Fit the model
history = model.fit(X_train, y_train,
                    validation_data=(X_dev, y_dev),
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
plt.savefig(f'{OUTPUT_TUNING_DIR}/post_tuning.png',
            dpi=300, bbox_inches='tight')
plt.close()
