#!/usr/bin/env python

import numpy as np
import pandas as pd
from config import DATA_DIR, train_prop, dev_prop, test_prop
from utils import split_data, prepare_xy

# Load data
df = pd.read_csv(f'{DATA_DIR}/ref_data.tsv', sep='\t')

# Convert sequences to str
df['ref_sequence'] = df['ref_sequence'].astype(str)

# Split data into train/dev/test
train_df, dev_df, test_df = split_data(df,
                                       train_prop=train_prop,
                                       dev_prop=dev_prop,
                                       test_prop=test_prop)

# Prepare X and Y for each set
X_train, y_train = prepare_xy(train_df)
X_dev, y_dev = prepare_xy(dev_df)
X_test, y_test = prepare_xy(test_df)

# Save data
np.savez(f'{DATA_DIR}/preprocessed_data.npz',
         X_train=X_train, y_train=y_train,
         X_dev=X_dev, y_dev=y_dev,
         X_test=X_test, y_test=y_test)
