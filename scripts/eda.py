import pandas as pd
import numpy as np

print("--- Loading train_dataset.csv ---")
df_train = pd.read_csv("data/train_dataset.csv", nrows=1000)
print("Train shape (sample):", df_train.shape)
print("Train columns:", list(df_train.columns))
print("Train head:\n", df_train.head(2).to_dict(orient='records'))

print("\n--- Loading private_features.csv ---")
df_test = pd.read_csv("data/private_features.csv", nrows=1000)
print("Test shape (sample):", df_test.shape)
print("Test columns:", list(df_test.columns))
print("Test head:\n", df_test.head(2).to_dict(orient='records'))

print("\n--- Checking is_synthetic_gap counts in sample ---")
print("Train synthetic gaps:", df_train['is_synthetic_gap'].value_counts(dropna=False).to_dict() if 'is_synthetic_gap' in df_train.columns else "N/A")
print("Test synthetic gaps:", df_test['is_synthetic_gap'].value_counts(dropna=False).to_dict() if 'is_synthetic_gap' in df_test.columns else "N/A")
