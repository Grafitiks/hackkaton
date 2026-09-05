import pandas as pd
import numpy as np

print("=== DEEP EDA ===")
df_train = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
df_test = pd.read_csv("data/private_features.csv", encoding='utf-8')

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

print(f"Train unique polygons: {df_train['anon_polygon_id'].nunique()}")
print(f"Test unique polygons: {df_test['anon_polygon_id'].nunique()}")

# Overlap of polygons between train and test
train_polys = set(df_train['anon_polygon_id'].unique())
test_polys = set(df_test['anon_polygon_id'].unique())
print(f"Polygons in both train and test: {len(train_polys.intersection(test_polys))}")
print(f"Polygons only in test: {len(test_polys - train_polys)}")
print(f"Polygons only in train: {len(train_polys - test_polys)}")

print("\n--- Dates ---")
print("Train date range:", df_train['date'].min(), "to", df_train['date'].max())
print("Test date range:", df_test['date'].min(), "to", df_test['date'].max())

print("\n--- Test is_synthetic_gap distribution ---")
print(df_test['is_synthetic_gap'].value_counts(dropna=False))

print("\n--- Test primary_ndvi nulls for is_synthetic_gap == True vs False ---")
print("When is_synthetic_gap == True, primary_ndvi null count:", df_test[df_test['is_synthetic_gap'] == True]['primary_ndvi'].isna().sum(), "total:", (df_test['is_synthetic_gap'] == True).sum())
print("When is_synthetic_gap == False, primary_ndvi non-null count:", df_test[df_test['is_synthetic_gap'] == False]['primary_ndvi'].notna().sum(), "null count:", df_test[df_test['is_synthetic_gap'] == False]['primary_ndvi'].isna().sum())

print("\n--- Train primary_ndvi stats ---")
print("Train primary_ndvi non-null count:", df_train['primary_ndvi'].notna().sum(), "null count:", df_train['primary_ndvi'].isna().sum())
print(df_train['primary_ndvi'].describe())

print("\n--- Train status distribution ---")
print(df_train['status'].value_counts(dropna=False))

print("\n--- Train ndvi_zscore stats ---")
print(df_train['ndvi_zscore'].describe())

print("\n--- Crop types ---")
print("Train crop types:", df_train['crop_type'].value_counts(dropna=False).head(10).to_dict())
print("Test crop types:", df_test['crop_type'].value_counts(dropna=False).head(10).to_dict())

print("\n--- Missingness per column in Train ---")
print(df_train.isna().mean().round(4).to_dict())

print("\n--- Missingness per column in Test (when is_synthetic_gap == True) ---")
print(df_test[df_test['is_synthetic_gap'] == True].isna().mean().round(4).to_dict())

print("\n--- Missingness per column in Test (when is_synthetic_gap == False) ---")
print(df_test[df_test['is_synthetic_gap'] == False].isna().mean().round(4).to_dict())
