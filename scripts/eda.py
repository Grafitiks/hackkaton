# Предварительный экспресс-анализ структуры обучающего и тестового датасетов
import pandas as pd
import numpy as np

# 1. Загрузка выборки обучающих данных
print("--- Loading train_dataset.csv ---")
df_train = pd.read_csv("data/train_dataset.csv", nrows=1000)
print("Train shape (sample):", df_train.shape)
print("Train columns:", list(df_train.columns))
print("Train head:\n", df_train.head(2).to_dict(orient='records'))

# 2. Загрузка выборки тестовых данных
print("\n--- Loading test dataset ---")
df_test = pd.read_csv("data/test_features (1).csv", nrows=1000) if pd.io.common.file_exists("data/test_features (1).csv") else pd.read_csv("data/train_dataset.csv", nrows=1000)
print("Test shape (sample):", df_test.shape)
print("Test columns:", list(df_test.columns))
print("Test head:\n", df_test.head(2).to_dict(orient='records'))

# 3. Анализ наличия синтетических пропусков в выборках
print("\n--- Checking is_synthetic_gap counts in sample ---")
print("Train synthetic gaps:", df_train['is_synthetic_gap'].value_counts(dropna=False).to_dict() if 'is_synthetic_gap' in df_train.columns else "N/A")
print("Test synthetic gaps:", df_test['is_synthetic_gap'].value_counts(dropna=False).to_dict() if 'is_synthetic_gap' in df_test.columns else "N/A")
