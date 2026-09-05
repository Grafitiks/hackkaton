import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("=== GAP ANALYSIS ===")
df_test = pd.read_csv("data/private_features.csv", encoding='utf-8')
gaps = df_test[df_test['is_synthetic_gap'] == True].copy()
print(f"Total synthetic gaps in test: {len(gaps)}")

# Распределение наблюдений по годам
df_test['date_dt'] = pd.to_datetime(df_test['date'])
gaps['date_dt'] = pd.to_datetime(gaps['date'])
gaps['year_extracted'] = gaps['date_dt'].dt.year

print("\nSynthetic gaps by year:")
print(gaps['year_extracted'].value_counts().sort_index().to_dict())

print("\nSynthetic gaps by polygon count (min, max, mean):")
print(gaps.groupby('anon_polygon_id').size().describe())

# Анализ распределения по обучающим и тестовым полигонам
train_polys = set(pd.read_csv("data/train_dataset.csv", usecols=['anon_polygon_id'])['anon_polygon_id'].unique())
test_gaps_polys = set(gaps['anon_polygon_id'].unique())
print(f"Polygons with synthetic gaps: {len(test_gaps_polys)}")
print(f"Gaps in known (train) polygons: {len(test_gaps_polys.intersection(train_polys))}")
print(f"Gaps in new polygons: {len(test_gaps_polys - train_polys)}")

# Анализ распределения длины пропусков (последовательные дни)
gaps_by_poly = []
for poly_id, group in df_test.groupby('anon_polygon_id'):
    group = group.sort_values('date').reset_index(drop=True)
    # Поиск последовательных синтетических пропусков
    s = group['is_synthetic_gap']
    gap_blocks = (s != s.shift()).cumsum()[s]
    if not gap_blocks.empty:
        lengths = gap_blocks.value_counts().values
        gaps_by_poly.extend(lengths)

gap_lengths = pd.Series(gaps_by_poly)
print("\nConsecutive synthetic gap lengths distribution:")
print(gap_lengths.value_counts().sort_index().to_dict())
print("Summary statistics of gap lengths:")
print(gap_lengths.describe())

# Доля известных наблюдений primary_ndvi в 2025 году по сравнению с прошлыми годами
print("\nTest primary_ndvi non-null count by year:")
df_test['year_calc'] = df_test['date_dt'].dt.year
print(df_test[df_test['primary_ndvi'].notna()].groupby('year_calc').size().to_dict())
