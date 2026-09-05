import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("=== TESTING ANOMALY INTERPOLATION & CLIMATOLOGY ===")

df_train = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
df_train['date_dt'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date_dt'].dt.year
df_train = df_train.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

# 1. Построение климатологии по дням года (DOY) для каждого полигона по известным точкам
# Использование скользящего среднего по окну DOY (+/- 5 дней) для подавления шума
clim_table = df_train[df_train['primary_ndvi'].notna()].groupby(['anon_polygon_id', 'doy'])['primary_ndvi'].agg(['mean', 'std', 'count']).reset_index()

# Линейная интерполяция пропущенных дней DOY внутри полигона
full_grid = []
for poly in df_train['anon_polygon_id'].unique():
    for d in range(91, 304):
        full_grid.append({'anon_polygon_id': poly, 'doy': d})
full_df = pd.DataFrame(full_grid)
full_df = full_df.merge(clim_table, on=['anon_polygon_id', 'doy'], how='left')

# Сглаживание скользящим средним по полигону
smoothed_clim = []
for poly, g in full_df.groupby('anon_polygon_id'):
    g = g.sort_values('doy').copy()
    g['clim_mean_smooth'] = g['mean'].interpolate(method='linear', limit_direction='both').rolling(11, center=True, min_periods=1).mean()
    g['clim_std_smooth'] = g['std'].interpolate(method='linear', limit_direction='both').rolling(11, center=True, min_periods=1).mean().fillna(0.05)
    smoothed_clim.append(g[['anon_polygon_id', 'doy', 'clim_mean_smooth', 'clim_std_smooth']])
clim_smooth_df = pd.concat(smoothed_clim, ignore_index=True)

# Объединение обратно с df_train
df_train = df_train.merge(clim_smooth_df, on=['anon_polygon_id', 'doy'], how='left')

# Сравнение рассчитанной климатологии с исходной ndvi_climatology_mean
comp = df_train[df_train['ndvi_climatology_mean'].notna()]
corr = np.corrcoef(comp['clim_mean_smooth'], comp['ndvi_climatology_mean'])[0, 1]
print(f"Correlation between calculated smooth clim and original ndvi_climatology_mean: {corr:.4f}")

# Валидационный тест методов
np.random.seed(42)
known_idx = df_train[df_train['primary_ndvi'].notna()].index.values
val_mask = np.zeros(len(df_train), dtype=bool)
sampled_val_idx = np.random.choice(known_idx, size=4500, replace=False)
val_mask[sampled_val_idx] = True

y_true = df_train.loc[val_mask, 'primary_ndvi'].values

df_sim = df_train[['anon_polygon_id', 'date_dt', 'year', 'doy', 'primary_ndvi', 'clim_mean_smooth']].copy()
df_sim.loc[val_mask, 'primary_ndvi'] = np.nan

# Метод A: Базовая линейная интерполяция
df_sim['raw_linear'] = df_sim.groupby(['anon_polygon_id', 'year'])['primary_ndvi'].transform(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
df_sim['raw_linear'] = df_sim['raw_linear'].fillna(
    df_sim.groupby('anon_polygon_id')['primary_ndvi'].transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
)

rmse_raw = np.sqrt(np.mean((y_true - df_sim.loc[val_mask, 'raw_linear'].values) ** 2))
print(f"Raw Linear: RMSE = {rmse_raw:.5f} | GapScore = {round(float(30*max(0, 1-rmse_raw/0.1)), 2)}")

# Метод B: Линейная интерполяция аномалий относительно нормы
# Аномалия = primary_ndvi - clim_mean_smooth
df_sim['anomaly'] = df_sim['primary_ndvi'] - df_sim['clim_mean_smooth']
df_sim['anomaly_linear'] = df_sim.groupby(['anon_polygon_id', 'year'])['anomaly'].transform(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
df_sim['anomaly_linear'] = df_sim['anomaly_linear'].fillna(
    df_sim.groupby('anon_polygon_id')['anomaly'].transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
).fillna(0.0)

df_sim['pred_anomaly_interp'] = df_sim['clim_mean_smooth'] + df_sim['anomaly_linear']
rmse_anom = np.sqrt(np.mean((y_true - df_sim.loc[val_mask, 'pred_anomaly_interp'].values) ** 2))
print(f"Anomaly Linear: RMSE = {rmse_anom:.5f} | GapScore = {round(float(30*max(0, 1-rmse_anom/0.1)), 2)}")
