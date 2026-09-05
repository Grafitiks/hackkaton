import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=== STARTING EXPERIMENTAL GAP RESTORATION BREAKTHROUGH ===")

# 1. Загрузка данных
dft = pd.read_csv("data/train_dataset.csv")
dft['date_dt'] = pd.to_datetime(dft['date'])
dft['year'] = dft['date_dt'].dt.year
dft['doy'] = dft['date_dt'].dt.dayofyear
dft = dft.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

df1 = pd.read_csv(r"data/test_features.csv")
df1['date_dt'] = pd.to_datetime(df1['date'])
df1['year'] = df1['date_dt'].dt.year
df1['doy'] = df1['date_dt'].dt.dayofyear
df1 = df1.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

df_all = pd.concat([dft, df1], ignore_index=True)

# 2. Построение индекса пролетов спутников на уровне дат
date_sat_stats = df_all.groupby('date').agg(
    tot_s2=('s2_ndvi', lambda s: s.notna().sum()),
    tot_ls=('landsat_ndvi', lambda s: s.notna().sum()),
    tot_mod=('modis_ndvi', lambda s: s.notna().sum()),
    tot_obs=('primary_ndvi', lambda s: s.notna().sum()),
    mean_obs=('primary_ndvi', 'mean')
).reset_index()

date_crop_stats = df_all[df_all['primary_ndvi'].notna()].groupby(['crop_type', 'date']).agg(
    crop_tot_obs=('primary_ndvi', 'count'),
    crop_mean_obs=('primary_ndvi', 'mean'),
    crop_sum_obs=('primary_ndvi', 'sum')
).reset_index()

print("Precomputed date-level satellite pass and regional crop statistics.")

# 3. Расчет трансдуктивной климатологии по известным наблюдениям
known_all = df_all[df_all['primary_ndvi'].notna()].copy()
poly_clim = known_all.groupby(['anon_polygon_id', 'doy'])['primary_ndvi'].agg(['mean', 'std']).reset_index()
poly_clim.columns = ['anon_polygon_id', 'doy', 'poly_clim_mean', 'poly_clim_std']

crop_clim = known_all.groupby(['crop_type', 'doy'])['primary_ndvi'].agg(['mean', 'std']).reset_index()
crop_clim.columns = ['crop_type', 'doy', 'crop_clim_mean', 'crop_clim_std']

global_clim = known_all.groupby('doy')['primary_ndvi'].agg(['mean', 'std']).reset_index()
global_clim.columns = ['doy', 'global_clim_mean', 'global_clim_std']

# Сглаживание климатологии скользящим окном
for col in ['poly_clim_mean', 'poly_clim_std']:
    poly_clim[col] = poly_clim.groupby('anon_polygon_id')[col].transform(
        lambda s: s.rolling(11, center=True, min_periods=1).mean()
    )
for col in ['crop_clim_mean', 'crop_clim_std']:
    crop_clim[col] = crop_clim.groupby('crop_type')[col].transform(
        lambda s: s.rolling(11, center=True, min_periods=1).mean()
    )
for col in ['global_clim_mean', 'global_clim_std']:
    global_clim[col] = global_clim[col].rolling(11, center=True, min_periods=1).mean()

print("Climatology smoothed.")

# 4. Генерация реалистичного датасета пропусков из обучающих данных
np.random.seed(42)
train_known = dft[dft['primary_ndvi'].notna()].copy()
train_known_indices = train_known.index.values

mask_prob = 0.15
mask = np.random.rand(len(train_known_indices)) < mask_prob
gap_indices = train_known_indices[mask]
print(f"Masked {len(gap_indices)} points ({len(gap_indices)/len(train_known_indices):.1%}) in train dataset.")

def extract_rich_features(df, target_indices, is_train=True):
    df_work = df.copy()
    df_work.loc[target_indices, 'primary_ndvi'] = np.nan
    known = df_work[df_work['primary_ndvi'].notna()].copy()
    
    records = []
    for poly_id, grp in df_work.groupby('anon_polygon_id'):
        p_gaps = grp[grp.index.isin(target_indices)]
        if p_gaps.empty:
            continue
        p_known = known[known['anon_polygon_id'] == poly_id]
        if p_known.empty:
            continue
            
        p_known_dates = p_known['date_dt'].values
        p_known_vals = p_known['primary_ndvi'].values
        p_known_s2 = p_known['s2_ndvi'].values
        p_known_ls = p_known['landsat_ndvi'].values
        p_known_mod = p_known['modis_ndvi'].values
        
        for idx, row in p_gaps.iterrows():
            t = np.datetime64(row['date_dt'])
            
            prev_idx = np.where(p_known_dates < t)[0]
            next_idx = np.where(p_known_dates > t)[0]
            
            has_prev = len(prev_idx) > 0
            has_next = len(next_idx) > 0
            
            i_p1 = prev_idx[-1] if has_prev else None
            i_p2 = prev_idx[-2] if len(prev_idx) > 1 else i_p1
            i_n1 = next_idx[0] if has_next else None
            i_n2 = next_idx[1] if len(next_idx) > 1 else i_n1
            
            dt_p1 = (row['date_dt'] - pd.to_datetime(p_known_dates[i_p1])).days if has_prev else 999
            dt_p2 = (row['date_dt'] - pd.to_datetime(p_known_dates[i_p2])).days if i_p2 is not None else 999
            dt_n1 = (pd.to_datetime(p_known_dates[i_n1]) - row['date_dt']).days if has_next else 999
            dt_n2 = (pd.to_datetime(p_known_dates[i_n2]) - row['date_dt']).days if i_n2 is not None else 999
            
            y_p1 = p_known_vals[i_p1] if has_prev else np.nan
            y_p2 = p_known_vals[i_p2] if i_p2 is not None else np.nan
            y_n1 = p_known_vals[i_n1] if has_next else np.nan
            y_n2 = p_known_vals[i_n2] if i_n2 is not None else np.nan
            
            if has_prev and has_next:
                dt_total = dt_p1 + dt_n1
                y_linear = y_p1 + (y_n1 - y_p1) * (dt_p1 / dt_total)
                w_p = dt_n1 / dt_total
                w_n = dt_p1 / dt_total
            elif has_prev:
                dt_total = dt_p1
                y_linear = y_p1
                w_p = 1.0
                w_n = 0.0
            elif has_next:
                dt_total = dt_n1
                y_linear = y_n1
                w_p = 0.0
                w_n = 1.0
            else:
                dt_total = 999
                y_linear = np.nan
                w_p = 0.0
                w_n = 0.0
                
            src_p1 = 0 if (has_prev and not np.isnan(p_known_s2[i_p1])) else (1 if (has_prev and not np.isnan(p_known_ls[i_p1])) else 2)
            src_n1 = 0 if (has_next and not np.isnan(p_known_s2[i_n1])) else (1 if (has_next and not np.isnan(p_known_ls[i_n1])) else 2)
            
            slope_p = (y_p1 - y_p2) / max(1, (dt_p2 - dt_p1)) if has_prev and i_p2 != i_p1 else 0.0
            slope_n = (y_n2 - y_n1) / max(1, (dt_n2 - dt_n1)) if has_next and i_n2 != i_n1 else 0.0
            
            records.append({
                'index': idx,
                'anon_polygon_id': poly_id,
                'date': str(row['date']),
                'doy': int(row['doy']),
                'year': int(row['year']),
                'crop_type': str(row['crop_type']),
                'y_linear': y_linear,
                'y_p1': y_p1,
                'y_n1': y_n1,
                'y_p2': y_p2,
                'y_n2': y_n2,
                'dt_p1': dt_p1,
                'dt_n1': dt_n1,
                'dt_total': dt_total,
                'min_dt': min(dt_p1, dt_n1),
                'diff_pn': y_n1 - y_p1 if (has_prev and has_next) else 0.0,
                'w_p': w_p,
                'w_n': w_n,
                'slope_p': slope_p,
                'slope_n': slope_n,
                'src_p1': src_p1,
                'src_n1': src_n1,
                'era5_temp_c': row.get('era5_temp_c', np.nan),
                'era5_precip_mm': row.get('era5_precip_mm', np.nan),
            })
            
    res = pd.DataFrame(records)
    
    # Привязка календаря пролетов спутников
    res = res.merge(date_sat_stats, on='date', how='left')
    
    res['inf_s2'] = res['tot_s2'] > 0
    res['inf_ls'] = (res['tot_s2'] == 0) & (res['tot_ls'] > 0)
    res['inf_mod'] = (res['tot_s2'] == 0) & (res['tot_ls'] == 0) & (res['tot_mod'] > 0)
    res['inferred_sensor'] = np.where(res['inf_s2'], 0, np.where(res['inf_ls'], 1, 2))
    
    # Привязка региональной статистики по культурам
    res = res.merge(date_crop_stats, on=['crop_type', 'date'], how='left')
    res['crop_mean_obs'] = res['crop_mean_obs'].fillna(res['mean_obs'])
    
    # Привязка климатологической нормы
    res = res.merge(poly_clim, on=['anon_polygon_id', 'doy'], how='left')
    res = res.merge(crop_clim, on=['crop_type', 'doy'], how='left')
    res = res.merge(global_clim, on='doy', how='left')
    
    res['clim_mean'] = res['poly_clim_mean'].fillna(res['crop_clim_mean']).fillna(res['global_clim_mean']).fillna(0.35)
    res['clim_std'] = res['poly_clim_std'].fillna(res['crop_clim_std']).fillna(res['global_clim_std']).fillna(0.06)
    
    res['y_linear'] = res['y_linear'].fillna(res['clim_mean'])
    res['y_p1'] = res['y_p1'].fillna(res['clim_mean'])
    res['y_n1'] = res['y_n1'].fillna(res['clim_mean'])
    res['y_p2'] = res['y_p2'].fillna(res['y_p1'])
    res['y_n2'] = res['y_n2'].fillna(res['y_n1'])
    
    res['clim_diff'] = res['y_linear'] - res['clim_mean']
    res['crop_mean_diff'] = res['crop_mean_obs'] - res['clim_mean']
    
    res['sin_doy'] = np.sin(2 * np.pi * res['doy'] / 365.25)
    res['cos_doy'] = np.cos(2 * np.pi * res['doy'] / 365.25)
    
    bias_map = {0: 0.0, 1: 0.037, 2: 0.082}
    b_t = res['inferred_sensor'].map(bias_map)
    b_p = res['src_p1'].map(bias_map)
    b_n = res['src_n1'].map(bias_map)
    res['sensor_bias_shift'] = b_t - (res['w_p'] * b_p + res['w_n'] * b_n)
    
    return res

print("Extracting features for training synthetic gaps...")
X_train_df = extract_rich_features(dft, gap_indices, is_train=True)
y_true = dft.loc[X_train_df['index'], 'primary_ndvi'].values
y_linear = X_train_df['y_linear'].values
delta_true = y_true - y_linear

base_rmse = np.sqrt(mean_squared_error(y_true, y_linear))
base_gap = 30 * max(0, 1 - base_rmse / 0.10)
print(f"Training Gap Baseline Linear: RMSE = {base_rmse:.5f}, GapScore = {base_gap:.2f}")

feature_cols = [
    'doy', 'sin_doy', 'cos_doy', 'crop_type',
    'y_linear', 'y_p1', 'y_n1', 'y_p2', 'y_n2',
    'dt_p1', 'dt_n1', 'dt_total', 'min_dt', 'diff_pn',
    'w_p', 'w_n', 'slope_p', 'slope_n',
    'src_p1', 'src_n1', 'inferred_sensor', 'sensor_bias_shift',
    'tot_s2', 'tot_ls', 'tot_mod',
    'crop_mean_obs', 'crop_mean_diff',
    'clim_mean', 'clim_std', 'clim_diff'
]

X = X_train_df[feature_cols].copy()
X['crop_type'] = X['crop_type'].astype('category')
groups = X_train_df['anon_polygon_id'].values

print(f"\nRunning 5-Fold GroupKFold across {len(np.unique(groups))} polygons...")
gkf = GroupKFold(n_splits=5)
oof_pred = np.zeros(len(X))

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, delta_true, groups=groups)):
    X_tr, y_tr = X.iloc[trn_idx], delta_true[trn_idx]
    X_va, y_va = X.iloc[val_idx], delta_true[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 20,
        'seed': 42 + fold,
        'verbose': -1
    }
    
    m = lgb.train(params, train_data, num_boost_round=800, valid_sets=[train_data, val_data],
                  callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    
    pred_delta = m.predict(X_va)
    oof_pred[val_idx] = pred_delta
    
    f_pred = np.clip(y_linear[val_idx] + pred_delta, -0.2, 1.0)
    f_rmse = np.sqrt(mean_squared_error(y_true[val_idx], f_pred))
    f_gap = 30 * max(0, 1 - f_rmse / 0.10)
    print(f"Fold {fold+1} (Unseen Polygons) -> RMSE: {f_rmse:.5f}, GapScore: {f_gap:.2f}")

oof_final = np.clip(y_linear + oof_pred, -0.2, 1.0)
oof_rmse = np.sqrt(mean_squared_error(y_true, oof_final))
oof_gap = 30 * max(0, 1 - oof_rmse / 0.10)
print(f"\n==================================================")
print(f"GROUP-K-FOLD OOF RESULT (Strictly Unseen Polygons):")
print(f"Baseline Linear: RMSE = {base_rmse:.5f} | GapScore = {base_gap:.2f}")
print(f"Model OOF:       RMSE = {oof_rmse:.5f} | GapScore = {oof_gap:.2f}")
print(f"Improvement:     {base_rmse - oof_rmse:.5f} RMSE reduction (+{oof_gap - base_gap:.2f} score)")
print(f"==================================================")
