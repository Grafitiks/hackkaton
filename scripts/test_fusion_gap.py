import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=== TESTING HARMONIZATION + SPATIAL COVARIANCE FUSION ===")

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

# 2. Полиномы гармонизации сенсоров к уровню Sentinel-2
# Преобразование Landsat к Sentinel-2: y_s2 = 0.0177 * y**2 + 1.002 * y - 0.0409
# Преобразование MODIS к Sentinel-2:   y_s2 = 0.2988 * y**2 + 0.7471 * y - 0.0412
def to_s2(val, sensor):
    if np.isnan(val): return np.nan
    if sensor == 0: return val
    elif sensor == 1: return 0.0177 * val**2 + 1.002 * val - 0.0409
    else: return 0.2988 * val**2 + 0.7471 * val - 0.0412

# 3. Глобальный календарь пролетов спутников по дням
date_sat_stats = df_all.groupby('date').agg(
    tot_s2=('s2_ndvi', lambda s: s.notna().sum()),
    tot_ls=('landsat_ndvi', lambda s: s.notna().sum()),
    tot_mod=('modis_ndvi', lambda s: s.notna().sum()),
    tot_obs=('primary_ndvi', lambda s: s.notna().sum()),
    mean_obs=('primary_ndvi', 'mean')
).reset_index()

date_sat_stats['inferred_sensor'] = np.where(
    date_sat_stats['tot_s2'] > 0, 0,
    np.where(date_sat_stats['tot_ls'] > 0, 1, 2)
)

# 4. Трансдуктивная климатология по всем полигонам
known_all = df_all[df_all['primary_ndvi'].notna()].copy()
poly_clim = known_all.groupby(['anon_polygon_id', 'doy'])['primary_ndvi'].agg(['mean', 'std']).reset_index()
poly_clim.columns = ['anon_polygon_id', 'doy', 'poly_clim_mean', 'poly_clim_std']

crop_clim = known_all.groupby(['crop_type', 'doy'])['primary_ndvi'].agg(['mean', 'std']).reset_index()
crop_clim.columns = ['crop_type', 'doy', 'crop_clim_mean', 'crop_clim_std']

global_clim = known_all.groupby('doy')['primary_ndvi'].agg(['mean', 'std']).reset_index()
global_clim.columns = ['doy', 'global_clim_mean', 'global_clim_std']

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

# 5. Таблица пространственных аномалий в тот же календарный день
known_all = known_all.merge(crop_clim[['crop_type', 'doy', 'crop_clim_mean']], on=['crop_type', 'doy'], how='left')
known_all['crop_anom'] = known_all['primary_ndvi'] - known_all['crop_clim_mean']

date_crop_spatial = known_all.groupby(['crop_type', 'date'])['crop_anom'].agg(['sum', 'count']).reset_index()

# 6. Функция извлечения признаков с гармонизацией к S2 и пространственными аномалиями
def extract_fusion_features(df, target_indices=None):
    df_work = df.copy()
    if target_indices is not None:
        df_work.loc[target_indices, 'primary_ndvi'] = np.nan
        
    known = df_work[df_work['primary_ndvi'].notna()].copy()
    
    # Фильтрация облачных теней
    for pid, grp in df_work.groupby('anon_polygon_id'):
        sub = grp[grp['primary_ndvi'].notna()]
        if len(sub) < 3: continue
        vals = sub['primary_ndvi'].values
        dates = sub['date_dt'].values
        idx_arr = sub.index.values
        for i in range(1, len(vals)-1):
            dt1 = (dates[i] - dates[i-1]).astype('timedelta64[D]').astype(int)
            dt2 = (dates[i+1] - dates[i]).astype('timedelta64[D]').astype(int)
            if dt1 <= 12 and dt2 <= 12 and vals[i] < vals[i-1] - 0.25 and vals[i] < vals[i+1] - 0.25:
                df_work.loc[idx_arr[i], 'primary_ndvi'] = np.nan
            if vals[i] < -0.05 or vals[i] > 1.05:
                df_work.loc[idx_arr[i], 'primary_ndvi'] = np.nan
                
    known_clean = df_work[df_work['primary_ndvi'].notna()].copy()
    
    samples = []
    for poly_id, grp in df_work.groupby('anon_polygon_id'):
        p_known = known_clean[known_clean['anon_polygon_id'] == poly_id]
        if p_known.empty:
            continue
            
        p_dates = p_known['date_dt'].values
        p_vals = p_known['primary_ndvi'].values
        p_s2 = p_known['s2_ndvi'].values if 's2_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        p_ls = p_known['landsat_ndvi'].values if 'landsat_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        
        if target_indices is not None:
            eval_rows = grp[grp.index.isin(target_indices)]
        else:
            eval_rows = p_known.iloc[1:-1]
            
        for idx, row in eval_rows.iterrows():
            t = np.datetime64(row['date_dt'])
            prev_idx = np.where(p_dates < t)[0]
            next_idx = np.where(p_dates > t)[0]
            
            has_p = len(prev_idx) > 0
            has_n = len(next_idx) > 0
            if not has_p and not has_n: continue
            
            i_p1 = prev_idx[-1] if has_p else None
            i_p2 = prev_idx[-2] if len(prev_idx) > 1 else i_p1
            i_n1 = next_idx[0] if has_n else None
            i_n2 = next_idx[1] if len(next_idx) > 1 else i_n1
            
            dt_p1 = (row['date_dt'] - pd.to_datetime(p_dates[i_p1])).days if has_p else 999
            dt_p2 = (row['date_dt'] - pd.to_datetime(p_dates[i_p2])).days if i_p2 is not None else 999
            dt_n1 = (pd.to_datetime(p_dates[i_n1]) - row['date_dt']).days if has_n else 999
            dt_n2 = (pd.to_datetime(p_dates[i_n2]) - row['date_dt']).days if i_n2 is not None else 999
            
            yp1 = p_vals[i_p1] if has_p else np.nan
            yp2 = p_vals[i_p2] if i_p2 is not None else np.nan
            yn1 = p_vals[i_n1] if has_n else np.nan
            yn2 = p_vals[i_n2] if i_n2 is not None else np.nan
            
            sp1 = 0 if (has_p and not np.isnan(p_s2[i_p1])) else (1 if (has_p and not np.isnan(p_ls[i_p1])) else 2)
            sn1 = 0 if (has_n and not np.isnan(p_s2[i_n1])) else (1 if (has_n and not np.isnan(p_ls[i_n1])) else 2)
            sp2 = 0 if (i_p2 is not None and not np.isnan(p_s2[i_p2])) else (1 if (i_p2 is not None and not np.isnan(p_ls[i_p2])) else 2)
            sn2 = 0 if (i_n2 is not None and not np.isnan(p_s2[i_n2])) else (1 if (i_n2 is not None and not np.isnan(p_ls[i_n2])) else 2)
            
            # Значения, гармонизированные к уровню Sentinel-2
            yp1_s2 = to_s2(yp1, sp1)
            yn1_s2 = to_s2(yn1, sn1)
            yp2_s2 = to_s2(yp2, sp2)
            yn2_s2 = to_s2(yn2, sn2)
            
            if has_p and has_n:
                dt_tot = dt_p1 + dt_n1
                y_lin_s2 = yp1_s2 + (yn1_s2 - yp1_s2) * (dt_p1 / dt_tot)
                y_lin_raw = yp1 + (yn1 - yp1) * (dt_p1 / dt_tot)
                wp = dt_n1 / dt_tot
                wn = dt_p1 / dt_tot
            elif has_p:
                dt_tot = dt_p1
                y_lin_s2 = yp1_s2
                y_lin_raw = yp1
                wp, wn = 1.0, 0.0
            else:
                dt_tot = dt_n1
                y_lin_s2 = yn1_s2
                y_lin_raw = yn1
                wp, wn = 0.0, 1.0
                
            slope_p = (yp1_s2 - yp2_s2) / max(1, (dt_p2 - dt_p1)) if has_p and i_p2 != i_p1 else 0.0
            slope_n = (yn2_s2 - yn1_s2) / max(1, (dt_n2 - dt_n1)) if has_n and i_n2 != i_n1 else 0.0
            
            samples.append({
                'index': idx,
                'anon_polygon_id': poly_id,
                'date': str(row['date']),
                'doy': int(row['doy']),
                'year': int(row['year']),
                'crop_type': str(row['crop_type']),
                'y_lin_s2': y_lin_s2,
                'y_lin_raw': y_lin_raw,
                'yp1_s2': yp1_s2,
                'yn1_s2': yn1_s2,
                'yp2_s2': yp2_s2,
                'yn2_s2': yn2_s2,
                'dt_p1': dt_p1,
                'dt_n1': dt_n1,
                'dt_tot': dt_tot,
                'min_dt': min(dt_p1, dt_n1),
                'dt_diff': abs(dt_p1 - dt_n1),
                'diff_pn': yn1_s2 - yp1_s2 if (has_p and has_n) else 0.0,
                'wp': wp,
                'wn': wn,
                'slope_p': slope_p,
                'slope_n': slope_n,
                'sp1': sp1,
                'sn1': sn1,
                'era5_temp_c': row.get('era5_temp_c', np.nan),
                'era5_precip_mm': row.get('era5_precip_mm', np.nan),
            })
            
    res = pd.DataFrame(samples)
    
    # Привязка календаря пролетов спутников
    res = res.merge(date_sat_stats[['date', 'tot_s2', 'tot_ls', 'tot_mod', 'inferred_sensor', 'mean_obs']], on='date', how='left')
    res['tot_s2'] = res['tot_s2'].fillna(0)
    res['tot_ls'] = res['tot_ls'].fillna(0)
    res['tot_mod'] = res['tot_mod'].fillna(0)
    res['inferred_sensor'] = res['inferred_sensor'].fillna(1).astype(int)
    
    # Привязка пространственной аномалии по культуре
    res = res.merge(date_crop_spatial[['crop_type', 'date', 'sum', 'count']], on=['crop_type', 'date'], how='left')
    res['crop_anom_mean'] = np.where(res['count'] > 0, res['sum'] / res['count'], 0.0)
    
    # Привязка климатологической нормы
    res = res.merge(poly_clim, on=['anon_polygon_id', 'doy'], how='left')
    res = res.merge(crop_clim, on=['crop_type', 'doy'], how='left')
    res = res.merge(global_clim, on='doy', how='left')
    
    res['clim_mean'] = res['poly_clim_mean'].fillna(res['crop_clim_mean']).fillna(res['global_clim_mean']).fillna(0.35)
    res['clim_std'] = res['poly_clim_std'].fillna(res['crop_clim_std']).fillna(res['global_clim_std']).fillna(0.06)
    
    res['y_lin_s2'] = res['y_lin_s2'].fillna(res['clim_mean'])
    res['y_lin_raw'] = res['y_lin_raw'].fillna(res['clim_mean'])
    res['yp1_s2'] = res['yp1_s2'].fillna(res['clim_mean'])
    res['yn1_s2'] = res['yn1_s2'].fillna(res['clim_mean'])
    
    # Пространственно-направляемый прогноз
    res['y_spatial_guide'] = res['clim_mean'] + res['crop_anom_mean']
    res['spatial_diff'] = res['y_spatial_guide'] - res['y_lin_s2']
    
    # Кривизна
    res['clim_diff'] = res['y_lin_s2'] - res['clim_mean']
    res['curv'] = res['y_lin_s2'] - 0.5 * (res['yp1_s2'] + res['yn1_s2'])
    
    # Смещение сенсоров
    bias_map = {0: 0.0, 1: 0.037, 2: 0.082}
    res['sensor_bias_shift'] = res['inferred_sensor'].map(bias_map) - (res['wp'] * res['sp1'].map(bias_map) + res['wn'] * res['sn1'].map(bias_map))
    
    # Циклическое кодирование времени
    res['sin_doy'] = np.sin(2 * np.pi * res['doy'] / 365.25)
    res['cos_doy'] = np.cos(2 * np.pi * res['doy'] / 365.25)
    
    return res

print("Extracting rich fusion features from training dataset...")
train_df = extract_fusion_features(dft)
if len(train_df) > 22000:
    np.random.seed(42)
    sub_idx = np.random.choice(train_df.index.values, size=22000, replace=False)
    train_df = train_df.loc[sub_idx].reset_index(drop=True)

y_train_true = dft.loc[train_df['index'], 'primary_ndvi'].values
y_train_lin_s2 = train_df['y_lin_s2'].values
delta_train_true = y_train_true - y_train_lin_s2
delta_clipped = np.clip(delta_train_true, -0.25, 0.25)

feats = [
    'doy', 'sin_doy', 'cos_doy', 'crop_type',
    'y_lin_s2', 'yp1_s2', 'yn1_s2', 'yp2_s2', 'yn2_s2',
    'dt_p1', 'dt_n1', 'dt_tot', 'min_dt', 'dt_diff', 'diff_pn',
    'wp', 'wn', 'slope_p', 'slope_n',
    'sp1', 'sn1', 'inferred_sensor', 'sensor_bias_shift',
    'tot_s2', 'tot_ls', 'tot_mod',
    'crop_anom_mean', 'clim_mean', 'clim_std', 'clim_diff', 'curv',
    'y_spatial_guide', 'spatial_diff',
    'era5_temp_c', 'era5_precip_mm'
]

X_tr = train_df[feats].copy()
X_tr['crop_type'] = X_tr['crop_type'].astype('category')
groups = train_df['anon_polygon_id'].values

X_tr_cat = X_tr.copy()
X_tr_cat['crop_type'] = X_tr_cat['crop_type'].astype(str)

print("\n--- Training 5-Fold GroupKFold Ensemble ---")
gkf = GroupKFold(n_splits=5)
lgb_models = []
cat_models = []
oof_pred_lgb = np.zeros(len(X_tr))
oof_pred_cat = np.zeros(len(X_tr))

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tr, delta_clipped, groups=groups)):
    X_f_tr, y_f_tr = X_tr.iloc[trn_idx], delta_clipped[trn_idx]
    X_f_va, y_f_va = X_tr.iloc[val_idx], delta_clipped[val_idx]
    
    # LightGBM с функцией потерь Huber
    train_data = lgb.Dataset(X_f_tr, label=y_f_tr)
    val_data = lgb.Dataset(X_f_va, label=y_f_va, reference=train_data)
    params = {
        'objective': 'huber', 'alpha': 0.85, 'metric': 'rmse',
        'learning_rate': 0.03, 'num_leaves': 35, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1, 'min_data_in_leaf': 25,
        'seed': 42 + fold, 'verbose': -1
    }
    m_lgb = lgb.train(params, train_data, num_boost_round=800, valid_sets=[train_data, val_data],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    lgb_models.append(m_lgb)
    oof_pred_lgb[val_idx] = m_lgb.predict(X_f_va)
    
    # Модель CatBoost
    cb_tr = X_tr_cat.iloc[trn_idx]
    cb_va = X_tr_cat.iloc[val_idx]
    m_cb = CatBoostRegressor(iterations=600, learning_rate=0.035, depth=6, loss_function='RMSE',
                             cat_features=['crop_type'], random_seed=42+fold, verbose=0)
    m_cb.fit(cb_tr, y_f_tr, eval_set=(cb_va, y_f_va), early_stopping_rounds=40)
    cat_models.append(m_cb)
    oof_pred_cat[val_idx] = m_cb.predict(cb_va)
    
    fold_p = np.clip(y_train_lin_s2[val_idx] + 0.6 * oof_pred_lgb[val_idx] + 0.4 * oof_pred_cat[val_idx], -0.05, 0.98)
    f_rmse = np.sqrt(mean_squared_error(y_train_true[val_idx], fold_p))
    print(f"Fold {fold+1} OOF (Unseen Polygons): RMSE = {f_rmse:.5f}, GapScore = {30*max(0, 1-f_rmse/0.1):.2f}")

oof_p = np.clip(y_train_lin_s2 + 0.6 * oof_pred_lgb + 0.4 * oof_pred_cat, -0.05, 0.98)
oof_rmse = np.sqrt(mean_squared_error(y_train_true, oof_p))
print(f"\nOVERALL GROUP-K-FOLD OOF: RMSE = {oof_rmse:.5f}, GapScore = {30*max(0, 1-oof_rmse/0.1):.2f} / 30")

# Оценка на test_features (1)
known_test = df1[df1['primary_ndvi'].notna()].copy()
np.random.seed(42)
val_gap_idx = np.random.choice(known_test.index.values, size=min(2500, len(known_test)), replace=False)

test_feat_df = extract_fusion_features(df1, target_indices=val_gap_idx)
test_feat_df = test_feat_df.sort_values('index').reset_index(drop=True)
y_test_true = df1.loc[test_feat_df['index'], 'primary_ndvi'].values
y_test_lin_s2 = test_feat_df['y_lin_s2'].values

X_te = test_feat_df[feats].copy()
X_te['crop_type'] = X_te['crop_type'].astype('category')
X_te_cat = X_te.copy()
X_te_cat['crop_type'] = X_te_cat['crop_type'].astype(str)

p_lgb = np.zeros(len(X_te))
for m in lgb_models: p_lgb += m.predict(X_te) / len(lgb_models)
p_cb = np.zeros(len(X_te))
for m in cat_models: p_cb += m.predict(X_te_cat) / len(cat_models)

final_test_p = np.clip(y_test_lin_s2 + 0.6 * p_lgb + 0.4 * p_cb, -0.05, 0.98)
test_rmse = np.sqrt(mean_squared_error(y_test_true, final_test_p))
test_mae = mean_absolute_error(y_test_true, final_test_p)
print(f"\n=======================================================")
print(f"TEST_FEATURES (1) EVALUATION:")
print(f"RMSE = {test_rmse:.5f}, MAE = {test_mae:.5f}, GapScore = {30*max(0, 1-test_rmse/0.1):.2f} / 30")
print(f"=======================================================")
