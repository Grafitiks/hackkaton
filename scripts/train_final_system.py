import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

print("=== STARTING FINAL HIGH-PERFORMANCE GAP RESTORATION SYSTEM ===")

# 1. Загрузка обучающих и валидационных данных
dft = pd.read_csv("data/train_dataset.csv")
dft['date_dt'] = pd.to_datetime(dft['date'])
dft['year'] = dft['date_dt'].dt.year
dft['doy'] = dft['date_dt'].dt.dayofyear
dft = dft.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

df_priv = pd.read_csv("data/private_features.csv")
df_priv['date_dt'] = pd.to_datetime(df_priv['date'])
df_priv['year'] = df_priv['date_dt'].dt.year
df_priv['doy'] = df_priv['date_dt'].dt.dayofyear
df_priv = df_priv.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

df_all = pd.concat([dft, df_priv], ignore_index=True)

# 2. Обогащение метеоданными ERA5
def enrich_weather(df):
    df = df.copy()
    if 'era5_temp_c' in df.columns:
        df['temp_interp'] = df.groupby('anon_polygon_id')['era5_temp_c'].transform(
            lambda s: s.interpolate(method='linear', limit_direction='both')
        ).fillna(15.0)
    else:
        df['temp_interp'] = 15.0
        
    if 'era5_precip_mm' in df.columns:
        df['precip_interp'] = df.groupby('anon_polygon_id')['era5_precip_mm'].transform(
            lambda s: s.fillna(0.0)
        ).fillna(0.0)
    else:
        df['precip_interp'] = 0.0
        
    df['precip_7'] = df.groupby('anon_polygon_id')['precip_interp'].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df['precip_14'] = df.groupby('anon_polygon_id')['precip_interp'].transform(
        lambda s: s.rolling(14, min_periods=1).sum()
    )
    df['temp_7'] = df.groupby('anon_polygon_id')['temp_interp'].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return df

dft = enrich_weather(dft)
df_priv = enrich_weather(df_priv)
df_all = enrich_weather(df_all)

# 3. Календарь глобальных пролетов спутников по дням
date_sat_stats = df_all.groupby('date').agg(
    tot_s2=('s2_ndvi', lambda s: s.notna().sum()),
    tot_ls=('landsat_ndvi', lambda s: s.notna().sum()),
    tot_mod=('modis_ndvi', lambda s: s.notna().sum()),
    tot_obs=('primary_ndvi', lambda s: s.notna().sum()),
    mean_obs=('primary_ndvi', 'mean')
).reset_index()

date_sat_stats['has_s2'] = (date_sat_stats['tot_s2'] > 0).astype(int)
date_sat_stats['has_ls'] = ((date_sat_stats['tot_s2'] == 0) & (date_sat_stats['tot_ls'] > 0)).astype(int)
date_sat_stats['has_mod'] = ((date_sat_stats['tot_s2'] == 0) & (date_sat_stats['tot_ls'] == 0) & (date_sat_stats['tot_mod'] > 0)).astype(int)
date_sat_stats['inferred_sensor'] = np.where(
    date_sat_stats['has_s2'] == 1, 0,
    np.where(date_sat_stats['has_ls'] == 1, 1, 2)
)

# 4. Региональная динамика по культурам
known_all = df_all[df_all['primary_ndvi'].notna()].copy()
date_crop_stats = known_all.groupby(['crop_type', 'date']).agg(
    crop_tot_obs=('primary_ndvi', 'count'),
    crop_mean_obs=('primary_ndvi', 'mean'),
    crop_std_obs=('primary_ndvi', 'std')
).reset_index()
date_crop_stats['crop_std_obs'] = date_crop_stats['crop_std_obs'].fillna(0.0)

# 5. Расчет климатологической нормы NDVI
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

os.makedirs("artifacts/models", exist_ok=True)
with open("artifacts/models/climatology.pkl", "wb") as f:
    pickle.dump({'poly_clim': poly_clim, 'crop_clim': crop_clim, 'global_clim': global_clim}, f)
print("Climatology saved.")

# 6. Извлечение признаков с фильтрацией теней и калибровкой сенсоров
def extract_samples(df, target_indices=None):
    df_clean = df.copy()
    if target_indices is not None:
        df_clean.loc[target_indices, 'primary_ndvi'] = np.nan
        
    for pid, grp in df_clean.groupby('anon_polygon_id'):
        sub = grp[grp['primary_ndvi'].notna()]
        if len(sub) < 3: continue
        vals = sub['primary_ndvi'].values
        dates = sub['date_dt'].values
        idx_arr = sub.index.values
        for i in range(1, len(vals)-1):
            dt1 = (dates[i] - dates[i-1]).astype('timedelta64[D]').astype(int)
            dt2 = (dates[i+1] - dates[i]).astype('timedelta64[D]').astype(int)
            if dt1 <= 12 and dt2 <= 12 and vals[i] < vals[i-1] - 0.25 and vals[i] < vals[i+1] - 0.25:
                df_clean.loc[idx_arr[i], 'primary_ndvi'] = np.nan
            if vals[i] < -0.05 or vals[i] > 1.05:
                df_clean.loc[idx_arr[i], 'primary_ndvi'] = np.nan

    known = df_clean[df_clean['primary_ndvi'].notna()].copy()
    samples = []
    
    for poly_id, grp in df_clean.groupby('anon_polygon_id'):
        p_known = known[known['anon_polygon_id'] == poly_id]
        if p_known.empty: continue
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
            
            if has_p and has_n:
                dt_tot = dt_p1 + dt_n1
                y_linear = yp1 + (yn1 - yp1) * (dt_p1 / dt_tot)
                wp = dt_n1 / dt_tot
                wn = dt_p1 / dt_tot
            elif has_p:
                dt_tot = dt_p1
                y_linear = yp1
                wp, wn = 1.0, 0.0
            else:
                dt_tot = dt_n1
                y_linear = yn1
                wp, wn = 0.0, 1.0
                
            sp1 = 0 if (has_p and not np.isnan(p_s2[i_p1])) else (1 if (has_p and not np.isnan(p_ls[i_p1])) else 2)
            sn1 = 0 if (has_n and not np.isnan(p_s2[i_n1])) else (1 if (has_n and not np.isnan(p_ls[i_n1])) else 2)
            
            slope_p = (yp1 - yp2) / max(1, (dt_p2 - dt_p1)) if has_p and i_p2 != i_p1 else 0.0
            slope_n = (yn2 - yn1) / max(1, (dt_n2 - dt_n1)) if has_n and i_n2 != i_n1 else 0.0
            
            samples.append({
                'index': idx,
                'anon_polygon_id': poly_id,
                'date': str(row['date']),
                'doy': int(row['doy']),
                'year': int(row['year']),
                'crop_type': str(row['crop_type']),
                'y_linear': y_linear,
                'yp1': yp1,
                'yn1': yn1,
                'yp2': yp2,
                'yn2': yn2,
                'y_p1': yp1,
                'y_n1': yn1,
                'y_p2': yp2,
                'y_n2': yn2,
                'dt_p1': dt_p1,
                'dt_n1': dt_n1,
                'dt_total': dt_tot,
                'min_dt': min(dt_p1, dt_n1),
                'dt_diff': abs(dt_p1 - dt_n1),
                'diff_pn': yn1 - yp1 if (has_p and has_n) else 0.0,
                'w_p': wp,
                'w_n': wn,
                'slope_p': slope_p,
                'slope_n': slope_n,
                'slope_diff': slope_n - slope_p,
                'src_p1': sp1,
                'src_n1': sn1,
                'sp1': sp1,
                'sn1': sn1,
                'temp_interp': row.get('temp_interp', 15.0),
                'precip_interp': row.get('precip_interp', 0.0),
                'precip_7': row.get('precip_7', 0.0),
                'precip_14': row.get('precip_14', 0.0),
                'temp_7': row.get('temp_7', 15.0),
            })
            
    res = pd.DataFrame(samples)
    if res.empty: return res
    
    # Привязка календаря пролетов спутников
    res = res.merge(date_sat_stats[['date', 'tot_s2', 'tot_ls', 'tot_mod', 'inferred_sensor', 'mean_obs']], on='date', how='left')
    res['tot_s2'] = res['tot_s2'].fillna(0)
    res['tot_ls'] = res['tot_ls'].fillna(0)
    res['tot_mod'] = res['tot_mod'].fillna(0)
    res['inferred_sensor'] = res['inferred_sensor'].fillna(1).astype(int)
    
    # Расчет нелинейного смещения сенсоров (Sentinel-2 / Landsat / MODIS)
    def get_bias(sensor, ndvi):
        if sensor == 0: return 0.0
        elif sensor == 1: return 0.045 - 0.050 * np.clip(ndvi, 0.0, 0.9)
        else: return 0.120 - 0.180 * np.clip(ndvi, 0.0, 0.9)
        
    b_t = [get_bias(s, y) for s, y in zip(res['inferred_sensor'], res['y_linear'])]
    b_p = [get_bias(s, y) for s, y in zip(res['src_p1'], res['yp1'])]
    b_n = [get_bias(s, y) for s, y in zip(res['src_n1'], res['yn1'])]
    res['sensor_bias_shift'] = np.array(b_t) - (res['w_p'] * np.array(b_p) + res['w_n'] * np.array(b_n))
    res['sensor_bias_shift'] = res['sensor_bias_shift'].fillna(0.0)
    res['y_calibrated'] = np.clip(res['y_linear'] + res['sensor_bias_shift'], -0.05, 0.98)
    
    # Привязка региональной статистики по культуре
    res = res.merge(date_crop_stats[['crop_type', 'date', 'crop_tot_obs', 'crop_mean_obs', 'crop_std_obs']], on=['crop_type', 'date'], how='left')
    res['crop_mean_obs'] = res['crop_mean_obs'].fillna(res['mean_obs'])
    res['crop_std_obs'] = res['crop_std_obs'].fillna(0.0)
    
    # Привязка климатологической нормы
    res = res.merge(poly_clim, on=['anon_polygon_id', 'doy'], how='left')
    res = res.merge(crop_clim, on=['crop_type', 'doy'], how='left')
    res = res.merge(global_clim, on='doy', how='left')
    
    res['clim_mean'] = res['poly_clim_mean'].fillna(res['crop_clim_mean']).fillna(res['global_clim_mean']).fillna(0.35)
    res['clim_std'] = res['poly_clim_std'].fillna(res['crop_clim_std']).fillna(res['global_clim_std']).fillna(0.06)
    
    res['y_linear'] = res['y_linear'].fillna(res['clim_mean'])
    res['yp1'] = res['yp1'].fillna(res['clim_mean'])
    res['yn1'] = res['yn1'].fillna(res['clim_mean'])
    res['yp2'] = res['yp2'].fillna(res['yp1'])
    res['yn2'] = res['yn2'].fillna(res['yn1'])
    res['y_p1'] = res['yp1']
    res['y_n1'] = res['yn1']
    res['y_p2'] = res['yp2']
    res['y_n2'] = res['yn2']
    
    res['crop_mean_obs'] = res['crop_mean_obs'].fillna(res['clim_mean'])
    res['clim_diff'] = res['y_linear'] - res['clim_mean']
    res['crop_mean_diff'] = res['crop_mean_obs'] - res['clim_mean']
    res['curvature'] = res['y_linear'] - 0.5 * (res['yp1'] + res['yn1'])
    res['curvature_linear'] = res['curvature']
    res['p1_clim_diff'] = res['yp1'] - res['clim_mean']
    res['n1_clim_diff'] = res['yn1'] - res['clim_mean']
    
    res['y_spatial_guide'] = res['clim_mean'] + res['crop_mean_diff']
    res['spatial_diff'] = res['y_spatial_guide'] - res['y_linear']
    
    # Циклическое кодирование времени (sin/cos дня года)
    res['sin_doy'] = np.sin(2 * np.pi * res['doy'] / 365.25)
    res['cos_doy'] = np.cos(2 * np.pi * res['doy'] / 365.25)
    
    return res

feature_cols = [
    'doy', 'sin_doy', 'cos_doy', 'crop_type',
    'y_linear', 'yp1', 'yn1', 'yp2', 'yn2',
    'dt_p1', 'dt_n1', 'dt_total', 'min_dt', 'dt_diff', 'diff_pn',
    'w_p', 'w_n', 'slope_p', 'slope_n', 'slope_diff',
    'src_p1', 'src_n1', 'inferred_sensor', 'sensor_bias_shift',
    'tot_s2', 'tot_ls', 'tot_mod',
    'crop_mean_obs', 'crop_std_obs', 'crop_mean_diff',
    'clim_mean', 'clim_std', 'clim_diff', 'curvature',
    'y_spatial_guide', 'spatial_diff',
    'temp_interp', 'precip_interp', 'precip_7', 'precip_14', 'temp_7'
]

# 7. Извлечение обучающей выборки из обучающего датасета
print("Extracting training samples from train_dataset.csv...")
train_samples = extract_samples(dft)
if len(train_samples) > 22000:
    np.random.seed(42)
    sub_idx = np.random.choice(train_samples.index.values, size=22000, replace=False)
    train_samples = train_samples.loc[sub_idx].reset_index(drop=True)

y_train_true = dft.loc[train_samples['index'], 'primary_ndvi'].values
y_train_lin = train_samples['y_linear'].values
delta_train = np.clip(y_train_true - y_train_lin, -0.25, 0.25)

X = train_samples[feature_cols].copy()
X['crop_type'] = X['crop_type'].astype('category')
groups = train_samples['anon_polygon_id'].values

X_cat = X.copy()
X_cat['crop_type'] = X_cat['crop_type'].astype(str)

# 8. Обучение 5-фолдового ансамбля GroupKFold (LightGBM Huber + CatBoost Huber)
print("\nTraining 5-Fold GroupKFold Ensemble (LightGBM Huber + CatBoost Huber)...")
gkf = GroupKFold(n_splits=5)
lgb_models = []
cat_models = []
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, delta_train, groups=groups)):
    X_tr, y_tr = X.iloc[trn_idx], delta_train[trn_idx]
    X_va, y_va = X.iloc[val_idx], delta_train[val_idx]
    
    # Обучение модели LightGBM с функцией потерь Huber
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    lgb_params = {
        'objective': 'huber', 'alpha': 0.85, 'metric': 'rmse',
        'learning_rate': 0.03, 'num_leaves': 35, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1, 'min_data_in_leaf': 25,
        'seed': 42 + fold, 'verbose': -1
    }
    m_lgb = lgb.train(lgb_params, trn_data, num_boost_round=800, valid_sets=[trn_data, val_data],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    lgb_models.append(m_lgb)
    oof_lgb[val_idx] = m_lgb.predict(X_va)
    
    # Обучение модели CatBoost с функцией потерь Huber
    cb_tr = X_cat.iloc[trn_idx]
    cb_va = X_cat.iloc[val_idx]
    m_cb = CatBoostRegressor(iterations=600, learning_rate=0.035, depth=6, loss_function='Huber:delta=0.08',
                             cat_features=['crop_type'], random_seed=42+fold, verbose=0)
    m_cb.fit(cb_tr, y_tr, eval_set=(cb_va, y_va), early_stopping_rounds=40)
    cat_models.append(m_cb)
    oof_cat[val_idx] = m_cb.predict(cb_va)
    
    f_pred = np.clip(y_train_lin[val_idx] + 0.6 * oof_lgb[val_idx] + 0.4 * oof_cat[val_idx], 0.08, 0.95)
    f_rmse = np.sqrt(mean_squared_error(y_train_true[val_idx], f_pred))
    print(f"Fold {fold+1} OOF Result -> RMSE: {f_rmse:.5f} | GapScore: {30*max(0, 1-f_rmse/0.1):.2f} / 30")

# Оптимизация весов блендинга на OOF-предсказаниях
def opt_weights(w):
    w1, w2 = w[0], w[1]
    p = np.clip(y_train_lin + w1 * oof_lgb + w2 * oof_cat, 0.08, 0.95)
    return np.sqrt(mean_squared_error(y_train_true, p))

res_w = minimize(opt_weights, [0.6, 0.4], bounds=[(0.0, 1.0), (0.0, 1.0)])
w_lgb, w_cat = res_w.x[0] / (res_w.x[0] + res_w.x[1]), res_w.x[1] / (res_w.x[0] + res_w.x[1])
print(f"\nOptimal Ensemble Weights: LightGBM = {w_lgb:.3f}, CatBoost = {w_cat:.3f}")

oof_pred = np.clip(y_train_lin + w_lgb * oof_lgb + w_cat * oof_cat, 0.08, 0.95)
oof_rmse = np.sqrt(mean_squared_error(y_train_true, oof_pred))
print(f"\nOVERALL GROUP-K-FOLD OOF RESULT: RMSE = {oof_rmse:.5f}, GapScore = {30*max(0, 1-oof_rmse/0.1):.2f} / 30")

# Сохранение артефактов и моделей ансамбля
bundle = {
    'lgb_models': lgb_models,
    'cat_models': cat_models,
    'feature_cols': feature_cols,
    'date_sat_stats': date_sat_stats,
    'date_crop_stats': date_crop_stats,
    'lgb_weight': float(w_lgb),
    'cat_weight': float(w_cat)
}
with open("artifacts/models/ensemble_models.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("[OK] Production bundle saved to artifacts/models/ensemble_models.pkl")

# 9. Оценка качества на test_features (1).csv (2500 контрольных точек)
df1 = pd.read_csv(r"data/test_features.csv")
df1['date_dt'] = pd.to_datetime(df1['date'])
df1['year'] = df1['date_dt'].dt.year
df1['doy'] = df1['date_dt'].dt.dayofyear
df1 = df1.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

known_test = df1[df1['primary_ndvi'].notna()].copy()
np.random.seed(42)
val_gap_idx = np.random.choice(known_test.index.values, size=min(2500, len(known_test)), replace=False)

df1_e = enrich_weather(df1)
test_feats = extract_samples(df1_e, target_indices=val_gap_idx)
test_feats = test_feats.sort_values('index').reset_index(drop=True)

y_test_true = df1.loc[test_feats['index'], 'primary_ndvi'].values
y_test_lin = test_feats['y_linear'].values

X_te = test_feats[feature_cols].copy()
X_te['crop_type'] = X_te['crop_type'].astype('category')
X_te_cat = X_te.copy()
X_te_cat['crop_type'] = X_te_cat['crop_type'].astype(str)

p_lgb = np.zeros(len(X_te))
for m in lgb_models: p_lgb += m.predict(X_te) / len(lgb_models)
p_cb = np.zeros(len(X_te))
for m in cat_models: p_cb += m.predict(X_te_cat) / len(cat_models)

final_pred = np.clip(y_test_lin + w_lgb * p_lgb + w_cat * p_cb, 0.08, 0.95)
test_rmse = np.sqrt(mean_squared_error(y_test_true, final_pred))
test_mae = mean_absolute_error(y_test_true, final_pred)
test_gap = 30 * max(0, 1 - test_rmse / 0.10)

print(f"\n================================================================================")
print(f"FINAL EVALUATION ON TEST_FEATURES (1):")
print(f"RMSE:     {test_rmse:.5f}")
print(f"MAE:      {test_mae:.5f}")
print(f"GapScore: {test_gap:.2f} / 30")
print(f"================================================================================")
