import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=== STARTING FULL HIGH-ACCURACY GAP RESTORATION PIPELINE ===")

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

# 2. Обогащение метеоданными для всех полигонов
def enrich_weather_full(df):
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

dft = enrich_weather_full(dft)
df1 = enrich_weather_full(df1)
df_all = enrich_weather_full(df_all)

# 3. Статистика пролетов спутников на глобальном уровне дат
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
date_sat_stats['inferred_sensor'] = np.where(date_sat_stats['has_s2'] == 1, 0,
                                    np.where(date_sat_stats['has_ls'] == 1, 1, 2))

date_crop_stats = df_all[df_all['primary_ndvi'].notna()].groupby(['crop_type', 'date']).agg(
    crop_tot_obs=('primary_ndvi', 'count'),
    crop_mean_obs=('primary_ndvi', 'mean'),
    crop_std_obs=('primary_ndvi', 'std')
).reset_index()
date_crop_stats['crop_std_obs'] = date_crop_stats['crop_std_obs'].fillna(0.0)

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

# Производная климатологии (фенологический наклон)
global_clim['clim_slope'] = (global_clim['global_clim_mean'].shift(-2) - global_clim['global_clim_mean'].shift(2)) / 4.0
global_clim['clim_slope'] = global_clim['clim_slope'].bfill().ffill()

print("Climatologies and date-level statistics built.")

# 5. Быстрое и комплексное извлечение обучающей выборки
def extract_dataset_samples(df, max_samples=25000, is_test=False, target_indices=None):
    df_work = df.copy()
    if target_indices is not None:
        df_work.loc[target_indices, 'primary_ndvi'] = np.nan
        
    known = df_work[df_work['primary_ndvi'].notna()].copy()
    
    samples = []
    
    for poly_id, grp in df_work.groupby('anon_polygon_id'):
        p_known = known[known['anon_polygon_id'] == poly_id]
        if p_known.empty:
            continue
            
        p_known_dates = p_known['date_dt'].values
        p_known_vals = p_known['primary_ndvi'].values
        p_known_s2 = p_known['s2_ndvi'].values
        p_known_ls = p_known['landsat_ndvi'].values
        p_known_mod = p_known['modis_ndvi'].values
        
        if target_indices is not None:
            # Целевые строки для оценки
            eval_rows = grp[grp.index.isin(target_indices)]
        else:
            # Генерация внутренних точек для обучения заполнению пропусков
            # Для каждой внутренней известной точки i в диапазоне 1..n-2
            eval_rows = p_known.iloc[1:-1]
            
        for idx, row in eval_rows.iterrows():
            t = np.datetime64(row['date_dt'])
            
            # При генерации из известных точек исключаем текущую точку из соседей!
            if target_indices is None:
                prev_idx = np.where(p_known_dates < t)[0]
                next_idx = np.where(p_known_dates > t)[0]
            else:
                prev_idx = np.where(p_known_dates < t)[0]
                next_idx = np.where(p_known_dates > t)[0]
                
            has_prev = len(prev_idx) > 0
            has_next = len(next_idx) > 0
            if not has_prev and not has_next:
                continue
                
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
            
            samples.append({
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
                'dt_diff': abs(dt_p1 - dt_n1),
                'diff_pn': y_n1 - y_p1 if (has_prev and has_next) else 0.0,
                'w_p': w_p,
                'w_n': w_n,
                'slope_p': slope_p,
                'slope_n': slope_n,
                'slope_diff': slope_n - slope_p,
                'src_p1': src_p1,
                'src_n1': src_n1,
                'temp_interp': row.get('temp_interp', 15.0),
                'precip_interp': row.get('precip_interp', 0.0),
                'precip_7': row.get('precip_7', 0.0),
                'precip_14': row.get('precip_14', 0.0),
                'temp_7': row.get('temp_7', 15.0),
            })
            
    res = pd.DataFrame(samples)
    
    # Привязка календаря пролетов спутников
    res = res.merge(date_sat_stats[['date', 'tot_s2', 'tot_ls', 'tot_mod', 'inferred_sensor', 'mean_obs']], on='date', how='left')
    res['tot_s2'] = res['tot_s2'].fillna(0)
    res['tot_ls'] = res['tot_ls'].fillna(0)
    res['tot_mod'] = res['tot_mod'].fillna(0)
    res['inferred_sensor'] = res['inferred_sensor'].fillna(1).astype(int)
    
    # Расчет нелинейного смещения сенсоров в зависимости от уровня NDVI
    def get_bias(sensor, ndvi):
        # Sentinel-2: нулевая поправка (эталон)
        # Landsat: +0.050 при низком NDVI, +0.030 при среднем, -0.005 при высоком
        # MODIS: +0.130 при низком NDVI, +0.060 при среднем, -0.050 при высоком
        if sensor == 0:
            return 0.0
        elif sensor == 1:
            return 0.045 - 0.050 * np.clip(ndvi, 0.0, 0.9)
        else: # MODIS (250м)
            return 0.120 - 0.180 * np.clip(ndvi, 0.0, 0.9)
            
    b_t = [get_bias(s, y) for s, y in zip(res['inferred_sensor'], res['y_linear'])]
    b_p = [get_bias(s, y) for s, y in zip(res['src_p1'], res['y_p1'])]
    b_n = [get_bias(s, y) for s, y in zip(res['src_n1'], res['y_n1'])]
    res['sensor_bias_shift'] = np.array(b_t) - (res['w_p'] * np.array(b_p) + res['w_n'] * np.array(b_n))
    
    # Привязка региональной статистики по культурам
    res = res.merge(date_crop_stats[['crop_type', 'date', 'crop_tot_obs', 'crop_mean_obs', 'crop_std_obs']], on=['crop_type', 'date'], how='left')
    res['crop_mean_obs'] = res['crop_mean_obs'].fillna(res['mean_obs'])
    res['crop_std_obs'] = res['crop_std_obs'].fillna(0.0)
    
    # Привязка климатологической нормы
    res = res.merge(poly_clim, on=['anon_polygon_id', 'doy'], how='left')
    res = res.merge(crop_clim, on=['crop_type', 'doy'], how='left')
    res = res.merge(global_clim, on='doy', how='left')
    
    res['clim_mean'] = res['poly_clim_mean'].fillna(res['crop_clim_mean']).fillna(res['global_clim_mean']).fillna(0.35)
    res['clim_std'] = res['poly_clim_std'].fillna(res['crop_clim_std']).fillna(res['global_clim_std']).fillna(0.06)
    res['clim_slope'] = res['clim_slope'].fillna(0.0)
    
    res['y_linear'] = res['y_linear'].fillna(res['clim_mean'])
    res['y_p1'] = res['y_p1'].fillna(res['clim_mean'])
    res['y_n1'] = res['y_n1'].fillna(res['clim_mean'])
    res['y_p2'] = res['y_p2'].fillna(res['y_p1'])
    res['y_n2'] = res['y_n2'].fillna(res['y_n1'])
    
    res['clim_diff'] = res['y_linear'] - res['clim_mean']
    res['crop_mean_diff'] = res['crop_mean_obs'] - res['clim_mean']
    res['p1_clim_diff'] = res['y_p1'] - res['clim_mean']
    res['n1_clim_diff'] = res['y_n1'] - res['clim_mean']
    
    # Аппроксимация кривизны динамики
    res['curvature_linear'] = res['y_linear'] - 0.5 * (res['y_p1'] + res['y_n1'])
    res['taylor_curve'] = 0.5 * res['dt_p1'] * res['dt_n1'] * res['clim_slope']
    
    # Циклическое кодирование времени (sin/cos дня года)
    res['sin_doy'] = np.sin(2 * np.pi * res['doy'] / 365.25)
    res['cos_doy'] = np.cos(2 * np.pi * res['doy'] / 365.25)
    
    return res

print("Generating comprehensive training dataset from train_dataset.csv (30,442 points)...")
train_full_df = extract_dataset_samples(dft)
print(f"Generated {len(train_full_df)} rich training samples.")

# Сэмплирование сбалансированной подвыборки
if len(train_full_df) > 22000:
    np.random.seed(42)
    sample_idx = np.random.choice(train_full_df.index.values, size=22000, replace=False)
    train_full_df = train_full_df.loc[sample_idx].sort_values(['anon_polygon_id', 'date']).reset_index(drop=True)
    print(f"Sampled {len(train_full_df)} balanced points for training.")

y_train_true = dft.loc[train_full_df['index'], 'primary_ndvi'].values
y_train_linear = train_full_df['y_linear'].values
delta_train_true = y_train_true - y_train_linear

# Ограничение выбросов для предотвращения искажения решающих деревьев
delta_clipped = np.clip(delta_train_true, -0.30, 0.30)

feature_cols = [
    'doy', 'sin_doy', 'cos_doy', 'crop_type',
    'y_linear', 'y_p1', 'y_n1', 'y_p2', 'y_n2',
    'dt_p1', 'dt_n1', 'dt_total', 'min_dt', 'dt_diff', 'diff_pn',
    'w_p', 'w_n', 'slope_p', 'slope_n', 'slope_diff',
    'src_p1', 'src_n1', 'inferred_sensor', 'sensor_bias_shift',
    'tot_s2', 'tot_ls', 'tot_mod',
    'crop_mean_obs', 'crop_std_obs', 'crop_mean_diff',
    'clim_mean', 'clim_std', 'clim_diff', 'clim_slope',
    'p1_clim_diff', 'n1_clim_diff', 'curvature_linear', 'taylor_curve',
    'temp_interp', 'precip_interp', 'precip_7', 'precip_14', 'temp_7'
]

X_tr_all = train_full_df[feature_cols].copy()
X_tr_all['crop_type'] = X_tr_all['crop_type'].astype('category')
groups_all = train_full_df['anon_polygon_id'].values

print(f"\nTraining 5-Fold GroupKFold Ensemble (LightGBM + CatBoost) on unseen polygons...")
gkf = GroupKFold(n_splits=5)

lgb_models = []
cat_models = []
oof_pred_lgb = np.zeros(len(X_tr_all))
oof_pred_cat = np.zeros(len(X_tr_all))

X_cat_all = X_tr_all.copy()
X_cat_all['crop_type'] = X_cat_all['crop_type'].astype(str)

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tr_all, delta_clipped, groups=groups_all)):
    print(f"--- FOLD {fold+1} / 5 ---")
    X_f_tr, y_f_tr = X_tr_all.iloc[trn_idx], delta_clipped[trn_idx]
    X_f_va, y_f_va = X_tr_all.iloc[val_idx], delta_clipped[val_idx]
    
    # Модель LightGBM с функцией потерь Huber для робастности к выбросам
    train_data = lgb.Dataset(X_f_tr, label=y_f_tr)
    val_data = lgb.Dataset(X_f_va, label=y_f_va, reference=train_data)
    
    lgb_params = {
        'objective': 'huber',
        'alpha': 0.85,
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 35,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 25,
        'seed': 42 + fold,
        'verbose': -1
    }
    
    m_lgb = lgb.train(
        lgb_params, train_data, num_boost_round=800,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)]
    )
    lgb_models.append(m_lgb)
    oof_pred_lgb[val_idx] = m_lgb.predict(X_f_va)
    
    # Модель CatBoost
    cb_tr = X_cat_all.iloc[trn_idx]
    cb_va = X_cat_all.iloc[val_idx]
    m_cat = CatBoostRegressor(
        iterations=600,
        learning_rate=0.035,
        depth=6,
        loss_function='RMSE',
        cat_features=['crop_type'],
        random_seed=42 + fold,
        verbose=0
    )
    m_cat.fit(cb_tr, y_f_tr, eval_set=(cb_va, y_f_va), early_stopping_rounds=40)
    cat_models.append(m_cat)
    oof_pred_cat[val_idx] = m_cat.predict(cb_va)
    
    fold_blend = 0.6 * oof_pred_lgb[val_idx] + 0.4 * oof_pred_cat[val_idx]
    fold_pred_y = np.clip(y_train_linear[val_idx] + fold_blend, -0.1, 1.0)
    fold_rmse = np.sqrt(mean_squared_error(y_train_true[val_idx], fold_pred_y))
    fold_gap = 30 * max(0, 1 - fold_rmse / 0.10)
    print(f"Fold {fold+1} Result -> RMSE: {fold_rmse:.5f} | GapScore: {fold_gap:.2f} / 30")

oof_blend = 0.6 * oof_pred_lgb + 0.4 * oof_pred_cat
oof_pred_y = np.clip(y_train_linear + oof_blend, -0.1, 1.0)
oof_rmse = np.sqrt(mean_squared_error(y_train_true, oof_pred_y))
oof_gap = 30 * max(0, 1 - oof_rmse / 0.10)
base_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_linear))
base_gap = 30 * max(0, 1 - base_rmse / 0.10)

print(f"\n=======================================================")
print(f"FULL GROUP-K-FOLD OOF RESULT (Strictly Unseen Polygons):")
print(f"Baseline Linear: RMSE = {base_rmse:.5f} | GapScore = {base_gap:.2f}")
print(f"Model Ensemble:  RMSE = {oof_rmse:.5f} | GapScore = {oof_gap:.2f}")
print(f"Improvement:     {base_rmse - oof_rmse:.5f} RMSE reduction (+{oof_gap - base_gap:.2f} score)")
print(f"=======================================================")

# Сохранение обученных моделей и артефактов
os.makedirs("artifacts/models", exist_ok=True)
with open("artifacts/models/ensemble_models.pkl", "wb") as f:
    pickle.dump({
        'lgb_models': lgb_models,
        'cat_models': cat_models,
        'feature_cols': feature_cols,
        'date_sat_stats': date_sat_stats,
        'date_crop_stats': date_crop_stats
    }, f)

with open("artifacts/models/climatology.pkl", "wb") as f:
    pickle.dump({
        'poly_clim': poly_clim,
        'crop_clim': crop_clim,
        'global_clim': global_clim
    }, f)

print("Models and climatology saved.")

# 6. Оценка качества на test_features (1).csv (2500 контрольных точек)
known_test = df1[df1['primary_ndvi'].notna()].copy()
np.random.seed(42)
val_gap_idx = np.random.choice(known_test.index.values, size=min(2500, len(known_test)), replace=False)

print(f"\nEvaluating on {len(val_gap_idx)} ground-truth validation points in test_features (1)...")
X_test_df = extract_dataset_samples(df1, is_test=True, target_indices=val_gap_idx)
X_test_df = X_test_df.sort_values('index').reset_index(drop=True)

y_test_true = df1.loc[X_test_df['index'], 'primary_ndvi'].values
y_test_linear = X_test_df['y_linear'].values

base_test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_linear))
base_test_mae = mean_absolute_error(y_test_true, y_test_linear)
base_test_gap = 30 * max(0, 1 - base_test_rmse / 0.10)

X_test = X_test_df[feature_cols].copy()
X_test['crop_type'] = X_test['crop_type'].astype('category')
X_test_cat = X_test.copy()
X_test_cat['crop_type'] = X_test_cat['crop_type'].astype(str)

preds_lgb = np.zeros(len(X_test))
for m in lgb_models:
    preds_lgb += m.predict(X_test) / len(lgb_models)
    
preds_cat = np.zeros(len(X_test))
for m in cat_models:
    preds_cat += m.predict(X_test_cat) / len(cat_models)
    
preds_delta = 0.6 * preds_lgb + 0.4 * preds_cat
y_test_pred = np.clip(y_test_linear + preds_delta, -0.1, 1.0)

test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
test_mae = mean_absolute_error(y_test_true, y_test_pred)
test_gap = 30 * max(0, 1 - test_rmse / 0.10)

print('===============================================================')
print(f'FINAL TEST_FEATURES (1) EVALUATION RESULTS:')
print(f'Linear Baseline: RMSE = {base_test_rmse:.5f}, MAE = {base_test_mae:.5f}, GapScore = {base_test_gap:.2f} / 30')
print(f'Previous Model:  RMSE = 0.08617, MAE = 0.05704, GapScore = 4.15 / 30')
print(f'NEW MODEL:       RMSE = {test_rmse:.5f}, MAE = {test_mae:.5f}, GapScore = {test_gap:.2f} / 30')
print('===============================================================')
