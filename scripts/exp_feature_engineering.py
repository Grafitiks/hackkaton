import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold

sys.stdout.reconfigure(encoding='utf-8')

print("=== ADVANCED FEATURE ENGINEERING & RECONSTRUCTION ===")

# 1. Load Train Dataset
df_train = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
df_train['date_dt'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date_dt'].dt.year
df_train = df_train.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

# 2. Extract Weather Features across the continuous daily grid
# Temperature and precipitation are mostly continuous
# Forward/backward fill small missing weather
df_train['era5_temp_interp'] = df_train.groupby('anon_polygon_id')['era5_temp_c'].transform(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
df_train['era5_precip_interp'] = df_train.groupby('anon_polygon_id')['era5_precip_mm'].transform(
    lambda s: s.fillna(0.0)
)
# Rolling weather
df_train['precip_sum_7'] = df_train.groupby('anon_polygon_id')['era5_precip_interp'].transform(
    lambda s: s.rolling(7, min_periods=1).sum()
)
df_train['precip_sum_14'] = df_train.groupby('anon_polygon_id')['era5_precip_interp'].transform(
    lambda s: s.rolling(14, min_periods=1).sum()
)
df_train['temp_mean_7'] = df_train.groupby('anon_polygon_id')['era5_temp_interp'].transform(
    lambda s: s.rolling(7, min_periods=1).mean()
)

# 3. Build Polygon DOY climatology from train observations
clim_doy = df_train[df_train['primary_ndvi'].notna()].groupby(['anon_polygon_id', 'doy'])['primary_ndvi'].agg(
    clim_mean='mean', clim_std='std', clim_median='median'
).reset_index()

# Also crop_type DOY climatology (crucial for new polygons in test!)
clim_crop_doy = df_train[df_train['primary_ndvi'].notna()].groupby(['crop_type', 'doy'])['primary_ndvi'].agg(
    crop_clim_mean='mean', crop_clim_std='std', crop_clim_median='median'
).reset_index()

# Overall DOY climatology (fallback)
clim_global_doy = df_train[df_train['primary_ndvi'].notna()].groupby('doy')['primary_ndvi'].agg(
    global_clim_mean='mean', global_clim_std='std'
).reset_index()

print("Climatology tables constructed successfully.")

# Function to build gap dataset for any series
def create_gap_samples(df, gap_indices):
    """
    Given dataframe and indices of gaps to predict,
    computes features using ONLY points outside gap_indices!
    """
    df_work = df.copy()
    df_work.loc[gap_indices, 'primary_ndvi'] = np.nan
    
    # Identify known observations
    known = df_work[df_work['primary_ndvi'].notna()].copy()
    
    samples = []
    # Process polygon by polygon for speed
    for poly_id, p_df in df_work.groupby('anon_polygon_id'):
        p_gaps = p_df[p_df.index.isin(gap_indices)]
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
        p_known_evi = p_known['s2_evi'].fillna(p_known['landsat_evi']).fillna(p_known['modis_evi']).values
        p_known_ndwi = p_known['s2_ndwi'].fillna(p_known['landsat_ndwi']).values
        
        for idx, row in p_gaps.iterrows():
            t = np.datetime64(row['date_dt'])
            
            # Find previous and next known points
            prev_idx = np.where(p_known_dates < t)[0]
            next_idx = np.where(p_known_dates > t)[0]
            
            # Prev 1 & 2
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
            
            # Linear interpolation
            if has_prev and has_next:
                dt_total = dt_p1 + dt_n1
                y_linear = y_p1 + (y_n1 - y_p1) * (dt_p1 / dt_total)
                weight_p = dt_n1 / dt_total
                weight_n = dt_p1 / dt_total
            elif has_prev:
                y_linear = y_p1
                weight_p = 1.0
                weight_n = 0.0
                dt_total = dt_p1
            elif has_next:
                y_linear = y_n1
                weight_p = 0.0
                weight_n = 1.0
                dt_total = dt_n1
            else:
                y_linear = np.nan
                weight_p = 0.0
                weight_n = 0.0
                dt_total = 999
                
            # Sensor indicators
            p1_is_s2 = int(not np.isnan(p_known_s2[i_p1])) if has_prev else 0
            p1_is_mod = int(not np.isnan(p_known_mod[i_p1])) if has_prev else 0
            n1_is_s2 = int(not np.isnan(p_known_s2[i_n1])) if has_next else 0
            n1_is_mod = int(not np.isnan(p_known_mod[i_n1])) if has_next else 0
            
            evi_p1 = p_known_evi[i_p1] if has_prev else np.nan
            evi_n1 = p_known_evi[i_n1] if has_next else np.nan
            ndwi_p1 = p_known_ndwi[i_p1] if has_prev else np.nan
            ndwi_n1 = p_known_ndwi[i_n1] if has_next else np.nan
            
            # Slopes
            slope_p = (y_p1 - y_p2) / max(1, (dt_p2 - dt_p1)) if has_prev and i_p2 != i_p1 else 0.0
            slope_n = (y_n2 - y_n1) / max(1, (dt_n2 - dt_n1)) if has_next and i_n2 != i_n1 else 0.0
            
            samples.append({
                'index': idx,
                'anon_polygon_id': poly_id,
                'date': row['date'],
                'doy': row['doy'],
                'year': row['year'],
                'crop_type': row['crop_type'],
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
                'weight_p': weight_p,
                'weight_n': weight_n,
                'slope_p': slope_p,
                'slope_n': slope_n,
                'p1_is_s2': p1_is_s2,
                'p1_is_mod': p1_is_mod,
                'n1_is_s2': n1_is_s2,
                'n1_is_mod': n1_is_mod,
                'evi_p1': evi_p1,
                'evi_n1': evi_n1,
                'ndwi_p1': ndwi_p1,
                'ndwi_n1': ndwi_n1,
                'era5_temp_interp': row['era5_temp_interp'],
                'era5_precip_interp': row['era5_precip_interp'],
                'precip_sum_7': row['precip_sum_7'],
                'precip_sum_14': row['precip_sum_14'],
                'temp_mean_7': row['temp_mean_7'],
            })
            
    res_df = pd.DataFrame(samples)
    
    # Merge climatologies
    res_df = res_df.merge(clim_doy, on=['anon_polygon_id', 'doy'], how='left')
    res_df = res_df.merge(clim_crop_doy, on=['crop_type', 'doy'], how='left')
    res_df = res_df.merge(clim_global_doy, on='doy', how='left')
    
    # Fill fallback climatology
    res_df['clim_mean'] = res_df['clim_mean'].fillna(res_df['crop_clim_mean']).fillna(res_df['global_clim_mean'])
    res_df['clim_std'] = res_df['clim_std'].fillna(res_df['crop_clim_std']).fillna(res_df['global_clim_std'])
    
    # Climatology deviation features
    res_df['y_linear_clim_diff'] = res_df['y_linear'] - res_df['clim_mean']
    res_df['p1_clim_diff'] = res_df['y_p1'] - res_df['clim_mean']
    res_df['n1_clim_diff'] = res_df['y_n1'] - res_df['clim_mean']
    
    # Climatology interpolated anomaly
    # If we interpolate the anomaly from p1 and n1:
    res_df['interp_anom'] = res_df['weight_p'] * res_df['p1_clim_diff'] + res_df['weight_n'] * res_df['n1_clim_diff']
    res_df['clim_guided_pred'] = res_df['clim_mean'] + res_df['interp_anom']
    
    # Cyclic calendar features
    res_df['sin_doy'] = np.sin(2 * np.pi * res_df['doy'] / 365.25)
    res_df['cos_doy'] = np.cos(2 * np.pi * res_df['doy'] / 365.25)
    
    # Categorical
    res_df['crop_type'] = res_df['crop_type'].astype('category')
    res_df['anon_polygon_id'] = res_df['anon_polygon_id'].astype('category')
    
    return res_df

print("Feature extractor defined.")

# 4. Simulation Validation
# Sample synthetic gaps matching the distribution of test (consecutive 1s, occasional 2s)
np.random.seed(42)
all_known_idx = df_train[df_train['primary_ndvi'].notna()].index.values

# Select 3500 points for validation
val_idx = np.random.choice(all_known_idx, size=3500, replace=False)
val_features = create_gap_samples(df_train, val_idx)
val_features = val_features.sort_values('index').reset_index(drop=True)
y_val_true = df_train.loc[val_features['index'], 'primary_ndvi'].values

# Check Linear Baseline on this validation set
y_val_linear = val_features['y_linear'].values
rmse_lin = np.sqrt(mean_squared_error(y_val_true, y_val_linear))
gap_lin = round(float(30 * max(0, 1 - rmse_lin / 0.1)), 2)
print(f"\n[Baseline Linear]      RMSE: {rmse_lin:.5f} | GapScore: {gap_lin:5.2f} / 30")

# Check Climatology Guided on this validation set
y_val_clim = pd.Series(val_features['clim_guided_pred'].values).fillna(pd.Series(y_val_linear)).values
rmse_clim = np.sqrt(mean_squared_error(y_val_true, y_val_clim))
gap_clim = round(float(30 * max(0, 1 - rmse_clim / 0.1)), 2)
print(f"[Climatology-Guided]   RMSE: {rmse_clim:.5f} | GapScore: {gap_clim:5.2f} / 30")

# Now create TRAINING gap samples to train LightGBM!
# We can sample 12,000 points from remaining known points
train_candidate_idx = np.setdiff1d(all_known_idx, val_idx)
train_sampled_idx = np.random.choice(train_candidate_idx, size=12000, replace=False)

print(f"\nBuilding training features for {len(train_sampled_idx)} gap samples...")
train_features = create_gap_samples(df_train, train_sampled_idx)
y_train_true = df_train.loc[train_features['index'], 'primary_ndvi'].values

feature_cols = [
    'doy', 'sin_doy', 'cos_doy', 'crop_type',
    'y_linear', 'y_p1', 'y_n1', 'y_p2', 'y_n2',
    'dt_p1', 'dt_n1', 'dt_total', 'min_dt', 'diff_pn',
    'weight_p', 'weight_n', 'slope_p', 'slope_n',
    'p1_is_s2', 'p1_is_mod', 'n1_is_s2', 'n1_is_mod',
    'evi_p1', 'evi_n1', 'ndwi_p1', 'ndwi_n1',
    'era5_temp_interp', 'era5_precip_interp', 'precip_sum_7', 'precip_sum_14', 'temp_mean_7',
    'clim_mean', 'clim_std', 'y_linear_clim_diff', 'p1_clim_diff', 'n1_clim_diff',
    'interp_anom', 'clim_guided_pred'
]

print(f"Features count: {len(feature_cols)}")

train_features['y_linear'] = train_features['y_linear'].fillna(train_features['clim_mean']).fillna(0.35)
val_features['y_linear'] = val_features['y_linear'].fillna(val_features['clim_mean']).fillna(0.35)
y_val_linear = val_features['y_linear'].values

X_tr = train_features[feature_cols]
y_tr = y_train_true
# Predict residual delta = y_true - y_linear
delta_tr = y_tr - train_features['y_linear'].values

X_va = val_features[feature_cols]
delta_va = y_val_true - y_val_linear

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'seed': 42,
    'verbose': -1
}

train_data = lgb.Dataset(X_tr, label=delta_tr)
val_data = lgb.Dataset(X_va, label=delta_va, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=800,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

delta_pred = model.predict(X_va)
y_pred_lgb = y_val_linear + delta_pred
# Clip to physical NDVI range [-0.2, 1.0]
y_pred_lgb = np.clip(y_pred_lgb, -0.2, 1.0)

rmse_lgb = np.sqrt(mean_squared_error(y_val_true, y_pred_lgb))
gap_lgb = round(float(30 * max(0, 1 - rmse_lgb / 0.1)), 2)
print(f"\n=======================================================")
print(f"[LightGBM Residual Model] RMSE: {rmse_lgb:.5f} | GapScore: {gap_lgb:5.2f} / 30")
print(f"=======================================================")

# Feature importances
imp = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols).sort_values(ascending=False)
print("\nTop 15 Most Important Features:")
print(imp.head(15))
