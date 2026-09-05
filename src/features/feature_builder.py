import pandas as pd
import numpy as np

# Калибровочные полиномы для перевода сенсоров в эталонное пространство Sentinel-2 (10м)
POLY_LS_TO_S2 = [0.01770809, 1.00205161, -0.04086545]
POLY_MOD_TO_S2 = [0.29875542, 0.747099, -0.04116821]

POLY_S2_TO_LS = [-0.0139963, 0.9290673, 0.06358853]
POLY_S2_TO_MOD = [-0.16747713, 0.87583154, 0.16369809]

def to_s2_space(val: float, sensor_code: int) -> float:
    """Приведение исходного NDVI любого сенсора к эталонной шкале Sentinel-2."""
    if np.isnan(val):
        return np.nan
    if sensor_code == 0:
        return val
    elif sensor_code == 1:
        return POLY_LS_TO_S2[0] * val**2 + POLY_LS_TO_S2[1] * val + POLY_LS_TO_S2[2]
    else:
        return POLY_MOD_TO_S2[0] * val**2 + POLY_MOD_TO_S2[1] * val + POLY_MOD_TO_S2[2]

def from_s2_space(val: float, sensor_code: int) -> float:
    """Отображение гармонизированного прогноза S2 в шкалу ожидаемого сенсора пропуска."""
    if np.isnan(val):
        return np.nan
    if sensor_code == 0:
        return val
    elif sensor_code == 1:
        return POLY_S2_TO_LS[0] * val**2 + POLY_S2_TO_LS[1] * val + POLY_S2_TO_LS[2]
    else:
        return POLY_S2_TO_MOD[0] * val**2 + POLY_S2_TO_MOD[1] * val + POLY_S2_TO_MOD[2]

def enrich_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Интерполяция метеопараметров ERA5-Land и агроклиматические агрегаты за 7 и 14 дней."""
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

def clean_anchor_series(df: pd.DataFrame, gap_indices: np.ndarray = None) -> pd.DataFrame:
    """Фильтрация артефактов и ложных облачных теней из опорных измерений."""
    df_clean = df.copy()
    if gap_indices is not None and len(gap_indices) > 0:
        df_clean.loc[gap_indices, 'primary_ndvi'] = np.nan

    for pid, grp in df_clean.groupby('anon_polygon_id'):
        sub = grp[grp['primary_ndvi'].notna()]
        if len(sub) < 3:
            continue
        vals = sub['primary_ndvi'].values
        dates = sub['date_dt'].values
        idx_arr = sub.index.values
        
        for i in range(1, len(vals) - 1):
            dt1 = (dates[i] - dates[i - 1]).astype('timedelta64[D]').astype(int)
            dt2 = (dates[i + 1] - dates[i]).astype('timedelta64[D]').astype(int)
            if dt1 <= 12 and dt2 <= 12:
                if vals[i] < vals[i - 1] - 0.25 and vals[i] < vals[i + 1] - 0.25:
                    df_clean.loc[idx_arr[i], 'primary_ndvi'] = np.nan
            if vals[i] < -0.05 or vals[i] > 1.05:
                df_clean.loc[idx_arr[i], 'primary_ndvi'] = np.nan
                
    return df_clean

def get_sensor_bias(sensor_code: int, ndvi: float) -> float:
    """Эмпирическая нелинейная калибровочная функция смещения сенсоров."""
    ndvi_c = np.clip(ndvi, 0.0, 0.9)
    if sensor_code == 0:
        return 0.0
    elif sensor_code == 1:
        return 0.045 - 0.050 * ndvi_c
    else:  # Спектрометр MODIS (250м)
        return 0.120 - 0.180 * ndvi_c

def extract_gap_features(
    df: pd.DataFrame,
    gap_indices: np.ndarray,
    poly_clim_df: pd.DataFrame,
    crop_clim_df: pd.DataFrame,
    global_clim_df: pd.DataFrame,
    date_sat_stats: pd.DataFrame = None,
    date_crop_stats: pd.DataFrame = None,
    is_train: bool = False
) -> pd.DataFrame:
    """
    Унифицированная генерация признаков для точек пропуска.
    Включает межсенсорную гармонизацию, региональные аномалии, кривизну и метеопараметры.
    """
    df_clean = clean_anchor_series(df, None if is_train else gap_indices)
    known = df_clean[df_clean['primary_ndvi'].notna()].copy()
    
    samples = []
    
    for poly_id, grp in df_clean.groupby('anon_polygon_id'):
        p_gaps = grp[grp.index.isin(gap_indices)]
        if p_gaps.empty:
            continue
            
        p_known = known[known['anon_polygon_id'] == poly_id]
        if p_known.empty:
            continue
            
        p_dates = p_known['date_dt'].values
        p_vals = p_known['primary_ndvi'].values
        p_s2 = p_known['s2_ndvi'].values if 's2_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        p_ls = p_known['landsat_ndvi'].values if 'landsat_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        
        for idx, row in p_gaps.iterrows():
            t = np.datetime64(row['date_dt'])
            
            prev_idx = np.where(p_dates < t)[0]
            next_idx = np.where(p_dates > t)[0]
            
            has_p = len(prev_idx) > 0
            has_n = len(next_idx) > 0
            if not has_p and not has_n:
                continue
                
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
                
            slope_p = (yp1 - yp2) / max(1, (dt_p2 - dt_p1)) if has_p and i_p2 != i_p1 else 0.0
            slope_n = (yn2 - yn1) / max(1, (dt_n2 - dt_n1)) if has_n and i_n2 != i_n1 else 0.0
            
            samples.append({
                'index': idx,
                'anon_polygon_id': poly_id,
                'date': str(row['date']),
                'doy': int(pd.to_datetime(row['date_dt']).dayofyear),
                'year': int(pd.to_datetime(row['date_dt']).year),
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
    if res.empty:
        return res
        
    # Календарь спутников
    if date_sat_stats is not None:
        res = res.merge(date_sat_stats[['date', 'tot_s2', 'tot_ls', 'tot_mod', 'inferred_sensor', 'mean_obs']], on='date', how='left')
    else:
        res['tot_s2'] = 0
        res['tot_ls'] = 0
        res['tot_mod'] = 0
        res['inferred_sensor'] = 1
        res['mean_obs'] = np.nan
        
    res['tot_s2'] = res['tot_s2'].fillna(0)
    res['tot_ls'] = res['tot_ls'].fillna(0)
    res['tot_mod'] = res['tot_mod'].fillna(0)
    res['inferred_sensor'] = res['inferred_sensor'].fillna(1).astype(int)
    
    # Нелинейное калибровочное смещение сенсоров
    b_t = [get_sensor_bias(s, y) for s, y in zip(res['inferred_sensor'], res['y_linear'])]
    b_p = [get_sensor_bias(s, y) for s, y in zip(res['src_p1'], res['yp1'])]
    b_n = [get_sensor_bias(s, y) for s, y in zip(res['src_n1'], res['yn1'])]
    res['sensor_bias_shift'] = np.array(b_t) - (res['w_p'] * np.array(b_p) + res['w_n'] * np.array(b_n))
    res['sensor_bias_shift'] = res['sensor_bias_shift'].fillna(0.0)
    
    # Калиброванный базис
    res['y_calibrated'] = np.clip(res['y_linear'] + res['sensor_bias_shift'], -0.05, 0.98)
    
    # Региональные данные по культуре
    if date_crop_stats is not None:
        res = res.merge(date_crop_stats[['crop_type', 'date', 'crop_tot_obs', 'crop_mean_obs', 'crop_std_obs']], on=['crop_type', 'date'], how='left')
    else:
        res['crop_tot_obs'] = 0
        res['crop_mean_obs'] = res['mean_obs']
        res['crop_std_obs'] = 0.0
        
    res['crop_mean_obs'] = res['crop_mean_obs'].fillna(res['mean_obs'])
    res['crop_std_obs'] = res['crop_std_obs'].fillna(0.0)
    
    # Климатология
    res = res.merge(poly_clim_df, on=['anon_polygon_id', 'doy'], how='left')
    res = res.merge(crop_clim_df, on=['crop_type', 'doy'], how='left')
    res = res.merge(global_clim_df, on='doy', how='left')
    
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
    res['y_calibrated'] = res['y_calibrated'].fillna(res['clim_mean'])
    
    if 'clim_slope' not in res.columns:
        res['clim_slope'] = 0.0
    res['clim_slope'] = res['clim_slope'].fillna(0.0)
    res['taylor_curve'] = 0.5 * res['dt_p1'] * res['dt_n1'] * res['clim_slope']
    
    res['crop_mean_obs'] = res['crop_mean_obs'].fillna(res['clim_mean'])
    res['clim_diff'] = res['y_linear'] - res['clim_mean']
    res['crop_mean_diff'] = res['crop_mean_obs'] - res['clim_mean']
    res['curvature'] = res['y_linear'] - 0.5 * (res['yp1'] + res['yn1'])
    res['curvature_linear'] = res['curvature']
    res['p1_clim_diff'] = res['yp1'] - res['clim_mean']
    res['n1_clim_diff'] = res['yn1'] - res['clim_mean']
    
    res['y_spatial_guide'] = res['clim_mean'] + res['crop_mean_diff']
    res['spatial_diff'] = res['y_spatial_guide'] - res['y_linear']
    
    # Циклическое время
    res['sin_doy'] = np.sin(2 * np.pi * res['doy'] / 365.25)
    res['cos_doy'] = np.cos(2 * np.pi * res['doy'] / 365.25)
    
    return res

FEATURE_COLUMNS = [
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
