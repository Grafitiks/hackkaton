import pandas as pd
import numpy as np

def enrich_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Интерполяция непрерывных метеорологических параметров ERA5-Land и расчет
    накопленных агроклиматических метрик (суммы осадков за 7 и 14 дней, средняя температура).
    Позволяет учесть гидротермический стресс и засушливые фазы.
    """
    df = df.copy()
    if 'era5_temp_c' in df.columns:
        df['era5_temp_interp'] = df.groupby('anon_polygon_id')['era5_temp_c'].transform(
            lambda s: s.interpolate(method='linear', limit_direction='both')
        )
    else:
        df['era5_temp_interp'] = 20.0

    if 'era5_precip_mm' in df.columns:
        df['era5_precip_interp'] = df.groupby('anon_polygon_id')['era5_precip_mm'].transform(
            lambda s: s.fillna(0.0)
        )
    else:
        df['era5_precip_interp'] = 0.0

    # Агрономические агрегаты: дефицит влаги за 1-2 недели критически влияет на скорость деградации биомассы
    df['precip_sum_7'] = df.groupby('anon_polygon_id')['era5_precip_interp'].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df['precip_sum_14'] = df.groupby('anon_polygon_id')['era5_precip_interp'].transform(
        lambda s: s.rolling(14, min_periods=1).sum()
    )
    df['temp_mean_7'] = df.groupby('anon_polygon_id')['era5_temp_interp'].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return df

def extract_gap_features(df: pd.DataFrame, gap_indices: np.ndarray, poly_clim_df, crop_clim_df, global_clim_df) -> pd.DataFrame:
    """
    Генерация двунаправленных временных и агроклиматических признаков для точек пропуска.
    
    Архитектурная защита: предотвращение утечки данных (Data Leakage) — значения NDVI
    в точках пропуска предварительно заменяются на NaN, признаковое пространство строится
    исключительно на базе фактически доступных наблюдений до и после облачного окна.
    """
    df_work = df.copy()
    # Изоляция целевого признака в окне пропуска для исключения заглядывания вперед
    df_work.loc[gap_indices, 'primary_ndvi'] = np.nan
    
    known = df_work[df_work['primary_ndvi'].notna()].copy()
    
    samples = []
    for poly_id, p_df in df_work.groupby('anon_polygon_id'):
        p_gaps = p_df[p_df.index.isin(gap_indices)]
        if p_gaps.empty:
            continue
        
        p_known = known[known['anon_polygon_id'] == poly_id]
        if p_known.empty:
            continue
            
        p_known_dates = p_known['date_dt'].values
        p_known_vals = p_known['primary_ndvi'].values
        p_known_s2 = p_known['s2_ndvi'].values if 's2_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        p_known_ls = p_known['landsat_ndvi'].values if 'landsat_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        p_known_mod = p_known['modis_ndvi'].values if 'modis_ndvi' in p_known.columns else np.full(len(p_known), np.nan)
        
        evi_s = p_known['s2_evi'] if 's2_evi' in p_known.columns else pd.Series(np.nan, index=p_known.index)
        evi_l = p_known['landsat_evi'] if 'landsat_evi' in p_known.columns else pd.Series(np.nan, index=p_known.index)
        evi_m = p_known['modis_evi'] if 'modis_evi' in p_known.columns else pd.Series(np.nan, index=p_known.index)
        p_known_evi = evi_s.fillna(evi_l).fillna(evi_m).values
        
        ndwi_s = p_known['s2_ndwi'] if 's2_ndwi' in p_known.columns else pd.Series(np.nan, index=p_known.index)
        ndwi_l = p_known['landsat_ndwi'] if 'landsat_ndwi' in p_known.columns else pd.Series(np.nan, index=p_known.index)
        p_known_ndwi = ndwi_s.fillna(ndwi_l).values
        
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
            
            # Расчет линейного базиса по краям окна (y_linear) и весов близости
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
                
            p1_is_s2 = int(not np.isnan(p_known_s2[i_p1])) if has_prev else 0
            p1_is_mod = int(not np.isnan(p_known_mod[i_p1])) if has_prev else 0
            n1_is_s2 = int(not np.isnan(p_known_s2[i_n1])) if has_next else 0
            n1_is_mod = int(not np.isnan(p_known_mod[i_n1])) if has_next else 0
            
            evi_p1 = p_known_evi[i_p1] if has_prev else np.nan
            evi_n1 = p_known_evi[i_n1] if has_next else np.nan
            ndwi_p1 = p_known_ndwi[i_p1] if has_prev else np.nan
            ndwi_n1 = p_known_ndwi[i_n1] if has_next else np.nan
            
            # Наклон тренда (первая производная кривой вегетации до и после пропуска)
            slope_p = (y_p1 - y_p2) / max(1, (dt_p2 - dt_p1)) if has_prev and i_p2 != i_p1 else 0.0
            slope_n = (y_n2 - y_n1) / max(1, (dt_n2 - dt_n1)) if has_next and i_n2 != i_n1 else 0.0
            
            samples.append({
                'index': idx,
                'anon_polygon_id': poly_id,
                'date': str(row['date']),
                'doy': int(pd.to_datetime(row['date_dt']).dayofyear),
                'year': int(pd.to_datetime(row['date_dt']).year),
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
                'era5_temp_interp': row.get('era5_temp_interp', np.nan),
                'era5_precip_interp': row.get('era5_precip_interp', 0.0),
                'precip_sum_7': row.get('precip_sum_7', 0.0),
                'precip_sum_14': row.get('precip_sum_14', 0.0),
                'temp_mean_7': row.get('temp_mean_7', np.nan),
            })
            
    res_df = pd.DataFrame(samples)
    
    # Каскадное слияние климатологий: Полигон -> Культура -> Общемировая норма
    res_df = res_df.merge(poly_clim_df, on=['anon_polygon_id', 'doy'], how='left')
    res_df = res_df.merge(crop_clim_df, on=['crop_type', 'doy'], how='left')
    res_df = res_df.merge(global_clim_df, on='doy', how='left')
    
    res_df['clim_mean'] = res_df['clim_mean'].fillna(res_df['crop_clim_mean']).fillna(res_df['global_clim_mean']).fillna(0.35)
    res_df['clim_std'] = res_df['clim_std'].fillna(res_df['crop_clim_std']).fillna(res_df['global_clim_std']).fillna(0.06)
    
    res_df['y_linear'] = res_df['y_linear'].fillna(res_df['clim_mean'])
    res_df['y_p1'] = res_df['y_p1'].fillna(res_df['clim_mean'])
    res_df['y_n1'] = res_df['y_n1'].fillna(res_df['clim_mean'])
    res_df['y_p2'] = res_df['y_p2'].fillna(res_df['y_p1'])
    res_df['y_n2'] = res_df['y_n2'].fillna(res_df['y_n1'])
    
    # Агроклиматические аномалии и направляющий априорный прогноз
    res_df['y_linear_clim_diff'] = res_df['y_linear'] - res_df['clim_mean']
    res_df['p1_clim_diff'] = res_df['y_p1'] - res_df['clim_mean']
    res_df['n1_clim_diff'] = res_df['y_n1'] - res_df['clim_mean']
    res_df['interp_anom'] = res_df['weight_p'] * res_df['p1_clim_diff'] + res_df['weight_n'] * res_df['n1_clim_diff']
    res_df['clim_guided_pred'] = res_df['clim_mean'] + res_df['interp_anom']
    
    # Циклические признаки дня года для учета фенологической периодичности
    res_df['sin_doy'] = np.sin(2 * np.pi * res_df['doy'] / 365.25)
    res_df['cos_doy'] = np.cos(2 * np.pi * res_df['doy'] / 365.25)
    
    res_df['crop_type'] = res_df['crop_type'].astype('category')
    res_df['anon_polygon_id'] = res_df['anon_polygon_id'].astype('category')
    
    return res_df

FEATURE_COLUMNS = [
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
