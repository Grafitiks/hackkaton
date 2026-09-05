import os
import pandas as pd
import numpy as np

def compute_climatology(df: pd.DataFrame, rolling_window: int = 11) -> pd.DataFrame:
    """
    Вычисляет сглаженную многолетнюю суточную климатологию по дням года (DOY):
    среднее (clim_mean), стандартное отклонение (clim_std) и медиану (clim_median).
    Использует иерархическую структуру:
    1. Индивидуальный профиль полигона (poly_clim);
    2. Профиль культуры (crop_doy - фоллбэк для новых участков);
    3. Глобальный климатический профиль сезона (global_doy).
    """
    valid = df[df['primary_ndvi'].notna()].copy()
    
    # 1. Агрегация климатологии на уровне каждого полигона по DOY
    poly_doy = valid.groupby(['anon_polygon_id', 'doy'])['primary_ndvi'].agg(
        clim_mean='mean',
        clim_std='std',
        clim_median='median',
        clim_count='count'
    ).reset_index()
    
    # Выравнивание и непрерывное скользящее сглаживание профиля DOY
    polygons = df['anon_polygon_id'].unique()
    all_doys = pd.DataFrame([
        {'anon_polygon_id': p, 'doy': d}
        for p in polygons for d in range(91, 305)
    ])
    merged = all_doys.merge(poly_doy, on=['anon_polygon_id', 'doy'], how='left')
    
    smoothed = []
    for poly, group in merged.groupby('anon_polygon_id'):
        g = group.sort_values('doy').copy()
        g['clim_mean'] = g['clim_mean'].interpolate(method='linear', limit_direction='both')
        g['clim_mean'] = g['clim_mean'].rolling(rolling_window, center=True, min_periods=1).mean()
        
        g['clim_std'] = g['clim_std'].interpolate(method='linear', limit_direction='both')
        g['clim_std'] = g['clim_std'].rolling(rolling_window, center=True, min_periods=1).mean().fillna(0.06)
        
        g['clim_median'] = g['clim_median'].interpolate(method='linear', limit_direction='both')
        g['clim_median'] = g['clim_median'].rolling(rolling_window, center=True, min_periods=1).mean()
        smoothed.append(g[['anon_polygon_id', 'doy', 'clim_mean', 'clim_std', 'clim_median']])
        
    poly_clim_df = pd.concat(smoothed, ignore_index=True)
    
    # 2. Климатология по типу с/х культуры (фоллбэк для новых участков без истории)
    crop_doy = valid.groupby(['crop_type', 'doy'])['primary_ndvi'].agg(
        crop_clim_mean='mean',
        crop_clim_std='std',
        crop_clim_median='median'
    ).reset_index()
    
    # 3. Базовая общерегиональная климатология DOY
    global_doy = valid.groupby('doy')['primary_ndvi'].agg(
        global_clim_mean='mean',
        global_clim_std='std',
        global_clim_median='median'
    ).reset_index()
    
    return poly_clim_df, crop_doy, global_doy
