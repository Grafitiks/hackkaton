import numpy as np
import pandas as pd
from typing import Dict, List, Any

def compute_zscores(fact_ndvi: pd.Series, clim_mean: pd.Series, clim_std: pd.Series) -> pd.Series:
    """
    Вычисляет стандартизованное Z-score отклонение факта от многолетней нормы:
    Z = (NDVI_fact - NDVI_clim_mean) / NDVI_clim_std
    """
    # Защита от деления на ноль при вырожденной дисперсии климатологии
    safe_std = clim_std.replace(0, np.nan).fillna(0.06)
    z = (fact_ndvi - clim_mean) / safe_std
    return z

def classify_status(z_score: float) -> str:
    """
    Стандартизованная классификация состояния вегетации по регламенту хакатона:
    - Z >= -1.0: 'Штатное развитие'
    - -2.0 <= Z < -1.0: 'Угнетение биомассы'
    - Z < -2.0: 'Критическая аномалия'
    """
    if pd.isna(z_score):
        return "Не определено"
    if z_score >= -1.0:
        return "Штатное развитие"
    elif z_score >= -2.0:
        return "Угнетение биомассы"
    else:
        return "Критическая аномалия"

def detect_anomaly_intervals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Агрегирует непрерывные периоды отрицательных отклонений (стресс и критические аномалии)
    в единые временные окна для последующего факторного агрономического анализа.
    """
    df = df.sort_values('date').copy()
    anomalies = []
    current_interval = None
    
    for idx, row in df.iterrows():
        status = row.get('status', 'Штатное развитие')
        z = row.get('ndvi_zscore', 0.0)
        date = str(row['date'])
        
        if status in ['Угнетение биомассы', 'Критическая аномалия']:
            if current_interval is None:
                current_interval = {
                    'start_date': date,
                    'end_date': date,
                    'status': status,
                    'min_zscore': z,
                    'days_count': 1,
                    'records': [row]
                }
            else:
                current_interval['end_date'] = date
                current_interval['days_count'] += 1
                current_interval['records'].append(row)
                if z < current_interval['min_zscore']:
                    current_interval['min_zscore'] = z
                if status == 'Критическая аномалия':
                    current_interval['status'] = 'Критическая аномалия'
        else:
            if current_interval is not None:
                anomalies.append(current_interval)
                current_interval = None
                
    if current_interval is not None:
        anomalies.append(current_interval)
        
    return anomalies
