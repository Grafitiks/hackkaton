from typing import Dict, Any, List
import numpy as np
import pandas as pd

def interpret_anomaly(anomaly_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Генерирует агрономическую интерпретацию и причинно-следственное объяснение
    выявленного периода аномалии на основе метеоконтекста ERA5 (температура, осадки)
    и водного стресса по индексу NDWI.
    """
    records = pd.DataFrame(anomaly_record['records'])
    status = anomaly_record['status']
    duration = anomaly_record['days_count']
    min_z = round(float(anomaly_record['min_zscore']), 2)
    
    # Расчет погодных метрик за период аномалии
    avg_temp = float(records['era5_temp_c'].mean()) if 'era5_temp_c' in records.columns else 20.0
    total_precip = float(records['era5_precip_mm'].sum()) if 'era5_precip_mm' in records.columns else 0.0
    max_temp = float(records['era5_temp_c'].max()) if 'era5_temp_c' in records.columns else 20.0
    
    # Проверка индекса влагосодержания листьев NDWI
    ndwi_cols = [c for c in ['s2_ndwi', 'landsat_ndwi'] if c in records.columns]
    avg_ndwi = None
    for col in ndwi_cols:
        val = records[col].dropna().mean()
        if not pd.isna(val):
            avg_ndwi = float(val)
            break
            
    # Правила экспертной диагностики факторов стресса посевов
    causes = []
    recommendations = []
    
    # 1. Термический стресс и тепловой шок
    if max_temp >= 32.0 or avg_temp >= 28.0:
        causes.append(f"Экстремальный температурный стресс (пик {max_temp:.1f}°C, средняя {avg_temp:.1f}°C)")
        recommendations.append("Провести антистрессовую некорневую обработку аминокислотами и адаптогенами.")
    elif avg_temp >= 25.0:
        causes.append(f"Повышенный температурный фон ({avg_temp:.1f}°C)")
        
    # 2. Дефицит влаги и засуха
    daily_precip = total_precip / max(1, duration)
    if total_precip < 5.0 and duration >= 7:
        causes.append(f"Острая засуха: за {duration} дн. выпало всего {total_precip:.1f} мм осадков")
        recommendations.append("Контроль влагозапасов в корнеобитаемом слое почвы; при наличии мелиорации — проведение освежительного полива.")
    elif total_precip < 15.0 and duration >= 14:
        causes.append(f"Умеренный дефицит продуктивной влаги ({total_precip:.1f} мм за {duration} дн.)")
        
    # 3. Водный стресс растительности по NDWI
    if avg_ndwi is not None and avg_ndwi < -0.2:
        causes.append(f"Снижение тургора и влагосодержания листьев (средний NDWI: {avg_ndwi:.2f})")
        
    # 4. Классификация по продолжительности: кратковременный сбой vs системная деградация
    if duration <= 3:
        severity = "Кратковременный сбой"
        nature = "Возможно временное влияние облачности, краевых теней или прохождения суховея."
    elif duration <= 10:
        severity = "Локальное подавление вегетации"
        nature = "Умеренная фазовая задержка роста или погодный стресс."
    else:
        severity = "Длительная деградация посевов"
        nature = "Серьезное угнетение биомассы, угрожающее существенным снижением урожайности."
        
    if not causes:
        causes.append("Отклонение вегетационного индекса от климатической нормы для данной фазы сезона.")
        
    if not recommendations:
        recommendations.append("Регулярный спутниковый мониторинг динамики вегетации и наземное обследование участка (фитосанитарный контроль).")
        
    primary_cause = "; ".join(causes)
    rec_text = " ".join(recommendations)
    
    return {
        'start_date': anomaly_record['start_date'],
        'end_date': anomaly_record['end_date'],
        'duration_days': duration,
        'status': status,
        'min_zscore': min_z,
        'avg_temp': round(avg_temp, 1),
        'total_precip': round(total_precip, 1),
        'severity': severity,
        'primary_cause': primary_cause,
        'nature': nature,
        'recommendation': rec_text
    }

def generate_full_report(anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [interpret_anomaly(a) for a in anomalies]
