import os
import sys
import json
import argparse
import pickle
import pandas as pd
import numpy as np

# Настройка вывода UTF-8 для корректного отображения логов в консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Обеспечение импорта локальных модулей проекта
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.anomalies.detector import compute_zscores, classify_status, detect_anomaly_intervals
from src.anomalies.interpreter import generate_full_report

def main():
    parser = argparse.ArgumentParser(description="GEO-VEGA // Консольный запуск детекции и агрономической интерпретации аномалий (Задача 2)")
    
    candidates = [
        "data/train_dataset.csv",
        "train_dataset.csv",
        "data/test_features (1).csv",
        "data/private_features.csv"
    ]
    detected_input = next((c for c in candidates if os.path.exists(c)), "data/train_dataset.csv")
    
    parser.add_argument("--input", default=detected_input, help="Путь к датасету наблюдений (CSV)")
    parser.add_argument("--polygon-id", default=None, help="Идентификатор конкретного полигона (если не задан — анализ всех)")
    parser.add_argument("--clim", default="artifacts/models/climatology.pkl", help="Путь к файлу многолетней климатологии")
    parser.add_argument("--output", default="anomalies_report.json", help="Путь для сохранения итогового отчета в формате JSON")
    parser.add_argument("--limit-print", type=int, default=15, help="Максимальное количество аномалий для вывода в консоль")
    
    args = parser.parse_args()
    
    print("============================================================================")
    print("GEO-VEGA: ДЕТЕКЦИЯ И АГРОНОМИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ АНОМАЛИЙ (ЗАДАЧА 2)")
    print(f"Входной датасет:  {args.input}")
    print(f"Климатология:     {args.clim}")
    print(f"Выходной отчет:   {args.output}")
    print("============================================================================")
    
    if not os.path.exists(args.input):
        print(f"[ОШИБКА] Файл {args.input} не найден!")
        sys.exit(1)
        
    print("\n[1/4] Загрузка датасета и нормализация дат...")
    df = pd.read_csv(args.input, encoding='utf-8')
    df['date_dt'] = pd.to_datetime(df['date'])
    df['doy'] = df['date_dt'].dt.dayofyear
    
    # Фильтрация по полигону при необходимости
    poly_col = 'anon_polygon_id' if 'anon_polygon_id' in df.columns else 'polygon_id'
    if args.polygon_id:
        df = df[df[poly_col] == args.polygon_id].copy()
        if df.empty:
            print(f"[ОШИБКА] Полигон {args.polygon_id} не найден в датасете!")
            sys.exit(1)
        print(f"  Анализируется полигон: {args.polygon_id} ({len(df)} наблюдений)")
    else:
        polygons_count = df[poly_col].nunique()
        print(f"  Анализируется полигонов: {polygons_count} ({len(df)} суммарных наблюдений)")
        
    print("\n[2/4] Загрузка климатических профилей и расчет Z-Score отклонений...")
    if os.path.exists(args.clim):
        with open(args.clim, "rb") as f:
            clim_data = pickle.load(f)
        poly_clim = clim_data.get('poly_clim', pd.DataFrame())
        crop_clim = clim_data.get('crop_clim', pd.DataFrame())
        global_clim = clim_data.get('global_clim', pd.DataFrame())
    else:
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Файл {args.clim} не найден. Расчет скользящей локальной нормы...")
        poly_clim, crop_clim, global_clim = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # Сопоставление со сглаженной климатической нормой
    if not poly_clim.empty and 'anon_polygon_id' in df.columns:
        df = df.merge(poly_clim[['anon_polygon_id', 'doy', 'poly_clim_mean', 'poly_clim_std']], on=['anon_polygon_id', 'doy'], how='left')
        df['clim_mean'] = df['poly_clim_mean']
        df['clim_std'] = df['poly_clim_std']
    elif not crop_clim.empty and 'crop_type' in df.columns:
        df = df.merge(crop_clim[['crop_type', 'doy', 'crop_clim_mean', 'crop_clim_std']], on=['crop_type', 'doy'], how='left')
        df['clim_mean'] = df['crop_clim_mean']
        df['clim_std'] = df['crop_clim_std']
    else:
        df['clim_mean'] = df.groupby(poly_col)['primary_ndvi'].transform(lambda s: s.rolling(15, min_periods=1, center=True).mean())
        df['clim_std'] = 0.08
        
    df['clim_mean'] = df['clim_mean'].fillna(0.35)
    df['clim_std'] = df['clim_std'].fillna(0.06)
    
    # Целевой индекс NDVI: если primary_ndvi пуст, используем первичные сенсоры
    val_series = df['primary_ndvi'] if 'primary_ndvi' in df.columns else df['primary_ndvi_pred']
    df['ndvi_zscore'] = compute_zscores(val_series, df['clim_mean'], df['clim_std'])
    df['status'] = df['ndvi_zscore'].apply(classify_status)
    
    status_counts = df['status'].value_counts().to_dict()
    print("  Распределение статусов вегетации:")
    for st, count in status_counts.items():
        pct = (count / len(df)) * 100
        print(f"    • {st}: {count} дней ({pct:.1f}%)")
        
    print("\n[3/4] Кластеризация стрессовых интервалов и факторный анализ причин (ERA5, NDWI)...")
    all_anomalies = []
    for p_id, p_group in df.groupby(poly_col):
        p_intervals = detect_anomaly_intervals(p_group)
        p_report = generate_full_report(p_intervals)
        for anom in p_report:
            anom['polygon_id'] = p_id
            all_anomalies.append(anom)
            
    print(f"  Всего выявлено связанных стрессовых периодов: {len(all_anomalies)}")
    
    # Вывод сводной таблицы топ-аномалий в консоль
    print("\n[4/4] Сводный реестр зафиксированных аномалий:")
    print("-" * 115)
    print(f"{'Полигон':<12} | {'Период':<23} | {'Дней':<5} | {'Статус':<22} | {'Мин. Z':<7} | {'Первопричина'}")
    print("-" * 115)
    
    for anom in all_anomalies[:args.limit_print]:
        p_id = str(anom.get('polygon_id', 'AOI'))[:11]
        period = f"{anom['start_date']} .. {anom['end_date']}"
        days = str(anom['duration_days'])
        status = str(anom['status'])[:21]
        min_z = f"{anom['min_zscore']:.2f} σ"
        cause = str(anom.get('primary_cause', 'Недостаток влаги'))[:35]
        print(f"{p_id:<12} | {period:<23} | {days:<5} | {status:<22} | {min_z:<7} | {cause}")
        
    if len(all_anomalies) > args.limit_print:
        print(f"... и еще {len(all_anomalies) - args.limit_print} выявленных аномальных интервалов.")
    print("-" * 115)
    
    # Сохранение полного JSON отчета
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "status_counts": status_counts,
            "total_anomalies": len(all_anomalies),
            "anomalies": all_anomalies
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\n[УСПЕХ] Полный структурированный отчет сохранен в файл: {args.output}")
    print("Для интерактивного визуального анализа на карте запустите веб-интерфейс:")
    print("  python run_server.py")

if __name__ == "__main__":
    main()
