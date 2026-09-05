import os
import sys
import datetime
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')
project_root = r"c:\Users\Артем\Desktop\hackkaton"
os.chdir(project_root)
sys.path.append(project_root)

from src.models.inference import predict_gaps
from src.features.feature_builder import (
    enrich_weather, clean_anchor_series, extract_gap_features, FEATURE_COLUMNS
)

def run():
    print("=" * 80)
    print("ПРОВЕРКА НОВОЙ МОДЕЛИ НА ДАТАСЕТЕ test_features (1).csv (ЗАДАЧА 1)")
    print("=" * 80)

    raw_path = r"test_features (1).csv"
    if not os.path.exists(raw_path):
        raw_path = r"C:\Users\Артем\Downloads\Telegram Desktop\test_features (1).csv"
    if not os.path.exists(raw_path):
        print(f"Файл {raw_path} не найден!")
        return

    # 1. Запуск инференса на целевых пропусках (is_synthetic_gap == True)
    print(f"\n1. Выполнение прогноза для целевых пропусков (is_synthetic_gap == True)...")
    sub = predict_gaps(raw_path)
    print(f"[OK] Сформировано {len(sub)} предсказаний.")

    # 2. Сопоставление с метаданными
    df_raw = pd.read_csv(raw_path, encoding='utf-8')
    gaps_meta = df_raw[df_raw['is_synthetic_gap'] == True].copy()
    gaps_meta['date_str'] = gaps_meta['date'].astype(str)
    sub['date_str'] = sub['date'].astype(str)

    merged = pd.merge(
        sub,
        gaps_meta[['anon_polygon_id', 'date_str', 'crop_type']],
        on=['anon_polygon_id', 'date_str'],
        how='left'
    )
    dt_series = pd.to_datetime(merged['date'])
    merged['year'] = dt_series.dt.year
    merged['doy'] = dt_series.dt.dayofyear
    merged['month'] = dt_series.dt.month

    # Сохранение файлов предсказаний
    for s_name in ["test_featured(1)_predictions.csv", "test_featured_predictions.csv"]:
        sub[['anon_polygon_id', 'date', 'primary_ndvi_pred']].to_csv(s_name, index=False, encoding='utf-8')
    for d_name in ["test_featured(1)_predictions_detailed.csv", "test_featured_predictions_detailed.csv"]:
        merged[['anon_polygon_id', 'date', 'crop_type', 'year', 'month', 'doy', 'primary_ndvi_pred']].to_csv(
            d_name, index=False, encoding='utf-8'
        )
    print(f"[OK] Файлы предсказаний успешно сохранены.")

    # 3. Валидационная оценка качества (RMSE / GapScore) на ground-truth точках датасета
    print(f"\n2. Валидационная оценка качества (RMSE / GapScore) на ground-truth точках...")
    df_val = df_raw.copy()
    df_val['date_dt'] = pd.to_datetime(df_val['date'])
    df_val['year'] = df_val['date_dt'].dt.year
    df_val = df_val.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

    known_pts = df_val[df_val['primary_ndvi'].notna()].copy()
    print(f"Всего доступно фактических наземных измерений: {len(known_pts):,} строк.")
    
    # Сэмплируем ровно те же 2500 контрольных точек (seed=42) для объективного сравнения
    np.random.seed(42)
    val_gap_idx = np.random.choice(known_pts.index.values, size=min(2500, len(known_pts)), replace=False)
    
    # Загружаем климатологию и обученный ансамбль
    with open("artifacts/models/climatology.pkl", "rb") as f:
        clim = pickle.load(f)
    with open("artifacts/models/ensemble_models.pkl", "rb") as f:
        bundle = pickle.load(f)
        
    df_val_enriched = enrich_weather(df_val)
    val_features = extract_gap_features(
        df_val_enriched,
        gap_indices=val_gap_idx,
        poly_clim_df=clim['poly_clim'],
        crop_clim_df=clim['crop_clim'],
        global_clim_df=clim['global_clim'],
        date_sat_stats=bundle.get('date_sat_stats', None),
        date_crop_stats=bundle.get('date_crop_stats', None)
    )
    val_features = val_features.sort_values('index').reset_index(drop=True)
    
    y_val_true = df_val.loc[val_features['index'], 'primary_ndvi'].values
    y_val_linear = val_features['y_linear'].values
    
    # Оценка исходного линейного бейзлайна (без калибровки)
    rmse_linear = np.sqrt(mean_squared_error(y_val_true, y_val_linear))
    mae_linear = mean_absolute_error(y_val_true, y_val_linear)
    score_linear = round(float(30 * max(0, 1 - rmse_linear / 0.10)), 2)
    
    # Прогон нового ансамбля
    feature_cols = bundle.get('feature_cols', FEATURE_COLUMNS)
    X_val = val_features[feature_cols].copy()
    X_val['crop_type'] = X_val['crop_type'].astype('category')
    
    X_val_cat = X_val.copy()
    X_val_cat['crop_type'] = X_val_cat['crop_type'].astype(str)
    
    p_lgb = np.zeros(len(X_val))
    for m in bundle['lgb_models']:
        p_lgb += m.predict(X_val) / len(bundle['lgb_models'])
        
    p_cb = np.zeros(len(X_val))
    for m in bundle['cat_models']:
        p_cb += m.predict(X_val_cat) / len(bundle['cat_models'])
        
    w_lgb = bundle.get('lgb_weight', 0.6)
    w_cat = bundle.get('cat_weight', 0.4)
    delta_pred = w_lgb * p_lgb + w_cat * p_cb
    y_val_pred = np.clip(y_val_linear + delta_pred, -0.05, 0.98)
    
    rmse_model = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    mae_model = mean_absolute_error(y_val_true, y_val_pred)
    score_model = round(float(30 * max(0, 1 - rmse_model / 0.10)), 2)
    
    print(f"\nСравнение результатов на контрольных точках:")
    print(f"  • Линейный бейзлайн:          RMSE = {rmse_linear:.5f} | MAE = {mae_linear:.5f} | GapScore = {score_linear:5.2f} / 30")
    print(f"  • Предыдущая модель:          RMSE = 0.08617 | MAE = 0.05704 | GapScore =  4.15 / 30")
    print(f"  • НАША НОВАЯ МОДЕЛЬ:          RMSE = {rmse_model:.5f} | MAE = {mae_model:.5f} | GapScore = {score_model:5.2f} / 30")
    print(f"  • Прирост к предыдущей модели: +{score_model - 4.15:.2f} баллов (+{(1 - rmse_model/0.08617)*100:.1f}% снижения RMSE)")

    # 4. Формирование логов
    for log_path in ["test_featured(1)_log.txt", "test_featured_log.txt"]:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 95 + "\n")
            f.write("КОСМОХАКАТОН // ВАЛИДАЦИОННЫЙ ЛОГ МОДЕЛИ ВОССТАНОВЛЕНИЯ ПРОПУСКОВ (ЗАДАЧА 1)\n")
            f.write("=" * 95 + "\n")
            f.write(f"Дата и время формирования: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Входной тестовый датасет:   {raw_path}\n")
            f.write(f"Всего строк в датасете:     {len(df_raw):,}\n")
            f.write(f"Количество полигонов:       {df_raw['anon_polygon_id'].nunique()} полей\n")
            f.write(f"Целевых пропусков:          {len(sub):,} контрольных точек (is_synthetic_gap == True)\n")
            f.write(f"Архитектура модели:         GroupKFold Ансамбль 5-Fold LightGBM (Huber) + 5-Fold CatBoost\n")
            f.write(f"Ключевые инновации:         Синхронизация орбит спутников, очистка теней, агрофенология культур\n")
            f.write("-" * 95 + "\n\n")

            f.write("1. ВАЛИДАЦИОННАЯ ТОЧНОСТЬ МОДЕЛИ НА GROUND-TRUTH ТОЧКАХ:\n")
            f.write("-" * 95 + "\n")
            f.write(f"  • Линейная интерполяция (бейзлайн): RMSE = {rmse_linear:.5f} | MAE = {mae_linear:.5f} | GapScore = {score_linear:5.2f} / 30\n")
            f.write(f"  • Предыдущая версия модели:         RMSE = 0.08617 | MAE = 0.05704 | GapScore =  4.15 / 30\n")
            f.write(f"  • НАША НОВАЯ МОДЕЛЬ (ТЕКУЩАЯ):      RMSE = {rmse_model:.5f} | MAE = {mae_model:.5f} | GapScore = {score_model:5.2f} / 30\n")
            f.write(f"  • Прирост относительно прошлой:     +{score_model - 4.15:.2f} баллов (+{(1 - rmse_model/0.08617)*100:.1f}% точности)\n")
            f.write(f"  • Прирост относительно бейзлайна:   +{score_model - score_linear:.2f} баллов\n\n")

            f.write("2. СВОДНАЯ СТАТИСТИКА СФОРМИРОВАННЫХ ПРЕДСКАЗАНИЙ (primary_ndvi_pred):\n")
            f.write("-" * 95 + "\n")
            f.write(f"  • Всего ответов:                     {len(sub):,}\n")
            f.write(f"  • Среднее значение (Mean):           {merged['primary_ndvi_pred'].mean():.6f}\n")
            f.write(f"  • Стандартное отклонение (Std):      {merged['primary_ndvi_pred'].std():.6f}\n")
            f.write(f"  • Минимум (Min):                     {merged['primary_ndvi_pred'].min():.6f}\n")
            f.write(f"  • 25% квантиль (Q1):                 {merged['primary_ndvi_pred'].quantile(0.25):.6f}\n")
            f.write(f"  • Медиана (50% Median):              {merged['primary_ndvi_pred'].median():.6f}\n")
            f.write(f"  • 75% квантиль (Q3):                 {merged['primary_ndvi_pred'].quantile(0.75):.6f}\n")
            f.write(f"  • Максимум (Max):                    {merged['primary_ndvi_pred'].max():.6f}\n\n")

            f.write("3. СТАТИСТИКА ПРОГНОЗОВ ПО СЕЛЬСКОХОЗЯЙСТВЕННЫМ КУЛЬТУРАМ:\n")
            f.write("-" * 95 + "\n")
            f.write(f"{'Культура':<25} | {'Кол-во точек':<14} | {'Mean NDVI':<12} | {'Min NDVI':<10} | {'Max NDVI':<10}\n")
            f.write("-" * 95 + "\n")
            for crop, grp in merged.groupby('crop_type'):
                f.write(f"{crop:<25} | {len(grp):<14} | {grp['primary_ndvi_pred'].mean():<12.4f} | {grp['primary_ndvi_pred'].min():<10.4f} | {grp['primary_ndvi_pred'].max():<10.4f}\n")
            f.write("-" * 95 + "\n\n")

            f.write("4. СТАТИСТИКА ПО ПОЛИГОНАМ (ВСЕ 20 ПОЛЕЙ):\n")
            f.write("-" * 95 + "\n")
            f.write(f"{'Полигон':<12} | {'Культура':<20} | {'Точек':<8} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}\n")
            f.write("-" * 95 + "\n")
            for pid, grp in merged.groupby('anon_polygon_id'):
                c = grp['crop_type'].iloc[0] if 'crop_type' in grp.columns else "н/д"
                f.write(f"{pid:<12} | {c:<20} | {len(grp):<8} | {grp['primary_ndvi_pred'].mean():<8.4f} | {grp['primary_ndvi_pred'].std():<8.4f} | {grp['primary_ndvi_pred'].min():<8.4f} | {grp['primary_ndvi_pred'].max():<8.4f}\n")
            f.write("-" * 95 + "\n\n")

            f.write("5. ПРИМЕРЫ СФОРМИРОВАННЫХ ПРЕДСКАЗАНИЙ (ПЕРВЫЕ 25 СТРОК):\n")
            f.write("-" * 95 + "\n")
            f.write(f"{'№':<5} | {'Полигон':<12} | {'Дата':<12} | {'Культура':<20} | {'DOY':<5} | {'primary_ndvi_pred':<18}\n")
            f.write("-" * 95 + "\n")
            for i, (_, row) in enumerate(merged.head(25).iterrows()):
                f.write(f"{i+1:<5} | {row['anon_polygon_id']:<12} | {row['date']:<12} | {str(row['crop_type']):<20} | {row['doy']:<5} | {row['primary_ndvi_pred']:<18.6f}\n")
            f.write("-" * 95 + "\n")
            f.write("\n=== КОНЕЦ ЛОГА ===\n")

    print(f"[OK] Логи сохранены: test_featured(1)_log.txt и test_featured_log.txt")

if __name__ == "__main__":
    run()
