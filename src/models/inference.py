import os
import sys
import pickle
import pandas as pd
import numpy as np

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.feature_builder import enrich_weather, extract_gap_features, FEATURE_COLUMNS

def predict_gaps(test_df_path: str, model_artifact_path: str = "artifacts/models/ensemble_models.pkl", clim_artifact_path: str = "artifacts/models/climatology.pkl") -> pd.DataFrame:
    """
    Пакетный инференс ансамбля градиентного бустинга (LightGBM + CatBoost)
    для высокоточного восстановления пропусков NDVI на тестовых полигонах.
    """
    print(f"[Инференс] Загрузка тестового датасета: {test_df_path}...")
    df_test = pd.read_csv(test_df_path, encoding='utf-8')
    df_test['date_dt'] = pd.to_datetime(df_test['date'])
    df_test['year'] = df_test['date_dt'].dt.year
    df_test = df_test.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)
    
    # Выделение строк искусственных и облачных пропусков (is_synthetic_gap)
    gap_rows = df_test[df_test['is_synthetic_gap'] == True]
    print(f"[Инференс] Обнаружено целевых точек пропусков: {len(gap_rows)}.")
    gap_indices = gap_rows.index.values
    
    # Загрузка иерархических климатических профилей
    print(f"[Инференс] Загрузка климатических артефактов: {clim_artifact_path}...")
    with open(clim_artifact_path, "rb") as f:
        clim_data = pickle.load(f)
    poly_clim = clim_data['poly_clim']
    crop_clim = clim_data['crop_clim']
    global_clim = clim_data['global_clim']
    
    # Обогащение агрометеорологическими параметрами ERA5-Land
    print("[Инференс] Обогащение метеопараметрами...")
    df_test = enrich_weather(df_test)
    
    # Формирование признакового пространства без утечки данных
    print("[Инференс] Генерация двунаправленных признаков...")
    gap_features = extract_gap_features(df_test, gap_indices, poly_clim, crop_clim, global_clim)
    gap_features = gap_features.sort_values('index').reset_index(drop=True)
    
    y_linear = gap_features['y_linear'].values
    
    # Загрузка ансамблевых моделей
    print(f"[Инференс] Загрузка моделей бустинга: {model_artifact_path}...")
    with open(model_artifact_path, "rb") as f:
        bundle = pickle.load(f)
    lgb_models = bundle['lgb_models']
    cat_models = bundle['cat_models']
    
    X = gap_features[FEATURE_COLUMNS].copy()
    for c in ['crop_type']:
        X[c] = X[c].astype('category')
        
    X_cat = X.copy()
    for c in ['crop_type']:
        X_cat[c] = X_cat[c].astype(str)
        
    print(f"[Инференс] Прогнозирование ансамблем: {len(lgb_models)} LightGBM + {len(cat_models)} CatBoost...")
    preds_lgb = np.zeros(len(X))
    for m in lgb_models:
        preds_lgb += m.predict(X) / len(lgb_models)
        
    preds_cat = np.zeros(len(X))
    for m in cat_models:
        preds_cat += m.predict(X_cat) / len(cat_models)
        
    # Блендинг ансамбля (60% LightGBM + 40% CatBoost) и сложение с линейным базисом
    delta_pred = 0.6 * preds_lgb + 0.4 * preds_cat
    primary_ndvi_pred = np.clip(y_linear + delta_pred, -0.2, 1.0)
    
    # Формирование финального датафрейма для отправки решения (submission)
    sub = pd.DataFrame({
        'anon_polygon_id': gap_features['anon_polygon_id'].astype(str),
        'date': gap_features['date'].astype(str),
        'primary_ndvi_pred': primary_ndvi_pred
    })
    
    return sub
