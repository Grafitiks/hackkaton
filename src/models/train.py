import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

# Обеспечение корректного импорта локальных модулей проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.feature_builder import (
    enrich_weather, clean_anchor_series, extract_gap_features, FEATURE_COLUMNS,
    to_s2_space, from_s2_space
)

def build_date_statistics(df_all: pd.DataFrame):
    """
    Формирование глобального календаря пролетов спутников и региональной агродинамики культур.
    """
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
    date_sat_stats['inferred_sensor'] = np.where(
        date_sat_stats['has_s2'] == 1, 0,
        np.where(date_sat_stats['has_ls'] == 1, 1, 2)
    )

    known = df_all[df_all['primary_ndvi'].notna()]
    date_crop_stats = known.groupby(['crop_type', 'date']).agg(
        crop_tot_obs=('primary_ndvi', 'count'),
        crop_mean_obs=('primary_ndvi', 'mean'),
        crop_std_obs=('primary_ndvi', 'std')
    ).reset_index()
    date_crop_stats['crop_std_obs'] = date_crop_stats['crop_std_obs'].fillna(0.0)

    return date_sat_stats, date_crop_stats

def run_training():
    print("================================================================================")
    print("ЗАПУСК ОБУЧЕНИЯ ФИНАЛЬНОГО АНСАМБЛЯ ВОССТАНОВЛЕНИЯ ПРОПУСКОВ (GAP RESTORATION)")
    print("================================================================================")
    os.makedirs("artifacts/models", exist_ok=True)
    
    # 1. Загрузка данных для построения календаря пролётов
    print("1. Загрузка датасетов...")
    train_path = "data/train_dataset.csv"
    df_train = pd.read_csv(train_path, encoding='utf-8')
    df_train['date_dt'] = pd.to_datetime(df_train['date'])
    df_train['year'] = df_train['date_dt'].dt.year
    df_train = df_train.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)
    
    dfs_for_calendar = [df_train]
    for extra_p in ["data/private_features.csv", "data/test_features.csv", "data/test_features (1).csv", "test_features.csv"]:
        if os.path.exists(extra_p):
            ex_df = pd.read_csv(extra_p, encoding='utf-8')
            ex_df['date_dt'] = pd.to_datetime(ex_df['date'])
            ex_df['year'] = ex_df['date_dt'].dt.year
            dfs_for_calendar.append(ex_df)
    df_all = pd.concat(dfs_for_calendar, ignore_index=True).drop_duplicates(subset=['anon_polygon_id', 'date'])
    print(f"Массив наблюдений: {len(df_train):,} строк train, суммарно для календаря {len(df_all):,} строк на {df_all['anon_polygon_id'].nunique()} полях.")
        
    # 2. Обогащение метеоданными ERA5-Land
    print("\n2. Агрометеорологическое обогащение временных рядов...")
    df_train = enrich_weather(df_train)
    df_all = enrich_weather(df_all)
    
    # 3. Расписание спутниковых пролетов и региональная динамика культур
    print("3. Формирование календаря орбит спутников и динамики культур...")
    date_sat_stats, date_crop_stats = build_date_statistics(df_all)
    
    # 4. Трансдуктивная многомасштабная климатология (Полигон -> Культура -> Общемировая)
    print("4. Расчет фенологических климатических огибающих...")
    known_all = df_all[df_all['primary_ndvi'].notna()].copy()
    
    poly_clim = known_all.groupby(['anon_polygon_id', 'doy'])['primary_ndvi'].agg(['mean', 'std']).reset_index()
    poly_clim.columns = ['anon_polygon_id', 'doy', 'poly_clim_mean', 'poly_clim_std']

    crop_clim = known_all.groupby(['crop_type', 'doy'])['primary_ndvi'].agg(['mean', 'std']).reset_index()
    crop_clim.columns = ['crop_type', 'doy', 'crop_clim_mean', 'crop_clim_std']

    global_clim = known_all.groupby('doy')['primary_ndvi'].agg(['mean', 'std']).reset_index()
    global_clim.columns = ['doy', 'global_clim_mean', 'global_clim_std']

    # Центрированное сглаживание окном 11 дней
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

    global_clim['clim_slope'] = (global_clim['global_clim_mean'].shift(-2) - global_clim['global_clim_mean'].shift(2)) / 4.0
    global_clim['clim_slope'] = global_clim['clim_slope'].bfill().ffill()

    # Сохранение климатологии
    with open("artifacts/models/climatology.pkl", "wb") as f:
        pickle.dump({
            'poly_clim': poly_clim,
            'crop_clim': crop_clim,
            'global_clim': global_clim
        }, f)
    print("[OK] Климатические профили сохранены в artifacts/models/climatology.pkl")
    
    # 5. Генерация обучающих выборок с реалистичной структурой окон
    print("\n5. Формирование обучающей выборки синтетических пропусков...")
    known_train = df_train[df_train['primary_ndvi'].notna()].copy()
    
    interior_indices = []
    for _, grp in known_train.groupby('anon_polygon_id'):
        if len(grp) > 2:
            interior_indices.extend(grp.index.values[1:-1])
            
    np.random.seed(42)
    sample_size = min(24000, len(interior_indices))
    sampled_indices = np.random.choice(interior_indices, size=sample_size, replace=False)
    print(f"Сформировано {len(sampled_indices):,} обучающих пропусков на 39 полях.")
    
    train_features = extract_gap_features(
        df_train,
        gap_indices=sampled_indices,
        poly_clim_df=poly_clim,
        crop_clim_df=crop_clim,
        global_clim_df=global_clim,
        date_sat_stats=date_sat_stats,
        date_crop_stats=date_crop_stats,
        is_train=True
    )
    train_features = train_features.sort_values('index').reset_index(drop=True)
    
    y_train_true = df_train.loc[train_features['index'], 'primary_ndvi'].values
    y_train_linear = train_features['y_linear'].values
    delta_train = y_train_true - y_train_linear
    delta_clipped = np.clip(delta_train, -0.30, 0.30)
    
    base_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_linear))
    base_gap = round(float(30 * max(0, 1 - base_rmse / 0.10)), 2)
    print(f"Линейный гармонизированный бейзлайн: RMSE = {base_rmse:.5f} | GapScore = {base_gap:5.2f} / 30")
    
    # 6. Обучение ансамбля 5-Fold GroupKFold (Zero-Leakage)
    print("\n6. Обучение ансамбля (5-Fold LightGBM Huber + 5-Fold CatBoost) с разбиением по полям...")
    X = train_features[FEATURE_COLUMNS].copy()
    X['crop_type'] = X['crop_type'].astype('category')
    groups = train_features['anon_polygon_id'].values
    
    X_cat = X.copy()
    X_cat['crop_type'] = X_cat['crop_type'].astype(str)
    
    gkf = GroupKFold(n_splits=5)
    oof_pred_lgb = np.zeros(len(X))
    oof_pred_cat = np.zeros(len(X))
    lgb_models = []
    cat_models = []
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, delta_clipped, groups=groups)):
        print(f"--- Фолд {fold + 1} / 5 (Валидация на непересекающихся полях) ---")
        X_tr, y_tr = X.iloc[trn_idx], delta_clipped[trn_idx]
        X_va, y_va = X.iloc[val_idx], delta_clipped[val_idx]
        
        # Обучение фолдовой модели LightGBM с робастной функцией потерь Huber
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
        
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
            lgb_params,
            train_data,
            num_boost_round=800,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)]
        )
        lgb_models.append(m_lgb)
        oof_pred_lgb[val_idx] = m_lgb.predict(X_va)
        
        # Обучение фолдовой модели CatBoost с функцией потерь Huber
        cb_tr = X_cat.iloc[trn_idx]
        cb_va = X_cat.iloc[val_idx]
        m_cb = CatBoostRegressor(
            iterations=600,
            learning_rate=0.035,
            depth=6,
            loss_function='Huber:delta=0.08',
            cat_features=['crop_type'],
            random_seed=42 + fold,
            verbose=0
        )
        m_cb.fit(cb_tr, y_tr, eval_set=(cb_va, y_va), early_stopping_rounds=40)
        cat_models.append(m_cb)
        oof_pred_cat[val_idx] = m_cb.predict(cb_va)
        
        fold_p = np.clip(y_train_linear[val_idx] + 0.8 * oof_pred_lgb[val_idx] + 0.2 * oof_pred_cat[val_idx], 0.08, 0.95)
        f_rmse = np.sqrt(mean_squared_error(y_train_true[val_idx], fold_p))
        f_gap = round(float(30 * max(0, 1 - f_rmse / 0.10)), 2)
        print(f"Фолд {fold + 1} OOF -> RMSE: {f_rmse:.5f} | GapScore: {f_gap:5.2f} / 30")
        
    # Численный подбор оптимальных весов блендинга на OOF
    def loss_func(w):
        w1, w2 = w[0], w[1]
        pred = np.clip(y_train_linear + w1 * oof_pred_lgb + w2 * oof_pred_cat, 0.08, 0.95)
        return np.sqrt(mean_squared_error(y_train_true, pred))
        
    res_opt = minimize(loss_func, [0.8, 0.2], bounds=[(0.0, 1.0), (0.0, 1.0)])
    w_lgb, w_cat = res_opt.x[0], res_opt.x[1]
    w_sum = w_lgb + w_cat
    w_lgb /= w_sum
    w_cat /= w_sum
    print(f"\nОптимальные веса ансамбля: LightGBM = {w_lgb:.3f}, CatBoost = {w_cat:.3f}")
    
    final_oof_pred = np.clip(y_train_linear + w_lgb * oof_pred_lgb + w_cat * oof_pred_cat, 0.08, 0.95)
    final_rmse = np.sqrt(mean_squared_error(y_train_true, final_oof_pred))
    final_gap = round(float(30 * max(0, 1 - final_rmse / 0.10)), 2)
    
    print("\n=======================================================")
    print(f"ИТОГОВЫЙ РЕЗУЛЬТАТ GROUP-K-FOLD OOF (НЕИЗВЕСТНЫЕ ПОЛЯ):")
    print(f"Линейный бейзлайн: RMSE = {base_rmse:.5f} | GapScore = {base_gap:5.2f} / 30")
    print(f"Ансамбль моделей:  RMSE = {final_rmse:.5f} | GapScore = {final_gap:5.2f} / 30")
    print(f"Прирост точности:  +{final_gap - base_gap:.2f} баллов")
    print("=======================================================")
    
    # 7. Сохранение артефактов
    bundle = {
        'lgb_models': lgb_models,
        'cat_models': cat_models,
        'feature_cols': FEATURE_COLUMNS,
        'date_sat_stats': date_sat_stats,
        'date_crop_stats': date_crop_stats,
        'lgb_weight': float(w_lgb),
        'cat_weight': float(w_cat)
    }
    with open("artifacts/models/ensemble_models.pkl", "wb") as f:
        pickle.dump(bundle, f)
        
    print("[OK] Все модели и калибровочные таблицы сохранены в artifacts/models/ensemble_models.pkl")
    print("=== ОБУЧЕНИЕ МОДЕЛЕЙ УСПЕШНО ЗАВЕРШЕНО ===")

if __name__ == "__main__":
    run_training()
