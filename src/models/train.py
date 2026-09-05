import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.climatology import compute_climatology
from src.features.feature_builder import enrich_weather, extract_gap_features, FEATURE_COLUMNS

def run_training():
    print("=== STARTING MODEL TRAINING & PIPELINE PREPARATION ===")
    os.makedirs("artifacts/models", exist_ok=True)
    
    # 1. Load data
    print("Loading datasets...")
    df_train = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
    df_train['date_dt'] = pd.to_datetime(df_train['date'])
    df_train['year'] = df_train['date_dt'].dt.year
    df_train = df_train.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)
    
    df_test = pd.read_csv("data/private_features.csv", encoding='utf-8')
    df_test['date_dt'] = pd.to_datetime(df_test['date'])
    df_test['year'] = df_test['date_dt'].dt.year
    df_test = df_test.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)
    
    # 2. Enrich weather
    print("Enriching weather data...")
    df_train = enrich_weather(df_train)
    df_test = enrich_weather(df_test)
    
    # 3. Transductive Climatology: Combine known observations from both train and test
    print("Computing comprehensive climatology across all 78 polygons...")
    df_combined = pd.concat([
        df_train[['anon_polygon_id', 'crop_type', 'doy', 'primary_ndvi']],
        df_test[df_test['primary_ndvi'].notna()][['anon_polygon_id', 'crop_type', 'doy', 'primary_ndvi']]
    ], ignore_index=True)
    
    poly_clim, crop_clim, global_clim = compute_climatology(df_combined, rolling_window=11)
    
    # Save climatology artifacts
    with open("artifacts/models/climatology.pkl", "wb") as f:
        pickle.dump({
            'poly_clim': poly_clim,
            'crop_clim': crop_clim,
            'global_clim': global_clim
        }, f)
    print("Climatology tables saved to artifacts/models/climatology.pkl")
    
    # 4. Generate Training Gap Samples from df_train
    print("Generating gap training samples...")
    np.random.seed(42)
    known_idx = df_train[df_train['primary_ndvi'].notna()].index.values
    
    # We sample 18,000 diverse synthetic gaps mimicking test structure
    train_sampled_idx = np.random.choice(known_idx, size=18000, replace=False)
    
    # Extract features
    print("Extracting features for training samples (this takes ~15 seconds)...")
    train_features = extract_gap_features(df_train, train_sampled_idx, poly_clim, crop_clim, global_clim)
    y_true = df_train.loc[train_features['index'], 'primary_ndvi'].values
    y_linear = train_features['y_linear'].values
    delta_true = y_true - y_linear
    
    # Baseline linear metrics
    base_rmse = np.sqrt(mean_squared_error(y_true, y_linear))
    base_gap = round(float(30 * max(0, 1 - base_rmse / 0.10)), 2)
    print(f"\n[Baseline Linear on Training Set] RMSE: {base_rmse:.5f} | GapScore: {base_gap:5.2f} / 30")
    
    # Prepare X
    X = train_features[FEATURE_COLUMNS].copy()
    
    # 5. 5-Fold Cross Validation Training
    print("\nTraining 5-Fold Ensemble (LightGBM + CatBoost)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    lgb_models = []
    cat_models = []
    
    # Categorical columns
    cat_cols = ['crop_type']
    for c in cat_cols:
        X[c] = X[c].astype('category')
        
    cat_indices = ['crop_type']
    X_cat = X.copy()
    for c in cat_cols:
        X_cat[c] = X_cat[c].astype(str)
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, delta_true)):
        print(f"--- Training Fold {fold + 1} / 5 ---")
        X_tr, y_tr = X.iloc[trn_idx], delta_true[trn_idx]
        X_va, y_va = X.iloc[val_idx], delta_true[val_idx]
        
        # LightGBM
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'seed': 42 + fold,
            'verbose': -1
        }
        
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=600,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        lgb_models.append(lgb_model)
        
        # CatBoost
        cb_tr = X_cat.iloc[trn_idx]
        cb_va = X_cat.iloc[val_idx]
        cb_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            cat_features=cat_indices,
            random_seed=42 + fold,
            verbose=0
        )
        cb_model.fit(cb_tr, y_tr, eval_set=(cb_va, y_va), early_stopping_rounds=40)
        cat_models.append(cb_model)
        
        # Blend prediction
        p_lgb = lgb_model.predict(X_va)
        p_cb = cb_model.predict(cb_va)
        fold_delta = 0.6 * p_lgb + 0.4 * p_cb
        oof_preds[val_idx] = fold_delta
        
        fold_y_pred = np.clip(y_linear[val_idx] + fold_delta, -0.2, 1.0)
        fold_rmse = np.sqrt(mean_squared_error(y_true[val_idx], fold_y_pred))
        fold_gap = round(float(30 * max(0, 1 - fold_rmse / 0.10)), 2)
        print(f"Fold {fold + 1} Result -> RMSE: {fold_rmse:.5f} | GapScore: {fold_gap:5.2f} / 30")
        
    # Overall OOF metrics
    final_oof_y = np.clip(y_linear + oof_preds, -0.2, 1.0)
    final_rmse = np.sqrt(mean_squared_error(y_true, final_oof_y))
    final_gap = round(float(30 * max(0, 1 - final_rmse / 0.10)), 2)
    print("\n=======================================================")
    print(f"OVERALL OUT-OF-FOLD RESULT:")
    print(f"RMSE:     {final_rmse:.5f}")
    print(f"GapScore: {final_gap:5.2f} / 30  (Baseline was {base_gap:5.2f})")
    print("=======================================================")
    
    # Save models
    with open("artifacts/models/ensemble_models.pkl", "wb") as f:
        pickle.dump({
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'feature_cols': FEATURE_COLUMNS
        }, f)
    print("All models saved successfully to artifacts/models/ensemble_models.pkl")
    print("=== MODEL TRAINING COMPLETE ===")

if __name__ == "__main__":
    run_training()
