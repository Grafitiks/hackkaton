import os
import json

# Базовая директория репозитория
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Значения конфигурации по умолчанию
DEFAULT_CONFIG = {
    "project": {
        "name": "GEO-VEGA",
        "description": "Мониторинг вегетационной динамики с/х территорий, реконструкция рядов ДЗЗ и детекция аномалий",
        "version": "1.0.0",
        "random_seed": 42
    },
    "data": {
        "train_dataset": os.path.join(BASE_DIR, "data/train_dataset.csv"),
        "test_features": os.path.join(BASE_DIR, "data/test_features (1).csv"),
        "private_features": os.path.join(BASE_DIR, "data/private_features.csv"),
        "polygons_geo": os.path.join(BASE_DIR, "src/web/polygons_geo.json"),
        "submission_output": os.path.join(BASE_DIR, "submission.csv")
    },
    "artifacts": {
        "models_dir": os.path.join(BASE_DIR, "artifacts/models"),
        "ensemble_models": os.path.join(BASE_DIR, "artifacts/models/ensemble_models.pkl"),
        "climatology": os.path.join(BASE_DIR, "artifacts/models/climatology.pkl")
    },
    "model": {
        "n_splits": 5,
        "group_column": "anon_polygon_id",
        "target_column": "primary_ndvi",
        "synthetic_gap_column": "is_synthetic_gap",
        "lgb_weight": 0.60,
        "cat_weight": 0.40
    },
    "anomalies": {
        "zscore_thresholds": {
            "normal_min": -1.0,
            "moderate_min": -2.0,
            "critical_max": -2.0
        },
        "safe_std_floor": 0.06
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False
    }
}

def load_config(config_path: str = None) -> dict:
    """Загружает параметры конфигурации из YAML файла или возвращает значения по умолчанию."""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, "configs/config.yaml")
        
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
            
    return DEFAULT_CONFIG

CONFIG = load_config()
