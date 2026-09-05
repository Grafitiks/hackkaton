import os
import sys
import argparse

# Настройка вывода UTF-8 для корректного отображения логов в консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Обеспечение импорта локальных модулей проекта
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.models.train import run_training
from src.features.climatology import compute_climatology

def main():
    parser = argparse.ArgumentParser(description="GEO-VEGA // Обучение 5-фолдового ансамбля моделей и расчет климатологии")
    
    # Автоопределение путей
    default_train = "data/train_dataset.csv" if os.path.exists("data/train_dataset.csv") else "train_dataset.csv"
    default_test = "data/test_features (1).csv" if os.path.exists("data/test_features (1).csv") else "data/private_features.csv"
    
    parser.add_argument("--train-path", default=default_train, help="Путь к обучающей выборке (CSV)")
    parser.add_argument("--test-path", default=default_test, help="Путь к тестовой выборке для трансдуктивной статистики (CSV)")
    parser.add_argument("--output-dir", default="artifacts/models", help="Директория для сохранения артефактов моделей")
    parser.add_argument("--recompute-climatology", action="store_true", help="Принудительно пересчитать климатологию DOY")
    
    args = parser.parse_args()
    
    print("============================================================================")
    print("GEO-VEGA: ПОЛНЫЙ ЦИКЛ ОБУЧЕНИЯ МОДЕЛЕЙ И ФОРМИРОВАНИЯ АРТЕФАКТОВ")
    print(f"Обучающий датасет: {args.train_path}")
    print(f"Тестовый датасет:  {args.test_path}")
    print(f"Выходная папка:    {args.output_dir}")
    print("============================================================================")
    
    if not os.path.exists(args.train_path):
        print(f"[ОШИБКА] Файл {args.train_path} не найден! Поместите train_dataset.csv в data/ или укажите --train-path.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Расчет и сохранение климатологических профилей DOY
    clim_path = os.path.join(args.output_dir, "climatology.pkl")
    if not os.path.exists(clim_path) or args.recompute_climatology:
        print("\n[Шаг 1/2] Расчет многолетней климатологии DOY по полигонам и культурам...")
        import pandas as pd
        import pickle
        df_tr = pd.read_csv(args.train_path, encoding='utf-8')
        df_te = pd.read_csv(args.test_path, encoding='utf-8') if os.path.exists(args.test_path) else None
        df_all = pd.concat([df_tr, df_te], ignore_index=True) if df_te is not None else df_tr
        if 'doy' not in df_all.columns:
            df_all['doy'] = pd.to_datetime(df_all['date']).dt.dayofyear
        poly_clim, crop_clim, global_clim = compute_climatology(df_all)
        with open(clim_path, "wb") as f:
            pickle.dump({
                'poly_clim': poly_clim,
                'crop_clim': crop_clim,
                'global_clim': global_clim
            }, f)
        print(f"[OK] Климатологические артефакты сохранены в {clim_path}")
    else:
        print(f"\n[Шаг 1/2] Использование существующей климатологии из {clim_path}")
        
    # 2. Обучение 5-Fold GroupKFold ансамбля LightGBM + CatBoost
    print("\n[Шаг 2/2] Запуск обучения 5-фолдового ансамбля LightGBM + CatBoost на 43 признаках...")
    run_training()
    
    print("\n============================================================================")
    print("Все артефакты успешно обновлены и готовы к инференсу!")
    print("Для формирования submission.csv запустите:")
    print("  python predict_submission.py")
    print("============================================================================")

if __name__ == "__main__":
    main()
