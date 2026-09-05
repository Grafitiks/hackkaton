import os
import sys
import argparse
import pandas as pd

# Настройка вывода UTF-8 для консоли Windows
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.models.inference import predict_gaps

def main():
    parser = argparse.ArgumentParser(description="Космохакатон: Batch-инференс восстановления primary_ndvi")
    parser.add_argument("--input", default="data/private_features.csv", help="Путь к тестовому файлу (private_features.csv)")
    parser.add_argument("--output", default="submission.csv", help="Путь для сохранения итогового submission.csv")
    parser.add_argument("--model", default="artifacts/models/ensemble_models.pkl", help="Путь к артефакту моделей")
    parser.add_argument("--clim", default="artifacts/models/climatology.pkl", help="Путь к артефакту климатологии")
    args = parser.parse_args()
    
    print(f"==================================================")
    print(f"Космохакатон: Запуск Batch-инференса (Задача 1)")
    print(f"Входной файл:   {args.input}")
    print(f"Выходной файл:  {args.output}")
    print(f"==================================================")
    
    # Корректный вывод кириллицы в консоли Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    if not os.path.exists(args.input):
        print(f"ОШИБКА: Файл {args.input} не найден!")
        sys.exit(1)
        
    # Запуск пакетного инференса через ансамбль LightGBM + CatBoost
    sub = predict_gaps(args.input, args.model, args.clim)
    
    # Контрольная верификация формата submission.csv по требованиям регламента
    print("\n--- Проверка формата submission.csv ---")
    print(f"Строк в итоговом датасете: {len(sub)}")
    print(f"Колонки: {list(sub.columns)}")
    print(f"Проверка на NaN: {sub.isna().sum().to_dict()}")
    print(f"Дубликаты по (anon_polygon_id, date): {sub.duplicated(subset=['anon_polygon_id', 'date']).sum()}")
    print("Первые 5 строк:")
    print(sub.head())
    
    # Сохранение итогового сабмита в кодировке UTF-8 с разделителем запятая
    sub.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n[УСПЕХ] Файл {args.output} успешно сохранен и готов к отправке на платформу!")

if __name__ == "__main__":
    main()
