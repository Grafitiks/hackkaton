import sys
import os
import socket
import argparse
import uvicorn

# Настройка вывода UTF-8 для корректного отображения логов в Windows
sys.stdout.reconfigure(encoding='utf-8')

def is_port_available(port: int) -> bool:
    """Проверяет доступность сетевого порта перед запуском сервера."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Выполняет безопасный автопоиск свободного порта при конфликте портов."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return start_port

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск веб-сервиса GEO-VEGA")
    parser.add_argument("--port", type=int, default=8000, help="Порт для запуска (по умолчанию 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Хост (по умолчанию 0.0.0.0)")
    parser.add_argument("--no-reload", action="store_true", help="Отключить автоперезагрузку кода (рекомендуется для продакшена и Docker)")
    args = parser.parse_args()

    port = args.port
    if not is_port_available(port):
        alt_port = find_available_port(port + 1)
        print(f"[ВНИМАНИЕ] Порт {port} уже занят другим процессом. Переключаемся на свободный порт {alt_port}...")
        port = alt_port

    # В продакшен-режиме или в Docker отключаем непрерывный опрос файлов для экономии ресурсов CPU
    is_prod = args.no_reload or (os.environ.get("PRODUCTION", "0") == "1")
    use_reload = not is_prod

    print("==================================================")
    print("Запуск геосервиса GEO-VEGA // Космохакатон")
    print(f"Адрес веб-интерфейса: http://localhost:{port}")
    print(f"Документация API:      http://localhost:{port}/docs")
    print(f"Режим работы:          {'Продакшен (без reload)' if is_prod else 'Разработка (reload включен)'}")
    print("==================================================")
    
    uvicorn.run("src.web.app:app", host=args.host, port=port, reload=use_reload)
