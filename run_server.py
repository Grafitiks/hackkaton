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
    args = parser.parse_args()

    port = args.port
    if not is_port_available(port):
        alt_port = find_available_port(port + 1)
        print(f"[ВНИМАНИЕ] Порт {port} уже занят другим процессом. Переключаемся на свободный порт {alt_port}...")
        port = alt_port

    print("==================================================")
    print("Запуск геосервиса GEO-VEGA // Космохакатон")
    print(f"Адрес веб-интерфейса: http://localhost:{port}")
    print(f"Документация API:      http://localhost:{port}/docs")
    print("==================================================")
    
    uvicorn.run("src.web.app:app", host=args.host, port=port, reload=False)
