import sys
import os

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print("Настройка и подключение Google Earth Engine (GEE)")
print("==================================================")

try:
    import ee
except ImportError:
    print("Установка библиотеки earthengine-api...")
    os.system("pip install earthengine-api")
    import ee

PROJECT_ID = "ee-agro-hackathon"

# 1. Check if already authenticated
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"\n[УСПЕХ] Google Earth Engine уже авторизован (проект: {PROJECT_ID}) и готов к работе!")
    
    # Test query Sentinel-2
    point = ee.Geometry.Point([50.2, 53.2])
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(point).filterDate('2024-06-01', '2024-06-30').first()
    info = s2.getInfo()
    print(f"Тестовый снимок Sentinel-2 получен: ID={info['id']}")
    print("Интеграция GEE активна в сервисе GEO-VEGA.")
    sys.exit(0)
except Exception as e:
    print(f"\nТребуется авторизация учетной записи Google Earth Engine.")
    print("Сейчас запустится мастер авторизации Google...")

# 2. Run interactive authentication
try:
    ee.Authenticate()
    print("\nАвторизация в браузере пройдена. Инициализация...")
    ee.Initialize()
    print("\n[УСПЕХ] Google Earth Engine успешно подключен к системе GEO-VEGA!")
except Exception as err:
    print(f"\n[ИНСТРУКЦИЯ] Если возникла ошибка, выполните следующие шаги:")
    print("1. Зарегистрируйтесь на бесплатном портале: https://code.earthengine.google.com/register")
    print("   (выберите 'Non-commercial / Research / Education' -> Создать облачный проект, например: ee-myproject)")
    print("2. В терминале выполните команду:")
    print("   earthengine authenticate")
    print("3. В браузере выберите ваш Google-аккаунт и нажмите 'Разрешить'.")
    print("4. Если у вас уже есть Cloud Project ID, запустите:")
    print("   python -c \"import ee; ee.Initialize(project='ВАШ_PROJECT_ID')\"")
