import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

_GEE_INITIALIZED = False
_GEE_CACHE: Dict[str, pd.DataFrame] = {}

def init_gee(project_id: Optional[str] = None) -> bool:
    """
    Инициализация Google Earth Engine через постоянные учетные данные,
    сервисный аккаунт (gee-key.json) или проект Google Cloud.
    """
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return True
        
    if not project_id:
        project_id = os.environ.get("GEE_PROJECT_ID", "ee-agro-hackathon")
        
    try:
        import ee
        key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../gee-key.json"))
        if os.path.exists(key_path):
            print(f"[GEE] Обнаружен ключ сервисного аккаунта: {key_path}")
            # Аутентификация через JSON ключ сервисного аккаунта
            import json
            with open(key_path, "r") as f:
                key_data = json.load(f)
            client_email = key_data.get("client_email")
            credentials = ee.ServiceAccountCredentials(client_email, key_path)
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(project=project_id)
            
        _GEE_INITIALIZED = True
        print(f"[GEE] Google Earth Engine успешно инициализирован с проектом: {project_id}")
        return True
    except Exception as e:
        print(f"[GEE] Earth Engine в текущей сессии недоступен (проект: {project_id}): {e}")
        return False

def is_gee_available() -> bool:
    """Проверка доступности и активного подключения к Google Earth Engine."""
    return _GEE_INITIALIZED or init_gee()

def fetch_gee_sentinel2_ndvi(coords: List[List[float]], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Выборка временного ряда NDVI из коллекции Sentinel-2 SR Harmonized (10м) с маской облачности SCL.
    Коллекция: COPERNICUS/S2_SR_HARMONIZED (каналы B8 NIR, B4 Red, SCL).
    """
    if not is_gee_available():
        return None
        
    cache_key = f"s2_{start_date}_{end_date}_" + "_".join(f"{pt[0]:.4f},{pt[1]:.4f}" for pt in coords[:6])
    if cache_key in _GEE_CACHE:
        return _GEE_CACHE[cache_key].copy()
        
    try:
        import ee
        poly = ee.Geometry.Polygon(coords)
        
        def mask_s2_clouds(image):
            # Фильтрация пикселей облаков, теней и перистых облаков по карте классификации сцены (SCL)
            scl = image.select('SCL')
            mask = scl.neq(3).And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('s2_ndvi')
            return image.addBands(ndvi).updateMask(mask)
            
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(poly)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
            .map(mask_s2_clouds)
        )
        
        def extract_stats(img):
            mean_dict = img.select('s2_ndvi').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=poly,
                scale=20,
                maxPixels=1e7,
                bestEffort=True
            )
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'),
                's2_ndvi': mean_dict.get('s2_ndvi')
            })
            
        features = collection.map(extract_stats).getInfo().get('features', [])
        
        records = []
        for f in features:
            props = f.get('properties', {})
            val = props.get('s2_ndvi')
            if val is not None and -0.2 <= float(val) <= 1.0:
                records.append({
                    'date': props.get('date'),
                    's2_ndvi': float(val)
                })
                
        if records:
            df = pd.DataFrame(records).drop_duplicates('date').sort_values('date').reset_index(drop=True)
            _GEE_CACHE[cache_key] = df
            return df
    except Exception as e:
        print(f"[GEE] Ошибка при выборке Sentinel-2 из Earth Engine: {e}")
        
    return None

def fetch_gee_landsat_ndvi(coords: List[List[float]], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Выборка временного ряда NDVI из коллекций Landsat 8 и 9 Tier 1 Surface Reflectance (30м).
    Коллекции: LANDSAT/LC08/C02/T1_L2, LANDSAT/LC09/C02/T1_L2 (каналы SR_B5 NIR, SR_B4 Red, QA_PIXEL).
    """
    if not is_gee_available():
        return None
        
    cache_key = f"ls_{start_date}_{end_date}_" + "_".join(f"{pt[0]:.4f},{pt[1]:.4f}" for pt in coords[:6])
    if cache_key in _GEE_CACHE:
        return _GEE_CACHE[cache_key].copy()
        
    try:
        import ee
        poly = ee.Geometry.Polygon(coords)
        
        def mask_landsat_clouds(img):
            qa = img.select('QA_PIXEL')
            # Битовая маска QA_PIXEL: исключение облаков, плотной дымки и теней
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) \
                .And(qa.bitwiseAnd(1 << 4).eq(0)) \
                .And(qa.bitwiseAnd(1 << 2).eq(0))
            # Физическое масштабирование USGS Landsat Collection 2 Level 2: DN * 0.0000275 - 0.2
            b5 = img.select('SR_B5').multiply(0.0000275).add(-0.2)
            b4 = img.select('SR_B4').multiply(0.0000275).add(-0.2)
            ndvi = b5.subtract(b4).divide(b5.add(b4)).rename('landsat_ndvi')
            return img.addBands(ndvi).updateMask(cloud_mask)

        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(poly).filterDate(start_date, end_date).map(mask_landsat_clouds)
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(poly).filterDate(start_date, end_date).map(mask_landsat_clouds)
        combined = l8.merge(l9)

        def extract_stats(img):
            mean_dict = img.select('landsat_ndvi').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=poly,
                scale=30,
                maxPixels=1e7,
                bestEffort=True
            )
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'),
                'landsat_ndvi': mean_dict.get('landsat_ndvi')
            })

        features = combined.map(extract_stats).getInfo().get('features', [])
        records = []
        for f in features:
            props = f.get('properties', {})
            val = props.get('landsat_ndvi')
            if val is not None and -0.2 <= float(val) <= 1.0:
                records.append({
                    'date': props.get('date'),
                    'landsat_ndvi': float(val)
                })

        if records:
            df = pd.DataFrame(records).drop_duplicates('date').sort_values('date').reset_index(drop=True)
            _GEE_CACHE[cache_key] = df
            return df
    except Exception as e:
        print(f"[GEE] Ошибка при выборке Landsat из Earth Engine: {e}")

    return None

def fetch_gee_modis_ndvi(coords: List[List[float]], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Выборка временного ряда NDVI из композита MODIS (16-дневный композит 250м).
    Коллекция: MODIS/061/MOD13Q1 (масштабирование NDVI на коэффициент 0.0001).
    """
    if not is_gee_available():
        return None
        
    cache_key = f"mod_{start_date}_{end_date}_" + "_".join(f"{pt[0]:.4f},{pt[1]:.4f}" for pt in coords[:6])
    if cache_key in _GEE_CACHE:
        return _GEE_CACHE[cache_key].copy()
        
    try:
        import ee
        poly = ee.Geometry.Polygon(coords)
        
        modis = (
            ee.ImageCollection('MODIS/061/MOD13Q1')
            .filterBounds(poly)
            .filterDate(start_date, end_date)
        )
        
        def scale_ndvi(img):
            ndvi = img.select('NDVI').multiply(0.0001).rename('modis_ndvi')
            return img.addBands(ndvi)
            
        def extract_stats(img):
            mean_dict = img.select('modis_ndvi').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=poly,
                scale=250,
                maxPixels=1e7,
                bestEffort=True
            )
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'),
                'modis_ndvi': mean_dict.get('modis_ndvi')
            })
            
        features = modis.map(scale_ndvi).map(extract_stats).getInfo().get('features', [])
        records = []
        for f in features:
            props = f.get('properties', {})
            val = props.get('modis_ndvi')
            if val is not None and -0.2 <= float(val) <= 1.0:
                records.append({
                    'date': props.get('date'),
                    'modis_ndvi': float(val)
                })

        if records:
            df = pd.DataFrame(records).drop_duplicates('date').sort_values('date').reset_index(drop=True)
            _GEE_CACHE[cache_key] = df
            return df
    except Exception as e:
        print(f"[GEE] Ошибка при выборке MODIS из Earth Engine: {e}")

    return None

def fetch_gee_climatology(coords: List[List[float]]) -> Optional[pd.DataFrame]:
    """
    Расчет динамической многолетней климатологии (среднее и дисперсия NDVI по дням года DOY)
    напрямую в Google Earth Engine на основе 8-летнего архива MODIS (2016-2024).
    Работает для любого полигона планеты с учетом реальной фенологии местности.
    """
    if not is_gee_available():
        return None
        
    cache_key = "clim_" + "_".join(f"{pt[0]:.4f},{pt[1]:.4f}" for pt in coords[:6])
    if cache_key in _GEE_CACHE:
        return _GEE_CACHE[cache_key].copy()
        
    try:
        import ee
        poly = ee.Geometry.Polygon(coords)
        
        # 8-летний непрерывный глобальный исторический композит (2016-2024)
        modis = (
            ee.ImageCollection('MODIS/061/MOD13Q1')
            .filterBounds(poly)
            .filterDate('2016-01-01', '2024-12-31')
            .select('NDVI')
        )
        
        def extract_hist(img):
            mean_dict = img.select('NDVI').multiply(0.0001).rename('ndvi').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=poly,
                scale=250,
                maxPixels=1e7,
                bestEffort=True
            )
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'),
                'doy': img.date().getRelative('day', 'year').add(1),
                'ndvi': mean_dict.get('ndvi')
            })
            
        features = modis.map(extract_hist).getInfo().get('features', [])
        records = []
        for f in features:
            props = f.get('properties', {})
            val = props.get('ndvi')
            if val is not None and 0.05 <= float(val) <= 0.95:
                records.append({
                    'doy': int(props.get('doy', 1)),
                    'ndvi': float(val)
                })
                
        if records and len(records) >= 20:
            df_hist = pd.DataFrame(records)
            clim_stats = df_hist.groupby('doy')['ndvi'].agg(['mean', 'std']).reset_index()
            clim_stats.columns = ['doy', 'clim_mean', 'clim_std']
            
            # Построение профиля нормы на 366 дней года со скользящим сглаживанием
            full_doy = pd.DataFrame({'doy': list(range(1, 367))})
            clim_df = full_doy.merge(clim_stats, on='doy', how='left')
            clim_df['clim_mean'] = clim_df['clim_mean'].interpolate(method='linear', limit_direction='both')
            clim_df['clim_mean'] = clim_df['clim_mean'].rolling(21, min_periods=1, center=True).mean()
            clim_df['clim_std'] = clim_df['clim_std'].interpolate(method='linear', limit_direction='both').fillna(0.065).clip(0.03, 0.15)
            
            _GEE_CACHE[cache_key] = clim_df
            print(f"[GEE] Успешно рассчитана многолетняя климатология GEE ({len(records)} исторических наблюдений)!")
            return clim_df
    except Exception as e:
        print(f"[GEE] Ошибка расчета динамической климатологии GEE: {e}")
        
    return None

def get_current_year() -> int:
    """Returns the current calendar year."""
    return datetime.now().year

def get_available_years(start_year: int = 2014) -> List[int]:
    """
    Returns available years from start_year (2014) up to current year,
    dynamically synchronized with GEE.
    """
    curr = get_current_year()
    return list(range(curr, start_year - 1, -1))
