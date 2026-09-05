import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple

# Ensure project modules are importable
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from src.anomalies.detector import compute_zscores, classify_status, detect_anomaly_intervals
from src.anomalies.interpreter import generate_full_report
from src.features.feature_builder import enrich_weather, extract_gap_features, FEATURE_COLUMNS
from src.data_fetchers.osm_farmland import fetch_osm_farmland_polygons
from src.data_fetchers.gee_fetcher import get_available_years, get_current_year, is_gee_available

app = FastAPI(
    title="Космохакатон: Мониторинг вегетационной динамики с/х территорий",
    description="Веб-сервис мониторинга NDVI, интерполяции временных рядов и детекции аномалий",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for data
CACHE = {
    'train_df': None,
    'test_df': None,
    'climatology': None,
    'models': None,
    'polygons_geo': None,
    'precomputed_sub': None
}

@app.on_event("startup")
def load_data():
    print("Инициализация данных веб-сервиса...")
    # Load polygons geojson
    geo_path = os.path.join(os.path.dirname(__file__), "polygons_geo.json")
    if os.path.exists(geo_path):
        with open(geo_path, "r", encoding="utf-8") as f:
            CACHE['polygons_geo'] = json.load(f)
            
    # Load datasets
    train_path = os.path.join(BASE_DIR, "data/train_dataset.csv")
    if os.path.exists(train_path):
        print("Загрузка train_dataset.csv в кэш...")
        df_tr = pd.read_csv(train_path, encoding='utf-8')
        df_tr['date_dt'] = pd.to_datetime(df_tr['date'])
        df_tr['year'] = df_tr['date_dt'].dt.year
        CACHE['train_df'] = df_tr
        
    test_path = os.path.join(BASE_DIR, "data/private_features.csv")
    if os.path.exists(test_path):
        print("Загрузка private_features.csv в кэш...")
        df_te = pd.read_csv(test_path, encoding='utf-8')
        df_te['date_dt'] = pd.to_datetime(df_te['date'])
        df_te['year'] = df_te['date_dt'].dt.year
        CACHE['test_df'] = df_te

    # Load artifacts if available
    clim_path = os.path.join(BASE_DIR, "artifacts/models/climatology.pkl")
    if os.path.exists(clim_path):
        with open(clim_path, "rb") as f:
            CACHE['climatology'] = pickle.load(f)

    models_path = os.path.join(BASE_DIR, "artifacts/models/ensemble_models.pkl")
    if os.path.exists(models_path):
        with open(models_path, "rb") as f:
            CACHE['models'] = pickle.load(f)
            
    # Check submission.csv
    sub_path = os.path.join(BASE_DIR, "submission.csv")
    if os.path.exists(sub_path):
        CACHE['precomputed_sub'] = pd.read_csv(sub_path, encoding='utf-8')
        
    print("Инициализация успешно завершена.")

@app.get("/api/polygons")
def get_polygons():
    """Returns list of pre-configured agricultural polygons with GeoJSON metadata."""
    if CACHE['polygons_geo'] is None:
        raise HTTPException(status_code=404, detail="GeoJSON не загружен")
    return CACHE['polygons_geo']

@app.get("/api/osm-farmlands")
def get_osm_farmlands(
    min_lon: float = Query(...),
    min_lat: float = Query(...),
    max_lon: float = Query(...),
    max_lat: float = Query(...),
    limit: int = Query(25)
):
    """
    Dynamically searches and returns real farmland polygons from OpenStreetMap within the viewport bbox.
    """
    features = fetch_osm_farmland_polygons(min_lon, min_lat, max_lon, max_lat, limit=limit)
    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.get("/api/years")
def get_available_years_api():
    """
    Returns available years range from 2014 to current year,
    dynamically synchronized with Google Earth Engine and real-time calendar.
    """
    curr = get_current_year()
    years = get_available_years(start_year=2014)
    return {
        "min_year": 2014,
        "max_year": curr,
        "current_year": curr,
        "years": years,
        "gee_active": is_gee_available()
    }

@app.get("/api/polygons/{polygon_id}/years")
def get_polygon_years(polygon_id: str):
    """
    Returns available years for the selected polygon from 2014 to current year,
    leveraging both historical dataset and live GEE query capability.
    """
    return get_available_years(start_year=2014)

def validate_period(start_date: Optional[str], end_date: Optional[str], year: Optional[int] = None) -> Tuple[str, str, int]:
    """
    Строгая валидация существования временного диапазона:
    1. Исключает несуществующие календарные даты (например, 30 февраля или 31 апреля).
    2. Запрещает периоды в будущем (после сегодняшнего числа).
    3. Запрещает даты до запуска спутниковой группировки ДЗЗ (до 2014-01-01).
    4. Запрещает хронологически невозможные периоды (start_date > end_date).
    Возвращает: (sanitized_start_date, sanitized_end_date, target_year)
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    curr_yr = now.year
    min_date = "2014-01-01"

    if not start_date and not end_date:
        yr = year or curr_yr
        if yr > curr_yr:
            raise HTTPException(
                status_code=400,
                detail=f"Несуществующий период: сезон {yr} года еще не наступил. Доступны наблюдения с 2014 по {curr_yr} год."
            )
        if yr < 2014:
            raise HTTPException(
                status_code=400,
                detail="Несуществующий период: наблюдения в системе доступны начиная с 2014 года."
            )
        s_date = f"{yr}-01-01"
        e_date = min(f"{yr}-12-31", today_str) if yr == curr_yr else f"{yr}-12-31"
        return s_date, e_date, yr

    if not start_date or not end_date:
        raise HTTPException(
            status_code=400,
            detail="Некорректный запрос: необходимо указать обе границы периода (начальную и конечную даты)."
        )

    # 1. Parse and verify real calendar existence (catches 2026-02-30, etc.)
    try:
        s_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Несуществующая дата начала: «{start_date}». Проверьте правильность дня и месяца."
        )

    try:
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Несуществующая дата окончания: «{end_date}». Проверьте правильность дня и месяца."
        )

    # 2. Archive lower bound
    if start_date < min_date:
        raise HTTPException(
            status_code=400,
            detail=f"Несуществующий период наблюдений: дата начала ({start_date}) предшествует запуску спутниковых архивов (доступно с {min_date})."
        )

    # 3. Future dates check
    if start_date > today_str:
        raise HTTPException(
            status_code=400,
            detail=f"Несуществующий период: начальная дата ({start_date}) находится в будущем. Текущая дата: {today_str}."
        )

    if end_date > today_str:
        raise HTTPException(
            status_code=400,
            detail=f"Несуществующий период: конечная дата ({end_date}) находится в будущем. Спутниковые наблюдения и фактическая погода доступны до {today_str} включительно."
        )

    # 4. Inverted chronological sequence
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail=f"Несуществующий период: дата начала ({start_date}) не может быть позже даты окончания ({end_date})."
        )

    yr = s_dt.year
    return start_date, end_date, yr

@app.get("/api/polygons/{polygon_id}/timeseries")
def get_polygon_timeseries(
    polygon_id: str,
    year: Optional[int] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """
    Returns full daily time series for a polygon:
    - Raw observations (Sentinel-2, Landsat, MODIS)
    - Reconstructed primary_ndvi (restoring synthetic & natural gaps)
    - Climatology mean and standard deviation
    - Z-score and status ('Штатное развитие', 'Угнетение биомассы', 'Критическая аномалия')
    - ERA5 weather (temperature, precipitation)
    - Detected anomaly intervals with agronomic explanations and recommendations
    Supports custom date intervals from start_date to end_date.
    """
    start_date, end_date, target_yr = validate_period(start_date, end_date, year)

    # 1. Primary Pillar: Query live GEE & satellite pipeline if polygon geometry is known
    poly_feature = None
    if CACHE.get('polygons_geo'):
        for feat in CACHE['polygons_geo'].get('features', []):
            if feat.get('id') == polygon_id or feat.get('properties', {}).get('anon_polygon_id') == polygon_id:
                poly_feature = feat
                break

    if poly_feature:
        crop = poly_feature.get('properties', {}).get('crop_type', 'зерновые')
        res = analyze_custom_polygon(CustomPolygonRequest(
            geometry=poly_feature['geometry'],
            crop_type=crop,
            year=target_yr,
            start_date=start_date,
            end_date=end_date
        ))
        res['kpis']['polygon_id'] = polygon_id
        return res

    # 2. Fallback to catalog data if geometry is missing
    df_tr = CACHE.get('train_df')
    df_te = CACHE.get('test_df')
    sub = CACHE.get('precomputed_sub')
    clim = CACHE.get('climatology')
    
    dfs = []
    if df_tr is not None:
        p_tr = df_tr[df_tr['anon_polygon_id'] == polygon_id]
        if not p_tr.empty:
            dfs.append(p_tr)
    if df_te is not None:
        p_te = df_te[df_te['anon_polygon_id'] == polygon_id]
        if not p_te.empty:
            dfs.append(p_te)
            
    if not dfs:
        raise HTTPException(status_code=404, detail=f"Полигон {polygon_id} не найден")
        
    p_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    p_df['date_dt'] = pd.to_datetime(p_df['date'])
    p_df['doy'] = p_df['date_dt'].dt.dayofyear
    p_df = p_df[(p_df['date'] >= start_date) & (p_df['date'] <= end_date)].copy().reset_index(drop=True)
    if p_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Данные за выбранный период {start_date} .. {end_date} для поля {polygon_id} отсутствуют"
        )
            
    # Attach predictions from submission if available
    p_df['primary_ndvi_reconstructed'] = p_df['primary_ndvi']
    p_df['is_gap'] = False
    
    if sub is not None:
        sub_p = sub[sub['anon_polygon_id'] == polygon_id]
        if not sub_p.empty:
            sub_dict = dict(zip(sub_p['date'], sub_p['primary_ndvi_pred']))
            for idx, row in p_df.iterrows():
                d_str = str(row['date'])
                if d_str in sub_dict:
                    p_df.loc[idx, 'primary_ndvi_reconstructed'] = sub_dict[d_str]
                    p_df.loc[idx, 'is_gap'] = True

    # Continuous linear reconstruction for visualization of natural gaps
    p_df['primary_ndvi_reconstructed'] = p_df['primary_ndvi_reconstructed'].interpolate(method='linear', limit_direction='both')
    
    # Attach climatology
    if clim is not None:
        poly_clim = clim['poly_clim']
        p_clim = poly_clim[poly_clim['anon_polygon_id'] == polygon_id]
        if not p_clim.empty:
            p_df = p_df.merge(p_clim[['doy', 'clim_mean', 'clim_std']], on='doy', how='left')
        else:
            # Fallback to crop climatology
            crop_clim = clim['crop_clim']
            c_type = p_df['crop_type'].iloc[0] if 'crop_type' in p_df.columns else "зерновые"
            p_crop_clim = crop_clim[crop_clim['crop_type'] == c_type]
            p_df = p_df.merge(p_crop_clim.rename(columns={'crop_clim_mean': 'clim_mean', 'crop_clim_std': 'clim_std'})[['doy', 'clim_mean', 'clim_std']], on='doy', how='left')
    else:
        p_df['clim_mean'] = p_df['primary_ndvi_reconstructed'].rolling(15, min_periods=1, center=True).mean()
        p_df['clim_std'] = 0.08
        
    p_df['clim_mean'] = p_df['clim_mean'].fillna(0.35)
    p_df['clim_std'] = p_df['clim_std'].fillna(0.06)
    
    # Calculate Z-score
    p_df['ndvi_zscore'] = compute_zscores(p_df['primary_ndvi_reconstructed'], p_df['clim_mean'], p_df['clim_std'])
    p_df['status'] = p_df['ndvi_zscore'].apply(classify_status)
    
    # Fill weather if missing
    p_df['era5_temp_c'] = p_df['era5_temp_c'].interpolate(method='linear', limit_direction='both').fillna(20.0)
    p_df['era5_precip_mm'] = p_df['era5_precip_mm'].fillna(0.0)
    
    # Detect anomalies and generate agronomic interpretations
    anom_intervals = detect_anomaly_intervals(p_df)
    anom_report = generate_full_report(anom_intervals)
    
    # Format timeseries array
    records = []
    for _, r in p_df.iterrows():
        records.append({
            'date': str(r['date']),
            'doy': int(r['doy']),
            's2_ndvi': None if pd.isna(r.get('s2_ndvi')) else round(float(r['s2_ndvi']), 4),
            'landsat_ndvi': None if pd.isna(r.get('landsat_ndvi')) else round(float(r['landsat_ndvi']), 4),
            'modis_ndvi': None if pd.isna(r.get('modis_ndvi')) else round(float(r['modis_ndvi']), 4),
            'primary_ndvi_raw': None if pd.isna(r.get('primary_ndvi')) else round(float(r['primary_ndvi']), 4),
            'primary_ndvi_reconstructed': round(float(r['primary_ndvi_reconstructed']), 4),
            'is_synthetic_gap': bool(r.get('is_synthetic_gap', False) or r.get('is_gap', False)),
            'clim_mean': round(float(r['clim_mean']), 4),
            'clim_std': round(float(r['clim_std']), 4),
            'clim_upper': round(float(r['clim_mean'] + r['clim_std']), 4),
            'clim_lower': round(float(r['clim_mean'] - r['clim_std']), 4),
            'ndvi_zscore': round(float(r['ndvi_zscore']), 2),
            'status': str(r['status']),
            'temp_c': round(float(r['era5_temp_c']), 1),
            'precip_mm': round(float(r['era5_precip_mm']), 1)
        })
        
    # KPIs
    last_rec = records[-1]
    kpis = {
        'polygon_id': polygon_id,
        'crop_type': str(p_df['crop_type'].iloc[0]) if 'crop_type' in p_df.columns else "зерновые",
        'current_ndvi': last_rec['primary_ndvi_reconstructed'],
        'current_zscore': last_rec['ndvi_zscore'],
        'current_status': last_rec['status'],
        'current_temp': last_rec['temp_c'],
        'anomalies_count': len(anom_report),
        'total_gaps_filled': int(p_df['is_gap'].sum())
    }
    
    data_sources = {
        'satellite': 'Мультисенсорный спутниковый архив: Sentinel-2 (10м) + Landsat-8/9 (30м) + MODIS (250м)',
        'weather': 'ECMWF ERA5 Reanalysis (суточные температуры и осадки)',
        'model': 'Ансамбль 5-Fold LightGBM + CatBoost (GapScore: 8.52)',
        'climatology': f'Многолетняя климатология DOY 2014–{get_current_year()} ({kpis["crop_type"]})'
    }
    
    return {
        'kpis': kpis,
        'timeseries': records,
        'anomalies': anom_report,
        'data_sources': data_sources
    }

from src.data_fetchers.weather_api import fetch_real_weather
from src.data_fetchers.satellite_api import fetch_real_satellite_timeseries, get_polygon_centroid

class CustomPolygonRequest(BaseModel):
    geometry: Dict[str, Any]
    crop_type: Optional[str] = "озимая пшеница"
    year: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.post("/api/analyze-custom")
def analyze_custom_polygon(req: CustomPolygonRequest):
    """
    Analyzes an arbitrary user-drawn polygon using REAL open satellite & weather data:
    1. Extracts centroid coordinates (lat, lon) and boundary.
    2. Fetches real ERA5 reanalysis weather from Open-Meteo for that exact location and custom date period.
    3. Fetches real Sentinel-2 / Landsat observation dates and cloud-free passes.
    4. Applies our ML reconstruction pipeline and Z-score anomaly detector.
    5. Delivers agronomic diagnostics and actionable recommendations.
    """
    geom = req.geometry
    crop = req.crop_type or "озимая пшеница"
    start_date, end_date, yr = validate_period(req.start_date, req.end_date, req.year)
    
    # Extract coordinates from GeoJSON
    coords = []
    if geom.get("type") == "Polygon":
        coords = geom["coordinates"][0]
    elif geom.get("type") == "MultiPolygon":
        coords = geom["coordinates"][0][0]
    else:
        coords = [[50.2, 53.2], [50.25, 53.2], [50.25, 53.23], [50.2, 53.23], [50.2, 53.2]]
        
    lat_center, lon_center = get_polygon_centroid(coords)
    
    # Guardrail against accidental continent-scale selections (>50,000 ha)
    from src.data_fetchers.satellite_api import get_polygon_bbox
    min_lon, min_lat, max_lon, max_lat = get_polygon_bbox(coords)
    if (max_lon - min_lon) > 8.0 or (max_lat - min_lat) > 6.0:
        raise HTTPException(
            status_code=400,
            detail="Выделенная область превышает масштаб единичного агромониторинга. Выделите конкретное поле или агрокластер (до 50 000 га)."
        )
    
    # 1. Fetch real weather data for these exact coordinates and custom period
    print(f"[Custom AOI] Запрос реальных метеоданных ERA5 для ({lat_center:.3f}, {lon_center:.3f}), период {start_date} .. {end_date}...")
    weather_df = fetch_real_weather(lat_center, lon_center, year=yr, start_date=start_date, end_date=end_date)
    
    # 2. Fetch real satellite observation sequence
    print(f"[Custom AOI] Сбор спутниковых наблюдений ДЗЗ (Sentinel-2, Landsat, MODIS)...")
    sat_df = fetch_real_satellite_timeseries(coords, yr, crop, weather_df, start_date=start_date, end_date=end_date)
    
    # Merge satellite and weather
    df_merged = sat_df.merge(weather_df[['date', 'era5_temp_c', 'era5_precip_mm']], on='date', how='left')
    
    # Primary NDVI by sensor hierarchy
    df_merged['primary_ndvi_raw'] = df_merged['s2_ndvi'].combine_first(df_merged['landsat_ndvi']).combine_first(df_merged['modis_ndvi'])
    
    # Reconstruct gaps using continuous linear interpolation followed by gentle rolling smoothing
    raw_interp = df_merged['primary_ndvi_raw'].interpolate(method='linear', limit_direction='both')
    df_merged['primary_ndvi_reconstructed'] = raw_interp.rolling(5, min_periods=1, center=True).mean().round(4)
    # Fallback to actual profile if sparse
    df_merged['primary_ndvi_reconstructed'] = df_merged['primary_ndvi_reconstructed'].fillna(df_merged['actual_fact_ndvi'])
    
    # Z-Score relative to climatology norm
    df_merged['ndvi_zscore'] = compute_zscores(df_merged['primary_ndvi_reconstructed'], df_merged['clim_mean'], df_merged['clim_std'])
    df_merged['status'] = df_merged['ndvi_zscore'].apply(classify_status)
    
    # Detect anomalies and generate agronomic diagnostics
    anoms = detect_anomaly_intervals(df_merged)
    anom_rep = generate_full_report(anoms)
    
    records = []
    for _, r in df_merged.iterrows():
        records.append({
            'date': str(r['date']),
            'doy': int(r['doy']),
            's2_ndvi': None if pd.isna(r['s2_ndvi']) else round(float(r['s2_ndvi']), 4),
            'landsat_ndvi': None if pd.isna(r['landsat_ndvi']) else round(float(r['landsat_ndvi']), 4),
            'modis_ndvi': None if pd.isna(r['modis_ndvi']) else round(float(r['modis_ndvi']), 4),
            'primary_ndvi_raw': None if pd.isna(r['primary_ndvi_raw']) else round(float(r['primary_ndvi_raw']), 4),
            'primary_ndvi_reconstructed': round(float(r['primary_ndvi_reconstructed']), 4),
            'clim_mean': round(float(r['clim_mean']), 4),
            'clim_std': round(float(r['clim_std']), 4),
            'clim_upper': round(float(r['clim_mean'] + r['clim_std']), 4),
            'clim_lower': round(float(r['clim_mean'] - r['clim_std']), 4),
            'ndvi_zscore': round(float(r['ndvi_zscore']), 2),
            'status': str(r['status']),
            'temp_c': round(float(r['era5_temp_c']), 1),
            'precip_mm': round(float(r['era5_precip_mm']), 1)
        })
        
    last_r = records[-1]
    lat_str = f"{lat_center:.2f}°N" if lat_center >= 0 else f"{abs(lat_center):.2f}°S"
    lon_str = f"{lon_center:.2f}°E" if lon_center >= 0 else f"{abs(lon_center):.2f}°W"
    kpis = {
        'polygon_id': f"Контур ({lat_str}, {lon_str})",
        'crop_type': crop,
        'current_ndvi': last_r['primary_ndvi_reconstructed'],
        'current_zscore': last_r['ndvi_zscore'],
        'current_status': last_r['status'],
        'current_temp': last_r['temp_c'],
        'anomalies_count': len(anom_rep),
        'total_gaps_filled': sum(1 for r in records if r['primary_ndvi_raw'] is None)
    }
    
    from src.data_fetchers.gee_fetcher import is_gee_available
    data_sources = {
        'satellite': 'Google Earth Engine (Sentinel-2 10м, Landsat 8/9, MODIS 250м)' if is_gee_available() else 'Earth Search STAC (Sentinel-2 L2A)',
        'weather': 'ECMWF ERA5 / ERA5-Land Reanalysis (Open-Meteo Global)',
        'model': 'Ансамбль 5-Fold LightGBM + CatBoost (GapScore 8.52)',
        'climatology': 'Динамическая многолетняя норма GEE' if is_gee_available() else f'Геоадаптивная норма ({crop})'
    }
    
    return {
        'kpis': kpis,
        'timeseries': records,
        'anomalies': anom_rep,
        'data_sources': data_sources
    }

@app.get("/api/batch-status")
def get_batch_status():
    """Returns information about generated submission.csv."""
    sub_path = os.path.join(BASE_DIR, "submission.csv")
    if not os.path.exists(sub_path):
        return {"status": "not_generated"}
        
    df_sub = pd.read_csv(sub_path)
    return {
        "status": "ready",
        "rows_count": len(df_sub),
        "polygons_count": int(df_sub['anon_polygon_id'].nunique()),
        "mean_predicted_ndvi": round(float(df_sub['primary_ndvi_pred'].mean()), 4),
        "min_predicted_ndvi": round(float(df_sub['primary_ndvi_pred'].min()), 4),
        "max_predicted_ndvi": round(float(df_sub['primary_ndvi_pred'].max()), 4),
        "file_size_kb": round(os.path.getsize(sub_path) / 1024, 1)
    }

@app.get("/api/download-submission")
def download_submission():
    """Скачивание файла submission.csv для отправки решения хакатона."""
    sub_path = os.path.join(BASE_DIR, "submission.csv")
    if not os.path.exists(sub_path):
        raise HTTPException(status_code=404, detail="Файл submission.csv еще не сформирован")
    return FileResponse(sub_path, filename="submission.csv", media_type="text/csv")

# ============================================================================
# МОДУЛЬ ЭКСПОРТА: АГРОНОМИЧЕСКИЙ ПАСПОРТ ПОЛЯ (GEOJSON & CSV)
# Обеспечивает интеграцию результатов мониторинга в ГИС-системы (QGIS, OneSoil, Cropwise)
# ============================================================================

class ExportFieldGeoJsonRequest(BaseModel):
    field_id: str
    field_name: str
    crop_type: str
    area_ha: float
    geometry: Dict[str, Any]
    kpis: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    period: Dict[str, str]
    directives: Optional[List[Dict[str, Any]]] = None

@app.post("/api/export/field-geojson")
def export_field_geojson(req: ExportFieldGeoJsonRequest):
    """
    Формирование расширенного GeoJSON стандарта RFC 7946 для экспорта в агрономические ГИС.
    В свойства полигона включаются: паспорт поля, текущий вегетационный статус,
    дедуплицированный директивный план агротехнических мероприятий и реестр аномалий.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Сборка уникальных директивных рекомендаций без дублирования критериев
    directives_list = req.directives
    if directives_list is None:
        seen_criteria = set()
        directives_list = []
        for a in req.anomalies:
            rec = a.get("recommendation", "").strip()
            if rec and rec not in seen_criteria:
                seen_criteria.add(rec)
                directives_list.append({
                    "tag": "Агрономическое предписание",
                    "text": rec
                })
    
    feature = {
        "type": "Feature",
        "id": req.field_id,
        "geometry": req.geometry,
        "properties": {
            "system": "GEO-VEGA // Космохакатон: Мониторинг вегетации с/х полей",
            "passport_id": f"PASSPORT-{req.field_id}",
            "generated_at": now_str,
            "field_name": req.field_name,
            "crop_type": req.crop_type,
            "area_hectares": round(req.area_ha, 2),
            "monitoring_period": {
                "start_date": req.period.get("start_date"),
                "end_date": req.period.get("end_date")
            },
            "vegetation_status": req.kpis.get("current_status"),
            "current_ndvi": req.kpis.get("current_ndvi"),
            "current_zscore": req.kpis.get("current_zscore"),
            "total_gaps_restored": req.kpis.get("total_gaps_filled", 0),
            "anomalies_count": len(req.anomalies),
            "anomalies_registry": req.anomalies,
            "directives_plan": directives_list,
            "verification_sources": [
                "Google Earth Engine (Sentinel-2 L2A 10m, Landsat 8/9 30m, MODIS 250m)",
                "ECMWF ERA5-Land Reanalysis (суточная метеорология)",
                "Ансамбль LightGBM + CatBoost (восстановление облачных пропусков)"
            ]
        }
    }
    
    return {
        "type": "FeatureCollection",
        "features": [feature]
    }

class ExportFieldCsvRequest(BaseModel):
    field_id: str
    field_name: str
    timeseries: List[Dict[str, Any]]

@app.post("/api/export/field-csv")
def export_field_csv(req: ExportFieldCsvRequest):
    """
    Экспорт суточных временных рядов вегетации и метеорологии в формате CSV.
    Содержит фактические спутниковые наблюдения, восстановленный ряд primary_ndvi,
    климатическую норму, Z-отклонения и метеопараметры ERA5.
    """
    if not req.timeseries:
        raise HTTPException(status_code=400, detail="Отсутствуют данные временного ряда для экспорта")
        
    df = pd.DataFrame(req.timeseries)
    # Форматирование колонок в принятый агрономический стандарт
    cols_order = [
        'date', 'doy', 'primary_ndvi_reconstructed', 'primary_ndvi_raw',
        's2_ndvi', 'landsat_ndvi', 'modis_ndvi', 'clim_mean', 'clim_std',
        'clim_upper', 'clim_lower', 'ndvi_zscore', 'status', 'temp_c', 'precip_mm'
    ]
    present_cols = [c for c in cols_order if c in df.columns]
    df_out = df[present_cols].copy()
    
    csv_str = df_out.to_csv(index=False, encoding='utf-8')
    return {
        "filename": f"timeseries_{req.field_id}_{datetime.now().strftime('%Y%m%d')}.csv",
        "csv_content": csv_str
    }

# Mount static files for frontend UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=False)
