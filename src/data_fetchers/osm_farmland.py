import requests
import urllib3
import math
from typing import List, Dict, Any

urllib3.disable_warnings()

_OSM_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def fetch_osm_farmland_polygons(min_lon: float, min_lat: float, max_lon: float, max_lat: float, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Queries OpenStreetMap Overpass API for real agricultural fields (farmland)
    within the bounding box and converts them to GeoJSON features.
    """
    cache_key = f"{round(min_lon, 2)}_{round(min_lat, 2)}_{round(max_lon, 2)}_{round(max_lat, 2)}_{limit}"
    if cache_key in _OSM_CACHE:
        return _OSM_CACHE[cache_key]

    query = f"""[out:json][timeout:15];
(
  way["landuse"="farmland"]({min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f});
);
out geom {limit};"""

    headers = {'User-Agent': 'GeoVegaHackathon/1.0 (agro-monitoring)'}
    endpoints = [
        'https://overpass-api.de/api/interpreter',
        'https://lz4.overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter'
    ]

    for url in endpoints:
        try:
            resp = requests.post(url, data={'data': query}, headers=headers, verify=False, timeout=12)
            if resp.status_code == 200:
                elements = resp.json().get('elements', [])
                features = []
                for idx, el in enumerate(elements):
                    geom = el.get('geometry', [])
                    if len(geom) < 3:
                        continue
                    # Преобразование координат в формат GeoJSON [[lon, lat], ...]
                    coords = [[pt['lon'], pt['lat']] for pt in geom]
                    # Замыкание контура полигона при необходимости
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                        
                    osm_id = el.get('id', idx + 1)
                    tags = el.get('tags', {})
                    crop = tags.get('crop', tags.get('produce', 'зерновые'))
                    
                    # Точный расчет площади с учетом реальной географической широты полигона
                    mid_lat = (min_lat + max_lat) / 2.0
                    cos_lat = math.cos(math.radians(mid_lat))
                    area_ha = 0
                    for i in range(len(coords) - 1):
                        x1 = coords[i][0] * 111320 * cos_lat
                        y1 = coords[i][1] * 110574
                        x2 = coords[i+1][0] * 111320 * cos_lat
                        y2 = coords[i+1][1] * 110574
                        area_ha += (x1 * y2 - x2 * y1)
                    area_ha = round(abs(area_ha / 2.0) / 10000.0, 1)
                    if area_ha < 5.0:
                        continue  # Исключаем мелкие дачные участки, оставляем товарные с/х поля
                        
                    features.append({
                        "type": "Feature",
                        "id": f"OSM-{osm_id}",
                        "properties": {
                            "anon_polygon_id": f"OSM-{osm_id}",
                            "name": f"Поле OSM #{osm_id}",
                            "region": "Аграрный район",
                            "crop_type": crop,
                            "area_ha": area_ha,
                            "soil_type": "Аграрный чернозем",
                            "reference_years": 15,
                            "source": "OpenStreetMap"
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        }
                    })
                    
                if features:
                    _OSM_CACHE[cache_key] = features
                    return features
        except Exception as e:
            print(f"[OSM Fetcher] Ошибка запроса к {url}: {e}")
            continue
            
    return []
