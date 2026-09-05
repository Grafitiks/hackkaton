import requests
import urllib3
from typing import List, Dict, Any

urllib3.disable_warnings()

def fetch_osm_farmland_polygons(min_lon: float, min_lat: float, max_lon: float, max_lat: float, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Queries OpenStreetMap Overpass API for real agricultural fields (farmland)
    within the bounding box and converts them to GeoJSON features.
    """
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
                    # Convert to GeoJSON [[lon, lat], ...]
                    coords = [[pt['lon'], pt['lat']] for pt in geom]
                    # Close ring if not closed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                        
                    osm_id = el.get('id', idx + 1)
                    tags = el.get('tags', {})
                    crop = tags.get('crop', tags.get('produce', 'зерновые'))
                    
                    # Calculate area
                    area_ha = 0
                    for i in range(len(coords) - 1):
                        x1 = coords[i][0] * 111320 * 0.6  # approximate cos(53)
                        y1 = coords[i][1] * 110574
                        x2 = coords[i+1][0] * 111320 * 0.6
                        y2 = coords[i+1][1] * 110574
                        area_ha += (x1 * y2 - x2 * y1)
                    area_ha = round(abs(area_ha / 2.0) / 10000.0, 1)
                    if area_ha < 5.0:
                        continue  # Skip tiny backyard gardens, keep real agricultural fields
                        
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
                    return features
        except Exception as e:
            print(f"[OSM Fetcher] Ошибка запроса к {url}: {e}")
            continue
            
    return []
