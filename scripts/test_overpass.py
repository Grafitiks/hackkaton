import requests
import urllib3

urllib3.disable_warnings()

query = """[out:json][timeout:15];
(
  way["landuse"="farmland"](53.15,50.15,53.30,50.35);
);
out geom 5;"""

try:
    headers = {'User-Agent': 'GeoVegaHackathon/1.0 (agro-monitoring)'}
    resp = requests.post('https://overpass-api.de/api/interpreter', data={'data': query}, headers=headers, verify=False, timeout=15)
    print("Overpass Status:", resp.status_code)
    elements = resp.json().get('elements', [])
    print("Farmland polygons found:", len(elements))
    if elements:
        print("Sample tags:", elements[0].get('tags', {}))
        print("Sample geometry points count:", len(elements[0].get('geometry', [])))
except Exception as e:
    print("Error:", e)
