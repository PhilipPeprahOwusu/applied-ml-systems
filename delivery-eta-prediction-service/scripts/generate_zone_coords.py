import pandas as pd
import geopandas as gpd
import json
import urllib.request
import zipfile
import os

print("Downloading NYC Taxi Zone Shapefile...")
url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
urllib.request.urlretrieve(url, "taxi_zones.zip")

with zipfile.ZipFile("taxi_zones.zip", 'r') as zip_ref:
    zip_ref.extractall("taxi_zones")

print("Calculating zone centroids...")
# Load the shapefile
sf = gpd.read_file("taxi_zones/taxi_zones/taxi_zones.shp")

# Convert the geometry from NYC local projection to standard GPS projection (WGS84 / EPSG:4326)
sf = sf.to_crs("EPSG:4326")

# Calculate the center point (centroid) of each zone polygon
sf['lon'] = sf.geometry.centroid.x
sf['lat'] = sf.geometry.centroid.y

# Format it for our React frontend
zones_list = []
for index, row in sf.iterrows():
    zones_list.append({
        "id": int(row['LocationID']),
        "name": row['zone'],
        "borough": row['borough'],
        "lat": round(row['lat'], 4),
        "lng": round(row['lon'], 4)
    })

# Save to a JSON file inside the Next.js public directory so we can fetch it, 
# or just save it directly to src for importing.
output_path = "frontend/src/app/nyc_zones.json"
with open(output_path, 'w') as f:
    json.dump(zones_list, f, indent=2)

print(f"Successfully generated {len(zones_list)} zones with coordinates at {output_path}")

# Cleanup
os.remove("taxi_zones.zip")
import shutil
shutil.rmtree("taxi_zones")
