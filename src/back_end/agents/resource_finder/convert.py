import geopandas as gpd
import json

# Downloaded from TIGER/Line: tl_2023_us_county.shp
gdf = gpd.read_file("tl_2023_us_county.shp")

out = []
for _, row in gdf.iterrows():
    centroid = row.geometry.centroid
    out.append({
        "name": row["NAME"],        # County name
        "state": row["STATEFP"],    # State FIPS (you can map to abbreviation if needed)
        "fips": row["STATEFP"] + row["COUNTYFP"],
        "lat": centroid.y,
        "lon": centroid.x
    })

with open("us_county_centroids.json", "w") as f:
    json.dump(out, f, indent=2)
