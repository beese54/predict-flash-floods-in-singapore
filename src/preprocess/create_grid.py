"""
E1 — Generate Singapore 1km x 1km grid GeoJSON.
Output: data/processed/singapore_grid.geojson
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Approximate metres per degree at Singapore's latitude (~1.3°N)
METRES_PER_DEG_LAT = 110_574
METRES_PER_DEG_LON = 111_320 * np.cos(np.radians(1.3))


def _deg_per_km_lat() -> float:
    return 1000 / METRES_PER_DEG_LAT


def _deg_per_km_lon() -> float:
    return 1000 / METRES_PER_DEG_LON


def build_grid(config: dict, output_path: Path) -> None:
    sg = config["singapore"]
    lat_min, lat_max = sg["lat_min"], sg["lat_max"]
    lon_min, lon_max = sg["lon_min"], sg["lon_max"]

    dlat = _deg_per_km_lat()
    dlon = _deg_per_km_lon()

    lats = np.arange(lat_min, lat_max, dlat)
    lons = np.arange(lon_min, lon_max, dlon)

    cells = []
    cell_id = 0
    for lat in lats:
        for lon in lons:
            geom = box(lon, lat, lon + dlon, lat + dlat)
            cells.append({
                "grid_cell_id": f"SG-GRID-{cell_id:05d}",
                "lat_centroid": round(lat + dlat / 2, 6),
                "lon_centroid": round(lon + dlon / 2, 6),
                "geometry": geom,
            })
            cell_id += 1

    gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")

    # Attempt to clip to Singapore land boundary using a simple convex approximation
    # (A detailed boundary can be downloaded from data.gov.sg OneMap, but as a robust
    # fallback we use the bounding box itself — the grid will include some sea cells.)
    log.info(f"Created {len(gdf)} grid cells (pre-clip) covering bounding box")
    log.info("Note: clip to Singapore land polygon not applied — sea cells included.")
    log.info("To apply precise land clip: download Singapore boundary from data.gov.sg OneMap")
    log.info("and call: gdf = gdf[gdf.intersects(sg_boundary)]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    log.info(f"Saved {len(gdf)} grid cells → {output_path}")


if __name__ == "__main__":
    config = get_config()
    root = project_root()
    out = root / "data" / "processed" / "singapore_grid.geojson"
    build_grid(config, out)
