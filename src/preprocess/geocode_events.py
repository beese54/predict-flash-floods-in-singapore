"""
E3 — Geocode verified flood events to lat/lon and assign to 1km grid cells.
Input:  data/processed/verified_events.json + data/processed/sg_locations.json
        + data/processed/singapore_grid.geojson
Output: data/processed/flood_events.parquet
"""
import json
import logging
import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from shapely.geometry import Point

from src.utils import project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SG_BBOX = {"lat_min": 1.15, "lat_max": 1.48, "lon_min": 103.60, "lon_max": 104.05}
GEOCODER_DELAY = 1.5


def _load_location_ref(path: Path) -> dict[str, tuple[float, float]]:
    """Build a code → (lat, lon) lookup from sg_locations.json using known centroids."""
    # Known centroids for URA planning areas (approximate)
    _KNOWN = {
        "Orchard Road, Singapore": (1.3048, 103.8318),
        "Lucky Plaza, Singapore": (1.3043, 103.8317),
        "Bukit Timah, Singapore": (1.3294, 103.8021),
        "Bukit Timah Canal, Singapore": (1.3400, 103.8050),
        "Maxwell Road, Singapore": (1.2795, 103.8453),
        "Ang Mo Kio, Singapore": (1.3691, 103.8454),
        "Yio Chu Kang Road, Singapore": (1.3838, 103.8448),
        "Boon Keng, Singapore": (1.3198, 103.8624),
        "Bendemeer, Singapore": (1.3192, 103.8669),
        "Toa Payoh, Singapore": (1.3343, 103.8563),
        "Hougang, Singapore": (1.3712, 103.8930),
        "Tampines, Singapore": (1.3496, 103.9568),
        "Jurong West, Singapore": (1.3404, 103.7090),
        "Woodlands, Singapore": (1.4367, 103.7864),
        "Sengkang, Singapore": (1.3914, 103.8950),
        "Punggol, Singapore": (1.4043, 103.9021),
        "Changi, Singapore": (1.3644, 103.9915),
        "Bedok, Singapore": (1.3236, 103.9273),
    }
    return {v: k for k, v in _KNOWN.items()}, _KNOWN


_SIMPLIFY_RE = [
    re.compile(r"\s+from\s+.+?\s+to\s+.+", re.IGNORECASE),       # "from X to Y"
    re.compile(r"\s*\[[^\]]*\]", re.IGNORECASE),                   # "[anything]" incl. time tags like "[15:46 hours]"
    re.compile(r"\s*\([^)]+\)", re.IGNORECASE),                    # "(anything)"
    re.compile(r"\s+(near|towards|before|between|via)\s+.+", re.IGNORECASE),
    re.compile(r"\s+(slip road|slip rd)\s+.+", re.IGNORECASE),
    re.compile(r",\s*Singapore$", re.IGNORECASE),
]


def _simplify_for_geocoding(loc: str) -> str:
    """Strip 'from X to Y', parentheticals, and directional qualifiers."""
    for pattern in _SIMPLIFY_RE:
        loc = pattern.sub("", loc)
    return loc.strip()


def _geocode_nominatim(geocoder: Nominatim, location_str: str) -> tuple[float, float] | None:
    """Geocode a location string with Singapore bias via Nominatim."""
    # Append Singapore if not already present
    query = location_str if "singapore" in location_str.lower() else f"{location_str}, Singapore"
    try:
        result = geocoder.geocode(query, country_codes="SG", timeout=10)
        if result:
            return result.latitude, result.longitude
    except GeocoderTimedOut:
        log.warning(f"  Geocoder timeout for: {query}")
    except Exception as e:
        log.warning(f"  Geocoder error for {query}: {e}")
    return None


def _in_singapore(lat: float, lon: float) -> bool:
    return (SG_BBOX["lat_min"] <= lat <= SG_BBOX["lat_max"] and
            SG_BBOX["lon_min"] <= lon <= SG_BBOX["lon_max"])


def geocode_events(root: Path) -> None:
    verified_path = root / "data" / "processed" / "verified_events.json"
    if not verified_path.exists():
        raise FileNotFoundError(
            f"{verified_path} not found.\n"
            "Complete the manual verification step in notebooks/02_label_generation.ipynb first."
        )

    loc_ref_path = root / "data" / "processed" / "sg_locations.json"
    grid_path = root / "data" / "processed" / "singapore_grid.geojson"

    with open(verified_path, encoding="utf-8") as f:
        verified = json.load(f)
    with open(loc_ref_path, encoding="utf-8") as f:
        loc_ref = {e["name"]: e["code"] for e in json.load(f)}

    # Known centroids for fast lookup (planning areas + flood-prone Telegram locations)
    _known_centroids = {
        # Planning areas / districts
        "Orchard Road, Singapore": (1.3048, 103.8318),
        "Lucky Plaza, Singapore": (1.3043, 103.8317),
        "Bukit Timah, Singapore": (1.3294, 103.8021),
        "Bukit Timah Canal, Singapore": (1.3400, 103.8050),
        "Maxwell Road, Singapore": (1.2795, 103.8453),
        "Ang Mo Kio, Singapore": (1.3691, 103.8454),
        "Ang Mo Kio Avenue 3, Singapore": (1.3691, 103.8454),
        "Yio Chu Kang Road, Singapore": (1.3838, 103.8448),
        "Boon Keng, Singapore": (1.3198, 103.8624),
        "Bendemeer, Singapore": (1.3192, 103.8669),
        "Toa Payoh, Singapore": (1.3343, 103.8563),
        "Hougang, Singapore": (1.3712, 103.8930),
        "Tampines, Singapore": (1.3496, 103.9568),
        "Jurong West, Singapore": (1.3404, 103.7090),
        "Woodlands, Singapore": (1.4367, 103.7864),
        "Sengkang, Singapore": (1.3914, 103.8950),
        "Punggol, Singapore": (1.4043, 103.9021),
        "Changi, Singapore": (1.3644, 103.9915),
        "Bedok, Singapore": (1.3236, 103.9273),
        "Lorong Buangkok, Singapore": (1.3831, 103.8707),
        # Recurring Telegram FLASH_FLOOD locations
        "Yishun Avenue 7, Singapore": (1.4285, 103.8379),
        "Yishun Ave 7, Singapore": (1.4285, 103.8379),
        "Jurong Town Hall Road, Singapore": (1.3330, 103.7403),
        "Woollerton Drive, Singapore": (1.3200, 103.8073),
        "Coronation Walk, Singapore": (1.3196, 103.8042),
        "King's Road, Singapore": (1.3180, 103.8035),
        "Kings Road, Singapore": (1.3180, 103.8035),
        "Pandan Road, Singapore": (1.3183, 103.7456),
        "Pesawat Drive, Singapore": (1.3248, 103.6814),
        "Sims Avenue East, Singapore": (1.3137, 103.9047),
        "Upper East Coast Road, Singapore": (1.3249, 103.9354),
        "Boon Lay Avenue, Singapore": (1.3452, 103.7070),
        "Boon Lay Ave, Singapore": (1.3452, 103.7070),
        "Bukit Timah Road, Singapore": (1.3440, 103.7789),
        "Bt Timah Road, Singapore": (1.3440, 103.7789),
        "Punggol Way, Singapore": (1.4050, 103.9070),
        "KPE, Singapore": (1.3600, 103.8900),
    }

    # Load manual annotations (highest priority — user-verified locations)
    ann_path = root / "data" / "processed" / "manual_annotations.json"
    _manual: dict[tuple, tuple[float, float]] = {}      # (source, src_id, orig_loc) → (lat, lon)
    _manual_ann: dict[tuple, dict] = {}                  # same key → full annotation dict (for amended name)
    _dismissed: set[tuple] = set()                       # (source, src_id, orig_loc) → skip entirely
    _extra_annotations: list[dict] = []                  # is_extra=True entries → emit as synthetic rows
    if ann_path.exists():
        for a in json.loads(ann_path.read_text(encoding="utf-8")):
            src = a["source"]
            sid = a["source_row_id"]
            orig = a.get("original_location_str") or a.get("location_str", "")
            key = (src, sid, orig)
            if a.get("is_extra"):
                if a.get("lat") is not None and a.get("event_type") not in (None, "DISMISSED"):
                    _extra_annotations.append(a)
            elif a.get("event_type") == "DISMISSED":
                _dismissed.add(key)
            elif a.get("lat") is not None:
                _manual[key] = (a["lat"], a["lon"])
                _manual_ann[key] = a
        log.info(
            f"Loaded {len(_manual)} manual annotations, {len(_dismissed)} dismissed, "
            f"{len(_extra_annotations)} extra from {ann_path.name}"
        )

    grid_gdf = gpd.read_file(grid_path)

    geocoder = Nominatim(user_agent="sg_flash_flood_project")

    rows = []
    event_id = 0
    for item in verified:
        if not item.get("verified"):
            continue

        dates = item.get("flood_dates", [])
        locations = item.get("flooded_locations", [])
        source = item.get("source", "unknown")
        src_id = item.get("source_row_id")
        message_type = item.get("message_type")       # present for Telegram only
        event_datetime = item.get("event_datetime")   # present for Telegram only

        for date_str in dates:
            for loc_str in locations:
                # Skip locations the user explicitly dismissed
                if (source, src_id, loc_str) in _dismissed:
                    log.info(f"  Skipping dismissed: {loc_str}")
                    continue

                # Manual annotation has highest priority
                man_key = (source, src_id, loc_str)
                if man_key in _manual:
                    coords = _manual[man_key]
                    display_loc = _manual_ann[man_key].get("location_str") or loc_str
                else:
                    coords = None
                    display_loc = loc_str

                # Try known centroids
                if coords is None:
                    coords = _known_centroids.get(loc_str)
                if coords is None:
                    # Try with "Singapore" appended
                    loc_sg = loc_str if "singapore" in loc_str.lower() else f"{loc_str}, Singapore"
                    coords = _known_centroids.get(loc_sg)

                if coords is None:
                    # Fallback to Nominatim — try full string first, then simplified
                    loc_sg = loc_str if "singapore" in loc_str.lower() else f"{loc_str}, Singapore"
                    coords = _geocode_nominatim(geocoder, loc_sg)
                    time.sleep(GEOCODER_DELAY)

                if coords is None:
                    # Try simplified query (strip "from X to Y", parentheticals, etc.)
                    simplified = _simplify_for_geocoding(loc_sg)
                    if simplified != loc_sg:
                        simplified_sg = simplified if "singapore" in simplified.lower() else f"{simplified}, Singapore"
                        coords = _geocode_nominatim(geocoder, simplified_sg)
                        time.sleep(GEOCODER_DELAY)

                if coords is None:
                    log.warning(f"  Could not geocode: {loc_str}")
                    continue

                lat, lon = coords
                if not _in_singapore(lat, lon):
                    log.warning(f"  Out of Singapore bounds: {loc_str} → ({lat}, {lon})")
                    continue

                # Assign to grid cell
                pt = Point(lon, lat)
                matches = grid_gdf[grid_gdf.contains(pt)]
                grid_cell_id = matches.iloc[0]["grid_cell_id"] if len(matches) > 0 else None

                rows.append({
                    "event_id": event_id,
                    "source": source,
                    "source_row_id": src_id,
                    "date": date_str,
                    "event_datetime": event_datetime,  # ISO string or None
                    "message_type": message_type,      # str or None
                    "location_str": display_loc,
                    "lat": lat,
                    "lon": lon,
                    "grid_cell_id": grid_cell_id,
                })
                event_id += 1

    # Append extra user-added locations (is_extra=True annotations)
    for a in _extra_annotations:
        pt = Point(a["lon"], a["lat"])
        matches = grid_gdf[grid_gdf.contains(pt)]
        grid_cell_id = matches.iloc[0]["grid_cell_id"] if len(matches) > 0 else None
        rows.append({
            "event_id":      event_id,
            "source":        a["source"],
            "source_row_id": a["source_row_id"],
            "date":          "",
            "event_datetime": None,
            "message_type":  None,
            "location_str":  a.get("location_str", ""),
            "lat":           a["lat"],
            "lon":           a["lon"],
            "grid_cell_id":  grid_cell_id,
        })
        event_id += 1
    if _extra_annotations:
        log.info(f"  Added {len(_extra_annotations)} extra user-annotated locations")

    df = pd.DataFrame(rows)
    out_path = root / "data" / "processed" / "flood_events.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"Geocoded {len(df)} (event, location) pairs from {sum(1 for i in verified if i.get('verified'))} verified events")
    log.info(f"Saved → {out_path}")


def geocode_risk_events(root: Path) -> None:
    """Geocode FLOOD_RISK events from extracted_events.json → flood_risk_events.parquet.
    These are used as class-1 (precursor) labels in the ordinal ML model.
    """
    extracted_path = root / "data" / "processed" / "extracted_events.json"
    grid_path = root / "data" / "processed" / "singapore_grid.geojson"

    if not extracted_path.exists():
        log.warning("extracted_events.json not found — skipping FLOOD_RISK geocoding")
        return

    with open(extracted_path, encoding="utf-8") as f:
        extracted = json.load(f)

    risk_events = [
        e for e in extracted
        if e.get("message_type") == "FLOOD_RISK" and e.get("flooded_locations")
    ]
    if not risk_events:
        log.info("No FLOOD_RISK events found in extracted_events.json")
        return

    grid_gdf = gpd.read_file(grid_path)
    geocoder = Nominatim(user_agent="sg_flash_flood_project_risk")

    _known: dict[str, tuple[float, float]] = {
        "Enterprise Road, Singapore": (1.3342, 103.7073),
        "Upper Jurong Road, Singapore": (1.3370, 103.7133),
        "Tanjong Pagar Road, Singapore": (1.2787, 103.8442),
        "Teck Whye Lane, Singapore": (1.3607, 103.7597),
        "Bukit Timah Road, Singapore": (1.3440, 103.7789),
        "Orchard Road, Singapore": (1.3048, 103.8318),
        "Jurong East, Singapore": (1.3329, 103.7436),
        "Jurong West, Singapore": (1.3404, 103.7090),
        "Yishun, Singapore": (1.4285, 103.8380),
        "Ang Mo Kio, Singapore": (1.3691, 103.8454),
        "Tampines, Singapore": (1.3496, 103.9568),
        "Bedok, Singapore": (1.3236, 103.9273),
        "Woodlands, Singapore": (1.4367, 103.7864),
        "Sengkang, Singapore": (1.3914, 103.8950),
        "Punggol, Singapore": (1.4043, 103.9021),
        "Hougang, Singapore": (1.3712, 103.8930),
        "Clementi, Singapore": (1.3162, 103.7649),
        "Pasir Ris, Singapore": (1.3730, 103.9490),
        "Choa Chu Kang, Singapore": (1.3855, 103.7450),
        "Bukit Batok, Singapore": (1.3590, 103.7637),
    }

    rows = []
    for ev_idx, ev in enumerate(risk_events):
        event_dt_str = ev.get("event_datetime")
        date_str = ev.get("date", "")

        for loc_str in ev["flooded_locations"]:
            # Strip time tags and brackets from location strings (e.g. "[15:46 hours]")
            clean_loc = _simplify_for_geocoding(loc_str)
            loc_sg = clean_loc if "singapore" in clean_loc.lower() else f"{clean_loc}, Singapore"

            coords = _known.get(loc_sg) or _known.get(loc_str)
            if coords is None:
                coords = _geocode_nominatim(geocoder, loc_sg)
                time.sleep(GEOCODER_DELAY)
            if coords is None:
                simplified = _simplify_for_geocoding(loc_sg)
                if simplified != loc_sg:
                    coords = _geocode_nominatim(geocoder, simplified)
                    time.sleep(GEOCODER_DELAY)
            if coords is None:
                log.warning(f"  Cannot geocode FLOOD_RISK location: {loc_str}")
                continue

            lat, lon = coords
            if not _in_singapore(lat, lon):
                log.warning(f"  Out of SG bounds: {loc_str} → ({lat:.4f}, {lon:.4f})")
                continue

            pt = Point(lon, lat)
            matches = grid_gdf[grid_gdf.contains(pt)]
            grid_cell_id = matches.iloc[0]["grid_cell_id"] if len(matches) > 0 else None

            rows.append({
                "event_id": ev_idx,
                "source": "pub_telegram",
                "source_row_id": ev.get("source_row_id"),
                "date": date_str,
                "event_datetime": event_dt_str,
                "message_type": "FLOOD_RISK",
                "location_str": loc_sg,
                "lat": lat,
                "lon": lon,
                "grid_cell_id": grid_cell_id,
            })

    if rows:
        df = pd.DataFrame(rows)
        out_path = root / "data" / "processed" / "flood_risk_events.parquet"
        df.to_parquet(out_path, index=False)
        log.info(f"FLOOD_RISK: geocoded {len(df)} location rows from {len(risk_events)} events → {out_path.name}")
    else:
        log.warning("FLOOD_RISK geocoding: no rows produced")


if __name__ == "__main__":
    geocode_events(project_root())
    geocode_risk_events(project_root())
